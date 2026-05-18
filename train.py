"""
Pipeline de fine-tuning do RepVGG-A0 para classificação de emoções faciais.

Estratégia em duas fases:
  Fase 1 — congela o backbone, treina apenas a camada linear (cabeça).
  Fase 2 — descongela tudo e realiza fine-tuning completo com LR menor.

Uso:
  python train.py
  python train.py --epochs-head 10 --epochs-full 30 --device cuda
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from lib.repvgg import create_RepVGG_A0
from main import (
    DEFAULT_MODEL_CLASSES,
    LABEL_ALIASES,
    EmotionFolderDataset,
    build_transform,
    extract_state_dict,
    strip_module_prefix,
)

SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning RepVGG para emocoes.")
    p.add_argument("--dataset", type=Path, default=Path("dataset"))
    p.add_argument("--weights", type=Path, default=Path("lib/core/weights/repvgg.pth"))
    p.add_argument("--output-weights", type=Path, default=Path("lib/core/weights/repvgg_finetuned.pth"))
    p.add_argument("--output-dir", type=Path, default=Path("reports"))
    p.add_argument("--val-split", type=float, default=0.2, help="Fração do dataset reservada para validação.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs-head", type=int, default=10, help="Épocas treinando apenas a cabeça linear.")
    p.add_argument("--epochs-full", type=int, default=30, help="Épocas com fine-tuning completo do backbone.")
    p.add_argument("--lr-head", type=float, default=1e-3, help="LR para a cabeça linear (fase 1).")
    p.add_argument("--lr-full", type=float, default=1e-4, help="LR para fine-tuning completo (fase 2).")
    p.add_argument("--device", type=str, default=None, help="Dispositivo: 'cuda' ou 'cpu'. Auto-detectado se omitido.")
    p.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_MODEL_CLASSES),
        help="Ordem das classes da camada de saída (deve ser a mesma do treino original).",
    )
    return p.parse_args()


def build_train_transform(image_size: int) -> transforms.Compose:
    """Augmentações durante o treino para reduzir overfitting."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def stratified_split(
    dataset: EmotionFolderDataset, val_fraction: float, seed: int
) -> Tuple[Subset, Subset]:
    """Divide o dataset mantendo proporção de classes em treino e validação."""
    labels = [s["label"] for s in dataset.samples]
    indices = list(range(len(dataset)))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(indices, labels))
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def compute_class_weights(dataset: EmotionFolderDataset, model_classes: Tuple[str, ...], device: torch.device) -> torch.Tensor:
    """Pesos inversamente proporcionais à raiz da frequência de cada classe."""
    counts = {cls: 0 for cls in model_classes}
    for s in dataset.samples:
        if s["label"] in counts:
            counts[s["label"]] += 1

    total = sum(counts.values())
    weights = []
    for cls in model_classes:
        cnt = counts[cls]
        # sqrt(total/cnt) — moderado: evita pesos extremos para classes raras
        weights.append(np.sqrt(total / max(cnt, 1)))

    w = torch.tensor(weights, dtype=torch.float32)
    return (w / w.mean()).to(device)  # normaliza: média=1


def load_model(weights_path: Path, device: torch.device) -> nn.Module:
    model = create_RepVGG_A0(deploy=True)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(ckpt)
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def freeze_backbone(model: nn.Module):
    """Congela todo o modelo exceto a camada linear (cabeça)."""
    for name, param in model.named_parameters():
        if not name.startswith("linear"):
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def labels_to_indices(labels_str: List[str], model_classes: Tuple[str, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [model_classes.index(l) if l in model_classes else 0 for l in labels_str],
        dtype=torch.long,
        device=device,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    model_classes: Tuple[str, ...],
    device: torch.device,
    criterion: nn.Module,
    optimizer=None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, true_labels, *_ in loader:
            images = images.to(device)
            targets = labels_to_indices(list(true_labels), model_classes, device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)  # logits em treino; probabilities em eval
            if is_train:
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            else:
                # Em eval o modelo retorna softmax; usa NLLLoss
                log_probs = torch.log(outputs.clamp(min=1e-8))
                loss = nn.functional.nll_loss(log_probs, targets, weight=criterion.weight)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_classes: Tuple[str, ...],
    device: torch.device,
    criterion: nn.Module,
    lr: float,
    epochs: int,
    phase_name: str,
    best_val_acc: float,
    output_weights: Path,
) -> float:
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    print(f"\n{'='*60}")
    print(f"  {phase_name}  |  LR={lr}  |  Épocas={epochs}")
    print(f"{'='*60}")
    print(f"{'Época':>6}  {'Loss Treino':>12}  {'Acc Treino':>11}  {'Loss Val':>9}  {'Acc Val':>8}  {'Melhor':>7}")
    print(f"{'-'*60}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, model_classes, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, model_classes, device, criterion)
        scheduler.step()

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_weights)

        marker = "  ✓ salvo" if improved else ""
        print(
            f"{epoch:>6}  {train_loss:>12.4f}  {train_acc:>10.4f}  "
            f"{val_loss:>9.4f}  {val_acc:>8.4f}{marker}"
        )

    return best_val_acc


def main():
    set_seed(SEED)
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Dispositivo: {device}")

    model_classes = tuple(args.classes)

    # ── Datasets ──────────────────────────────────────────────────────
    train_transform = build_train_transform(args.image_size)
    val_transform = build_transform(args.image_size)

    full_dataset_train = EmotionFolderDataset(args.dataset, transform=train_transform, label_aliases=LABEL_ALIASES)
    full_dataset_val = EmotionFolderDataset(args.dataset, transform=val_transform, label_aliases=LABEL_ALIASES)

    train_subset, val_subset = stratified_split(full_dataset_train, args.val_split, SEED)

    # Aplica os índices do split também ao dataset com transform de validação
    val_indices = val_subset.indices
    train_indices = train_subset.indices
    val_dataset = Subset(full_dataset_val, val_indices)
    train_dataset = Subset(full_dataset_train, train_indices)

    print(f"Amostras treino : {len(train_dataset)}")
    print(f"Amostras val    : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    # ── Pesos de classe para balancear o loss ─────────────────────────
    class_weights = compute_class_weights(full_dataset_train, model_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\nPesos de classe (loss):")
    for cls, w in zip(model_classes, class_weights.cpu().tolist()):
        print(f"  {cls:12s}: {w:.3f}")

    # ── Modelo ────────────────────────────────────────────────────────
    model = load_model(args.weights, device)
    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    # ── Fase 1: Apenas a cabeça ───────────────────────────────────────
    if args.epochs_head > 0:
        freeze_backbone(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nFase 1 — parâmetros treináveis: {trainable:,} (apenas linear)")
        best_val_acc = train_phase(
            model, train_loader, val_loader, model_classes, device, criterion,
            lr=args.lr_head, epochs=args.epochs_head,
            phase_name="FASE 1 — Cabeça Linear",
            best_val_acc=best_val_acc,
            output_weights=args.output_weights,
        )

    # ── Fase 2: Fine-tuning completo ──────────────────────────────────
    if args.epochs_full > 0:
        # Carrega o melhor checkpoint da fase 1 (se existir)
        if args.output_weights.exists():
            ckpt = torch.load(args.output_weights, map_location=device, weights_only=False)
            model.load_state_dict(ckpt)

        unfreeze_all(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nFase 2 — parâmetros treináveis: {trainable:,} (backbone completo)")
        best_val_acc = train_phase(
            model, train_loader, val_loader, model_classes, device, criterion,
            lr=args.lr_full, epochs=args.epochs_full,
            phase_name="FASE 2 — Fine-tuning Completo",
            best_val_acc=best_val_acc,
            output_weights=args.output_weights,
        )

    print(f"\n{'='*60}")
    print(f"Melhor val accuracy: {best_val_acc:.4f}")
    print(f"Pesos salvos em   : {args.output_weights}")
    print(f"\nPara avaliar com os novos pesos:")
    print(f"  python main.py --weights {args.output_weights} --tta")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
