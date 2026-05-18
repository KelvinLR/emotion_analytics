import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

# ja tenho peso da yolo especifico
# fzr treinamento em cascata 
# e: modelo pre treinado
# s: modelo com fine tuning

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from lib.repvgg import create_RepVGG_A0


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_MODEL_CLASSES = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
LABEL_ALIASES = {
    "angry": "anger",
    "disgusted": "disgust",
}


class EmotionFolderDataset(Dataset):
    def __init__(self, dataset_dir: Path, transform=None, label_aliases: Dict[str, str] = None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.label_aliases = label_aliases or {}
        self.samples = self._collect_samples()

        if not self.samples:
            raise ValueError(f"Nenhuma imagem encontrada em: {dataset_dir}")

        self.folder_classes = sorted({sample["folder_label"] for sample in self.samples})
        self.labels = sorted({sample["label"] for sample in self.samples})

    def _collect_samples(self) -> List[Dict[str, str]]:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset nao encontrado: {self.dataset_dir}")

        samples = []
        class_dirs = sorted(path for path in self.dataset_dir.iterdir() if path.is_dir())
        for class_dir in class_dirs:
            folder_label = class_dir.name
            label = self.label_aliases.get(folder_label, folder_label)
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append(
                        {
                            "path": str(image_path),
                            "file_name": image_path.name,
                            "folder_label": folder_label,
                            "label": label,
                        }
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample["label"], sample["folder_label"], sample["path"], sample["file_name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de avaliacao do classificador RepVGG de emocoes."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Pasta raiz do dataset.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("lib/core/weights/repvgg.pth"),
        help="Arquivo .pth com os pesos treinados do RepVGG.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Pasta de saida.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch.")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Workers do DataLoader.")
    parser.add_argument("--image-size", type=int, default=224, help="Tamanho final da imagem.")
    parser.add_argument(
        "--centercrop",
        action="store_true",
        default=False,
        help=(
            "Usa Resize(resize-size) + CenterCrop(image-size) em vez de Resize direto. "
            "Ative apenas se o modelo foi treinado com esse pipeline."
        ),
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=256,
        help="Resize antes do CenterCrop (usado apenas com --centercrop).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_MODEL_CLASSES),
        help="Ordem das classes da camada de saida do modelo.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=True,
        help="Test Time Augmentation: media de predicoes com flip e rotacoes (ativo por padrao).",
    )
    parser.add_argument(
        "--no-tta",
        dest="tta",
        action="store_false",
        help="Desativa TTA.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de inferencia: 'cuda', 'cpu'. Auto-detectado se omitido.",
    )
    return parser.parse_args()


def build_transform(image_size: int, use_centercrop: bool = False, resize_size: int = 256):
    if use_centercrop:
        spatial = [transforms.Resize(resize_size), transforms.CenterCrop(image_size)]
    else:
        # Redimensionamento direto preserva toda a região facial sem corte lateral.
        # Preferível para datasets de recortes de face (ex: FER2013 48x48).
        spatial = [transforms.Resize((image_size, image_size))]
    return transforms.Compose(
        [
            *spatial,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_repvgg(weights_path: Path, device: torch.device) -> torch.nn.Module:
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesos nao encontrados: {weights_path}")

    model = create_RepVGG_A0(deploy=True)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _to_probs(outputs: torch.Tensor) -> torch.Tensor:
    """Garante que a saída do modelo seja uma distribuição de probabilidades."""
    row_sums = outputs.sum(dim=1)
    is_prob = bool(
        torch.all(outputs >= 0)
        and torch.all(outputs <= 1)
        and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    )
    return outputs if is_prob else torch.softmax(outputs, dim=1)


def _tta_variants(images: torch.Tensor) -> List[torch.Tensor]:
    """Gera variantes augmentadas para TTA. Usa somente flip e rotações leves."""
    return [
        torch.flip(images, dims=[3]),           # espelho horizontal
        TF.rotate(images, angle=5, fill=0),     # rotação +5°
        TF.rotate(images, angle=-5, fill=0),    # rotação -5°
    ]


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    model_classes: Sequence[str],
    device: torch.device,
    tta: bool = True,
) -> pd.DataFrame:
    rows = []

    for images, true_labels, folder_labels, paths, file_names in dataloader:
        images = images.to(device)

        # Predição base
        probabilities = _to_probs(model(images))

        # TTA: média com variantes augmentadas
        if tta:
            for variant in _tta_variants(images):
                probabilities = probabilities + _to_probs(model(variant))
            probabilities = probabilities / (1 + len(_tta_variants(images)))

        confidences, predicted_indices = torch.max(probabilities, dim=1)

        probabilities = probabilities.cpu().tolist()
        predicted_indices = predicted_indices.cpu().tolist()
        confidences = confidences.cpu().tolist()

        for idx, probs in enumerate(probabilities):
            predicted_class = model_classes[predicted_indices[idx]]
            row = {
                "image_path": paths[idx],
                "image_name": file_names[idx],
                "true_class": folder_labels[idx],
                "true_label": true_labels[idx],
                "predicted_class": predicted_class,
                "confidence": confidences[idx],
            }
            row["probabilities_json"] = json.dumps(dict(zip(model_classes, probs)), ensure_ascii=True)
            for class_name, probability in zip(model_classes, probs):
                row[f"prob_{class_name}"] = probability
            rows.append(row)

    return pd.DataFrame(rows)


def outputs_are_probabilities(outputs: torch.Tensor) -> bool:
    row_sums = outputs.sum(dim=1)
    return bool(
        torch.all(outputs >= 0)
        and torch.all(outputs <= 1)
        and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    )


def calculate_metrics(results: pd.DataFrame, labels: Sequence[str]) -> Tuple[pd.DataFrame, float, str]:
    y_true = results["true_label"].tolist()
    y_pred = results["predicted_class"].tolist()
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = calculate_specificity(cm)

    metrics_rows = []
    for index, label in enumerate(labels):
        metrics_rows.append(
            {
                "class": label,
                "precision": precision[index],
                "recall_sensitivity": recall[index],
                "specificity": specificity[index],
                "f1_score": f1[index],
                "support": int(support[index]),
            }
        )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    metrics_rows.extend(
        [
            {
                "class": "macro_avg",
                "precision": macro_precision,
                "recall_sensitivity": macro_recall,
                "specificity": specificity.mean() if len(specificity) else 0.0,
                "f1_score": macro_f1,
                "support": int(sum(support)),
            },
            {
                "class": "weighted_avg",
                "precision": weighted_precision,
                "recall_sensitivity": weighted_recall,
                "specificity": weighted_specificity(specificity, support),
                "f1_score": weighted_f1,
                "support": int(sum(support)),
            },
        ]
    )

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame(metrics_rows), accuracy, report


def calculate_confidence_metrics(results: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    evaluated = results.copy()
    evaluated["is_correct"] = evaluated["true_label"] == evaluated["predicted_class"]

    rows = [confidence_summary_row(evaluated, "global")]
    for label in labels:
        class_results = evaluated[evaluated["true_label"] == label]
        rows.append(confidence_summary_row(class_results, label))

    return pd.DataFrame(rows)


def confidence_summary_row(results: pd.DataFrame, label: str) -> Dict[str, float]:
    hits = results[results["is_correct"]]
    errors = results[~results["is_correct"]]

    return {
        "class": label,
        "total_samples": int(len(results)),
        "hits": int(len(hits)),
        "errors": int(len(errors)),
        "confidence_mean_errors": safe_mean(errors["confidence"]),
        "confidence_mean_hits": safe_mean(hits["confidence"]),
        "confidence_max": safe_max(results["confidence"]),
        "confidence_min": safe_min(results["confidence"]),
    }


def safe_mean(values: pd.Series):
    return float(values.mean()) if not values.empty else None


def safe_max(values: pd.Series):
    return float(values.max()) if not values.empty else None


def safe_min(values: pd.Series):
    return float(values.min()) if not values.empty else None


def calculate_specificity(cm):
    total = cm.sum()
    specificities = []
    for index in range(len(cm)):
        tp = cm[index, index]
        fp = cm[:, index].sum() - tp
        fn = cm[index, :].sum() - tp
        tn = total - tp - fp - fn
        denominator = tn + fp
        specificities.append(tn / denominator if denominator else 0.0)
    return pd.Series(specificities, dtype=float)


def weighted_specificity(specificity, support) -> float:
    support_sum = support.sum()
    if support_sum == 0:
        return 0.0
    return float((specificity * support).sum() / support_sum)


def save_visualizations(results: pd.DataFrame, metrics: pd.DataFrame, labels: Sequence[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    cm = confusion_matrix(results["true_label"], results["predicted_class"], labels=labels)
    plt.figure(figsize=(max(8, len(labels)), max(6, len(labels) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")
    plt.title("Matriz de confusao")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    class_counts = results["true_label"].value_counts().reindex(labels, fill_value=0)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, color="#4C78A8")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de imagens")
    plt.title("Distribuicao das classes")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=200)
    plt.close()

    per_class_metrics = metrics[~metrics["class"].isin(["macro_avg", "weighted_avg"])]
    save_metric_barplot(per_class_metrics, "precision", "Precisao por classe", output_dir / "precision_by_class.png")
    save_metric_barplot(
        per_class_metrics,
        "recall_sensitivity",
        "Recall/Sensibilidade por classe",
        output_dir / "recall_by_class.png",
    )


def save_metric_barplot(metrics: pd.DataFrame, metric_column: str, title: str, output_path: Path):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=metrics, x="class", y=metric_column, color="#59A14F")
    plt.ylim(0, 1)
    plt.xlabel("Classe")
    plt.ylabel(metric_column)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_summary(
    dataset: EmotionFolderDataset,
    output_dir: Path,
    results: pd.DataFrame,
    metrics: pd.DataFrame,
    confidence_metrics: pd.DataFrame,
    accuracy: float,
    report: str,
):
    print("\n====== Avaliacao RepVGG - Classificador Emocional ======")
    print(f"Imagens avaliadas: {len(results)}")
    print(f"Pastas/classes encontradas: {', '.join(dataset.folder_classes)}")
    print(f"Labels usados nas metricas: {', '.join(dataset.labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)
    print("\nResumo macro/weighted:")
    print(metrics[metrics["class"].isin(["macro_avg", "weighted_avg"])].to_string(index=False))
    print("\nResumo de confianca:")
    print(confidence_metrics.to_string(index=False))
    print("\nArquivos gerados:")
    print(f"- {output_dir / 'predictions.csv'}")
    print(f"- {output_dir / 'metrics.csv'}")
    print(f"- {output_dir / 'confidence_metrics.csv'}")
    print(f"- {output_dir / 'classification_report.txt'}")
    print(f"- {output_dir / 'confusion_matrix.png'}")
    print(f"- {output_dir / 'class_distribution.png'}")
    print(f"- {output_dir / 'precision_by_class.png'}")
    print(f"- {output_dir / 'recall_by_class.png'}")


def main():
    args = parse_args()
    _dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(_dev)
    print(f"Dispositivo de inferencia: {device}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_classes = tuple(args.classes)
    transform = build_transform(args.image_size, use_centercrop=args.centercrop, resize_size=args.resize_size)
    dataset = EmotionFolderDataset(args.dataset, transform=transform, label_aliases=LABEL_ALIASES)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    model = load_repvgg(args.weights, device)
    print(f"TTA: {'ativado (flip + rot ±5°)' if args.tta else 'desativado'}")
    results = run_inference(model, dataloader, model_classes, device, tta=args.tta)
    results.to_csv(output_dir / "predictions.csv", index=False)

    metric_labels = list(dict.fromkeys([*model_classes, *dataset.labels]))
    metrics, accuracy, report = calculate_metrics(results, metric_labels)
    metrics.insert(1, "accuracy_global", accuracy)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    confidence_metrics = calculate_confidence_metrics(results, metric_labels)
    confidence_metrics.to_csv(output_dir / "confidence_metrics.csv", index=False)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    save_visualizations(results, metrics, metric_labels, output_dir)
    print_summary(dataset, output_dir, results, metrics, confidence_metrics, accuracy, report)


if __name__ == "__main__":
    main()
