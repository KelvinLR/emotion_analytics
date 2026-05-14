"""
Emotion-specific processor implementation
This class contains all emotion detection logic separate from generic streaming
"""
import time
import os
import sys
import pathlib
import logging

# Configure path to shared module BEFORE any other imports
_current_file = pathlib.Path(__file__).resolve()
# From: /home/bullying-analytics/emotions/src/lib/emotion_processor.py
# Navigate up to: /home/bullying-analytics/
_bullying_analytics_root = _current_file.parent.parent.parent.parent  # Go up 4 levels
_shared_path = _bullying_analytics_root / "shared"  # /home/bullying-analytics/shared

# Add the parent directory that contains 'shared' module
if _shared_path.exists():
    sys.path.insert(0, str(_bullying_analytics_root))  # Add /home/bullying-analytics to path
    logging.info(f"Added to sys.path: {_bullying_analytics_root}")
    logging.info(f"Shared module path exists: {_shared_path}")
else:
    logging.warning(f"Shared module path not found: {_shared_path}")
    # Try to find it from environment or working directory
    possible_paths = [
        pathlib.Path("/home/bullying-analytics"),
        pathlib.Path.cwd().parent.parent.parent,
    ]
    for possible_path in possible_paths:
        test_shared = possible_path / "shared"
        if test_shared.exists():
            sys.path.insert(0, str(possible_path))
            logging.info(f"Found shared module at: {test_shared}")
            break

import threading
import numpy as np
import cv2
from typing import List, Dict, Any

# Import shared utilities with better error handling
BaseAnalyticProcessor = None
try:
    from shared.core.base_processor import BaseAnalyticProcessor
    logging.info("Successfully imported BaseAnalyticProcessor")
except ImportError as e:
    logging.error(f"Could not import BaseAnalyticProcessor: {e}")
    logging.error(f"Current working directory: {os.getcwd()}")
    logging.error(f"sys.path: {sys.path}")
    # Create a dummy base class if import fails
    class BaseAnalyticProcessor:
        def get_analytic_name(self) -> str:
            return "base"
        def is_ready(self) -> bool:
            return False

# Import shared utilities
try:
    from shared.utils.timestamp_utils import TimestampManager
    from shared.utils.analytics_publisher import AnalyticsDetectionPublisher, get_default_rabbitmq_config
except ImportError as e:
    logging.warning(f"Could not import shared utilities: {e}")
    TimestampManager = None
    AnalyticsDetectionPublisher = None

# MongoDB logger
try:
    from shared.utils.mongodb_logger import get_mongodb_logger
except ImportError:
    get_mongodb_logger = None

class EmotionProcessor(BaseAnalyticProcessor):
    
    def __init__(self, device: str = None, camera_id: str = None):
        # Auto-detect device if not specified
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
            
        self.device = device
        self.camera_id = camera_id or "unknown"
        self.camera_name = None
        self.analytic_id = None
        self.analytic_name = self.get_analytic_name()
        self.models_loaded = False
        self.enable_realtime_publish = True
        self.enable_realtime_mongo_log = True
        
        # Log the initialization parameters
        logging.info(f"🔧 EmotionProcessor.__init__ called with camera_id={camera_id}, device={device}")
        
        # Try to import torch and apply safety settings
        self.torch = None
        try:
            import torch
            self.torch = torch
            try:
                # reduce intra-op parallelism to avoid some CUDA/threading issues
                torch.set_num_threads(1)
            except Exception:
                pass
        except Exception:
            self.torch = None
        # inference lock to serialize access to GPU model
        self._infer_lock = threading.Lock()
        
        # Define model paths relative to this processor
        weights_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "weights")
        self.face_model_path = os.path.join(weights_dir, "yolov8n-face.pt")
        self.emotion_model_path = os.path.join(weights_dir, "repvgg.pth")
        
        # Initialize detectors as None
        self.face_detector = None
        self.emotion_detector = None
        self.logger = self._create_logger()
        
        # Initialize timestamp manager
        self.timestamp_manager = None
        if TimestampManager:
            try:
                self.timestamp_manager = TimestampManager()
                self.logger.info("Timestamp manager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize timestamp manager: {e}")
        
        # Initialize analytics publisher
        self.analytics_publisher = None
        if AnalyticsDetectionPublisher:
            try:
                config = get_default_rabbitmq_config()
                self.analytics_publisher = AnalyticsDetectionPublisher(config)
                self.logger.info("Analytics publisher initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize analytics publisher: {e}")
        
        # Initialize MongoDB logger directly
        self.mongodb_logger = None
        self.mongo_log_interval_sec = 1.0
        self._last_mongo_log_by_camera = {}
        if get_mongodb_logger:
            try:
                self.mongodb_logger = get_mongodb_logger()
                if self.mongodb_logger and self.mongodb_logger.enabled:
                    self.logger.info("MongoDB logger initialized")
                else:
                    self.logger.warning("MongoDB logger disabled or not connected")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MongoDB logger: {e}")

        try:
            interval_sec = float(os.getenv("MONGO_LOG_INTERVAL_SEC", "1.0"))
            self.mongo_log_interval_sec = max(0.2, interval_sec)
        except Exception:
            self.mongo_log_interval_sec = 1.0

        # Load models during initialization
        self._load_models()
        
        # Remove the TEST log - it was just for debugging
        
        # Log processor initialization if camera_id was provided
        if camera_id and camera_id != "unknown":
            self.logger.info(f"EmotionProcessor initialized with camera_id: {camera_id}")
            # Log real camera registration
            if self.mongodb_logger and self.mongodb_logger.enabled:
                try:
                    self.mongodb_logger.log_camera_registration(
                        camera_id=camera_id,
                        analytic_name=self.get_analytic_name(),
                        metadata={
                            "device": self.device,
                            "models_loaded": self.models_loaded,
                            "timestamp_manager_active": self.timestamp_manager is not None,
                            "analytics_publisher_active": self.analytics_publisher is not None,
                        }
                    )
                    self.logger.info(f"Camera registration logged during init: {camera_id}")
                except Exception as e:
                    self.logger.error(f"Failed to log camera registration: {e}")
        else:
            self.logger.warning(f"EmotionProcessor initialized WITHOUT camera_id (camera_id={camera_id})")

    def _load_emotion_model(self):
        try:
            import torch
            import torch.backends.cudnn as cudnn
            import torchvision.transforms as transforms
            from PIL import Image
            
            # Import RepVGG from the weights directory (assuming repvgg.py is there)
            weights_dir = pathlib.Path(__file__).parent / "weights"
            sys.path.append(str(weights_dir.parent.parent / "core" / "lib"))
            
            # Try to import from existing location
            try:
                from .repvgg import create_RepVGG_A0
            except ImportError:
                self.logger.warning("RepVGG model not found, using placeholder")
                return None
            
            # Create model
            model = create_RepVGG_A0(deploy=True)
            model.to(self.device)
            
            # Load weights
            if os.path.exists(self.emotion_model_path):
                model.load_state_dict(torch.load(self.emotion_model_path, map_location=self.device))
                cudnn.benchmark = True
                model.eval()
                
                # Create transform
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
                
                # Return a simple dict with model and utilities
                return {
                    'model': model,
                    'transform': transform,
                    'emotions': ("anger", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprise"),
                    'torch': torch,
                    'Image': Image
                }
            else:
                self.logger.warning(f"Emotion model weights not found at {self.emotion_model_path}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not load emotion model: {e}")
            return None
    
    def _create_logger(self):
        logger = logging.getLogger("EmotionProcessor")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        # Only add handler if none exists to avoid duplicates
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        return logger
    
    def _load_models(self):
        try:
            # Check if model files exist
            if not os.path.exists(self.face_model_path):
                self.logger.warning(f"Face model not found at {self.face_model_path}")
                self.models_loaded = False
                return
            
            if not os.path.exists(self.emotion_model_path):
                self.logger.warning(f"Emotion model not found at {self.emotion_model_path}")
                self.models_loaded = False
                return
            
            # Try to load models (simplified version - you can add your specific loading logic)
            try:
                from ultralytics import YOLO
                self.face_detector = YOLO(self.face_model_path)
                # put model in eval mode if available
                try:
                    if hasattr(self.face_detector, 'model'):
                        self.face_detector.model.eval()
                except Exception:
                    pass
                # attempt to move model to CUDA if requested
                try:
                    if self.torch is not None and isinstance(self.device, str) and self.device.startswith('cuda'):
                        try:
                            # some ultralytics versions support .to()
                            self.face_detector.to(self.device)
                        except Exception:
                            try:
                                if hasattr(self.face_detector, 'model'):
                                    self.face_detector.model.to(self.device)
                            except Exception:
                                pass
                except Exception:
                    pass

                self.logger.info("Face detector loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load face detector: {e}")
            
            try:
                # Initialize emotion detector using existing RepVGG model
                self.emotion_detector = self._load_emotion_model()
                if self.emotion_detector is not None:
                    self.logger.info("Emotion detector loaded successfully")
                else:
                    self.logger.warning(" Emotion detector not loaded - using placeholder")
            except Exception as e:
                self.logger.warning(f"Could not load emotion detector: {e}")
                self.emotion_detector = None
            
            self.models_loaded = True
            self.logger.info("Emotion Analytics: Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f" Error loading emotion models: {e}")
            self.models_loaded = False
    
    def process_frame(self, frame):
        if not self.is_ready():
            return frame
        
        # Update timestamp manager
        if self.timestamp_manager:
            self.timestamp_manager.increment_frame_count()
            
        try:
            annotated_frame = frame.copy()
            
            # Face detection
            if self.face_detector:
                # serialize inference to avoid concurrent CUDA access from multiple threads
                with self._infer_lock:
                    if self.torch is not None:
                        with self.torch.no_grad():
                            face_results = self.face_detector(frame, device=self.device, verbose=False)
                    else:
                        face_results = self.face_detector(frame, verbose=False)

                # Extract face crops for emotion detection
                face_crops = []
                face_boxes = []
                
                for result in face_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Extract face crop
                            face_crop = frame[y1:y2, x1:x2]
                            if face_crop.size > 0:
                                face_crops.append(face_crop)
                                face_boxes.append((x1, y1, x2, y2))
                
                # Emotion detection on face crops
                emotions = []
                if self.emotion_detector and face_crops:
                    emotions = self._detect_emotions_batch(face_crops)
                
                # Process detections and collect for frame-level publishing
                frame_emotions = []
                
                for i, (x1, y1, x2, y2) in enumerate(face_boxes):
                    # Draw face bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw emotion if detected
                    if i < len(emotions):
                        emotion_name, confidence = emotions[i]
                        color = self._get_emotion_color(emotion_name)
                        
                        # Create emotion label
                        label = f"{emotion_name}: {confidence:.2f}"
                        
                        # Calculate text size for background
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Draw background rectangle for text
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - text_height - 10), 
                                    (x1 + text_width, y1), 
                                    color, -1)
                        
                        # Draw emotion text
                        cv2.putText(annotated_frame, label, 
                                  (x1, y1 - 5), font, font_scale, 
                                  (255, 255, 255), thickness)
                        
                        # Collect emotion data for frame-level publishing
                        emotion_data = {
                            "bounding_box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "class_name": emotion_name,
                            "confidence": float(confidence),
                            "detection_id": f"emotion_{i}_{int(time.time() * 1000)}"
                        }
                        frame_emotions.append(emotion_data)
                
                # Publish all emotions for this frame at once
                if frame_emotions:
                    if self.enable_realtime_publish:
                        self._publish_frame_emotions(frame_emotions)

                    if self.enable_realtime_mongo_log and self.mongodb_logger and self.mongodb_logger.enabled and self.timestamp_manager:
                        try:
                            camera_key = self.camera_id or "unknown"
                            if camera_key == "unknown":
                                return annotated_frame

                            now_ts = time.time()
                            last_log_ts = self._last_mongo_log_by_camera.get(camera_key, 0.0)

                            if (now_ts - last_log_ts) < self.mongo_log_interval_sec:
                                return annotated_frame

                            ts_info = self.timestamp_manager.get_timestamp_for_detection(self.camera_id)
                            timestamp = ts_info.get("timestamp")
                            frame_index = ts_info.get("frame_index")

                            self.mongodb_logger.log_emotion_detection(
                                camera_id=self.camera_id,
                                name=self.camera_name,
                                timestamp=float(timestamp) if timestamp is not None else time.time(),
                                detections=frame_emotions,
                                frame_id=str(frame_index) if frame_index is not None else None,
                                analytic_id=self.analytic_id,
                                analytic_name=self.analytic_name,
                                metadata={
                                    "source": "emotion_analytics",
                                },
                            )
                            self._last_mongo_log_by_camera[camera_key] = now_ts
                        except Exception as e:
                            self.logger.warning(f"Failed to log emotions to MongoDB: {e}")
                
                # SEMPRE retornar o frame anotado
                return annotated_frame
            
            # If no face detector, just return original frame
            return frame

        except RuntimeError as e:
            msg = str(e)
            self.logger.error(f"Error in emotion processing: {msg}")
            
            # Log processing error to MongoDB
            if self.mongodb_logger and self.mongodb_logger.enabled:
                try:
                    self.mongodb_logger.log_analytics_event(
                        event_type="error",
                        camera_id=self.camera_id,
                        data={
                            "operation": "frame_processing",
                            "error": msg,
                            "error_type": "RuntimeError"
                        }
                    )
                except Exception:
                    pass
            
            # Handle specific CUDA errors
            if 'illegal memory access' in msg or 'CUDA error' in msg:
                try:
                    if self.torch is not None and self.torch.cuda.is_available():
                        self.logger.warning("CUDA error detected - emptying CUDA cache and switching to CPU")
                        try:
                            self.torch.cuda.empty_cache()
                        except Exception:
                            pass
                        self.device = 'cpu'
                except Exception:
                    pass
            return frame

        except Exception as e:
            self.logger.error(f"Error in emotion processing: {e}")
            
            # Log error to MongoDB
            if self.mongodb_logger and self.mongodb_logger.enabled:
                try:
                    self.mongodb_logger.log_analytics_event(
                        event_type="error",
                        camera_id=self.camera_id,
                        data={
                            "operation": "frame_processing",
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                except Exception:
                    pass
            
            return frame
    
    def _get_emotion_color(self, emotion):
        """Get color for emotion visualization"""
        color_map = {
            "anger": (0, 0, 255),      # Red
            "contempt": (128, 0, 128),  # Purple
            "disgust": (0, 100, 0),     # Dark Green
            "fear": (0, 0, 139),        # Dark Blue
            "happy": (0, 255, 255),     # Yellow
            "neutral": (128, 128, 128), # Gray
            "sad": (255, 0, 0),         # Blue
            "surprise": (0, 165, 255),  # Orange
            "unknown": (255, 255, 255)  # White
        }
        return color_map.get(emotion, (255, 255, 255))
    
    def _detect_emotions_batch(self, face_crops):
        """Detect emotions from list of face crops"""
        if not self.emotion_detector or not face_crops:
            return []
        
        results = []
        model = self.emotion_detector['model']
        transform = self.emotion_detector['transform']
        emotions = self.emotion_detector['emotions']
        torch = self.emotion_detector['torch']
        Image = self.emotion_detector['Image']
        
        with torch.no_grad():
            for face_crop in face_crops:
                try:
                    # Convert BGR to RGB
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(face_rgb)
                    
                    # Apply transforms
                    tensor_image = transform(pil_image)
                    tensor_batch = tensor_image.unsqueeze(0).to(self.device)
                    
                    # Get model prediction
                    outputs = model(tensor_batch)
                    
                    # Get emotion with highest confidence
                    emotion_idx = torch.argmax(outputs[0]).item()
                    confidence = outputs[0][emotion_idx].item()
                    
                    emotion_name = emotions[emotion_idx]
                    results.append((emotion_name, confidence))
                    
                except Exception as e:
                    self.logger.warning(f"Error processing face crop: {e}")
                    results.append(("unknown", 0.0))
        
        return results
    
    def _publish_frame_emotions(self, emotions_data: List[Dict[str, Any]]):
        if not self.analytics_publisher or not self.timestamp_manager:
            return
        
        try:
            # Get timestamp information
            timestamp_info = self.timestamp_manager.get_timestamp_for_detection(self.camera_id)
            
            # Publish frame-level detections
            success = self.analytics_publisher.publish_emotion_frame(
                camera_id=self.camera_id,
                timestamp=timestamp_info['timestamp'],
                emotions_data=emotions_data
            )
            
            # Removido log de sucesso/falha para reduzir verbosidade
                
        except Exception as e:
            self.logger.error(f"Error publishing frame emotions: {e}")
    
    def set_camera_id(self, camera_id: str, camera_name: str = None):
        """Called when camera_id is set after initialization"""
        old_camera_id = self.camera_id
        self.camera_id = camera_id
        if camera_name is not None:
            self.camera_name = camera_name
        elif not self.camera_name and camera_id and camera_id != "unknown":
            self.camera_name = camera_id
        self.logger.info(f"📹 set_camera_id() called: {old_camera_id} -> {camera_id}")
        
        # Only log if this is a real change (not from "unknown" to actual ID)
        if old_camera_id != camera_id and camera_id != "unknown":
            if self.mongodb_logger and self.mongodb_logger.enabled:
                try:
                    self.mongodb_logger.log_camera_registration(
                        camera_id=camera_id,
                        analytic_name=self.get_analytic_name(),
                        metadata={
                            "device": self.device,
                            "models_loaded": self.models_loaded,
                            "previous_camera_id": old_camera_id,
                            "analytic_id": self.analytic_id,
                        }
                    )
                    self.logger.info(f"✅ Camera ID change logged to MongoDB: {camera_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to log camera registration: {e}")

    def set_analytic_context(self, analytic_id: str = None, analytic_name: str = None):
        old_analytic_id = self.analytic_id
        old_analytic_name = self.analytic_name

        if analytic_id is not None and str(analytic_id).strip():
            self.analytic_id = str(analytic_id)

        if analytic_name is not None and str(analytic_name).strip():
            self.analytic_name = str(analytic_name)

        if old_analytic_id != self.analytic_id or old_analytic_name != self.analytic_name:
            self.logger.info(
                f"🧭 set_analytic_context() called: id {old_analytic_id} -> {self.analytic_id}, "
                f"name {old_analytic_name} -> {self.analytic_name}"
            )
    
    def extract_camera_id_from_rtmp(self, rtmp_url: str) -> str:
        try:
            if "/live/" in rtmp_url:
                # Extract everything after '/live/'
                camera_id = rtmp_url.split("/live/")[-1]
                # Remove any additional parameters
                camera_id = camera_id.split("?")[0]
                self.logger.info(f"Extracted camera_id from RTMP: {camera_id}")
                return camera_id
            else:
                self.logger.warning(f"Cannot extract camera_id from RTMP URL: {rtmp_url}")
                return "unknown"
        except Exception as e:
            self.logger.error(f"Error extracting camera_id from RTMP: {e}")
            return "unknown"
    
    def update_camera_from_stream_url(self, stream_url: str):
        self.logger.info(f"🎥 update_camera_from_stream_url() called with: {stream_url}")
        if stream_url and "rtmp://" in stream_url:
            extracted_id = self.extract_camera_id_from_rtmp(stream_url)
            if extracted_id != "unknown":
                self.set_camera_id(extracted_id)
                
                # Log stream start to MongoDB
                if self.mongodb_logger and self.mongodb_logger.enabled:
                    try:
                        self.mongodb_logger.log_stream_start(
                            camera_id=extracted_id,
                            analytic_name=self.get_analytic_name(),
                            stream_url=stream_url,
                            metadata={
                                "device": self.device,
                                "models_ready": self.is_ready(),
                            }
                        )
                        self.logger.info(f"Stream start logged to MongoDB: {extracted_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to log stream start: {e}")
    
    def get_analytic_name(self) -> str:
        return "emotion_detection"
    
    def is_ready(self) -> bool:
        # Consider ready if at least face detection is available
        # Emotion detection is optional but preferred
        return self.models_loaded and self.face_detector is not None