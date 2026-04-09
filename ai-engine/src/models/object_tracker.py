"""YOLO26 + BoT-SORT — Multi-object tracking for athletes, balls, and equipment.

Model Stack v2.0 (April 2026):
  - YOLO26 (released January 2026): NMS-free end-to-end detection
    43% faster CPU inference vs YOLOv11, native pose estimation with RLE
    Nano: 40.9 mAP at 1.7ms, X: 57.5 mAP at 11.8ms on T4 TensorRT
  - BoT-SORT: Still best multi-object tracker for sports
  - YOLO26 also includes pose estimation — can replace separate pose model for live use

Pretrained weights:
  - yolo26n.pt (nano, edge): https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n.pt
  - yolo26s.pt (small): https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26s.pt
  - yolo26m.pt (medium): https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26m.pt
  - yolo26x.pt (extra-large, production): https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26x.pt
  - yolo26m-pose.pt (pose, live streaming): integrated detection + pose

Reference: https://docs.ultralytics.com/models/yolo26/
Paper: https://arxiv.org/html/2601.12882v2
License: AGPL-3.0 (requires Ultralytics commercial license for proprietary use)
"""
import numpy as np
from loguru import logger
from ..config import settings

WEIGHTS = {
    "yolo26n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n.pt",
        "mAP": 40.9, "cpu_ms": 38.9, "gpu_ms": 1.7, "params": "2.4M",
    },
    "yolo26s.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26s.pt",
        "mAP": 48.6, "cpu_ms": 87.2, "gpu_ms": 2.5, "params": "9.5M",
    },
    "yolo26m.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26m.pt",
        "mAP": 53.1, "cpu_ms": 220.0, "gpu_ms": 4.7, "params": "20.4M",
    },
    "yolo26x.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26x.pt",
        "mAP": 57.5, "cpu_ms": 525.8, "gpu_ms": 11.8, "params": "55.7M",
    },
    # Legacy YOLO11 (fallback)
    "yolo11x.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        "note": "Legacy — use YOLO26 instead",
    },
}

SPORT_CLASSES = {
    "cricket": {"bat": 0, "ball": 1, "stumps": 2, "person": 3},
    "football": {"ball": 0, "person": 1, "goal": 2},
    "kabaddi": {"person": 0, "mat_boundary": 1},
    "basketball": {"ball": 0, "person": 1, "hoop": 2},
}


class ObjectTracker:
    """YOLO26 with BoT-SORT for persistent multi-athlete tracking.
    
    Key advantages over YOLOv11:
      - NMS-free: No post-processing bottleneck, constant-time inference
      - 43% faster on CPU: Optimized for edge devices (SmartPatch RV1106)
      - Native pose: yolo26-pose models include keypoint detection
      - Better small object detection: STAL + ProgLoss innovations
    """

    def __init__(self, use_edge: bool = False):
        """
        Args:
            use_edge: If True, use nano variant for edge/CPU deployment
        """
        self.model = None
        self.tracker_config = settings.tracker_config
        self.confidence_threshold = 0.5
        self.device = f"cuda:{settings.gpu_device}"
        self.model_name = settings.tracker_model_edge if use_edge else settings.tracker_model

    async def load(self):
        try:
            from ultralytics import YOLO
            weight_path = f"{settings.model_cache_dir}/{self.model_name}"
            self.model = YOLO(weight_path)
            self.model.to(self.device)

            model_info = WEIGHTS.get(self.model_name, {})
            logger.info(
                "YOLO26 tracker loaded: {} on {} (mAP={}, {}ms GPU, NMS-free)",
                self.model_name, self.device,
                model_info.get("mAP", "?"),
                model_info.get("gpu_ms", "?"),
            )
        except (ImportError, FileNotFoundError) as e:
            logger.warning("YOLO26 not available: {} — running in stub mode", e)
            self.model = None

    async def unload(self):
        self.model = None

    def track(self, frame: np.ndarray, sport: str = "cricket") -> list[dict]:
        """Track objects in a single frame with persistent IDs.
        
        YOLO26 advantage: NMS-free end-to-end — no post-processing latency variance.
        Critical for live sports where deterministic frame timing matters.
        
        Returns list of detections: [{bbox, confidence, class_id, track_id, speed_kmh}]
        """
        if self.model is None:
            return self.detect_placeholder()

        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            verbose=False,
        )
        
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                det = {
                    "bbox": boxes.xyxy[i].tolist(),
                    "confidence": float(boxes.conf[i]),
                    "class_id": int(boxes.cls[i]),
                    "track_id": int(boxes.id[i]) if boxes.id is not None else -1,
                    "speed_kmh": None,
                }
                detections.append(det)
        return detections

    def detect_placeholder(self) -> list:
        """Placeholder detections for testing without GPU."""
        return [
            {"player_id": "player_1", "bbox": [120, 80, 280, 450], "confidence": 0.95, "track_id": 1, "speed_kmh": 12.4},
            {"player_id": "player_2", "bbox": [400, 100, 560, 480], "confidence": 0.91, "track_id": 2, "speed_kmh": 8.7},
        ]
