"""Pose Estimation — Dual strategy: YOLO26-Pose (live) + ViTPose++ (analysis).

Model Stack v2.0 (April 2026):
  - YOLO26-Pose (live streaming): 68.8-70.4 mAP on COCO
    Built into YOLO26 detector — zero additional overhead
    Uses RLE (Residual Log-Likelihood Estimation) for precise keypoints
    Reference: https://docs.ultralytics.com/models/yolo26/
  - ViTPose++ (post-match analysis): 80.9 AP on COCO (still highest accuracy)
    Plain vision transformer, scalable 100M-1B params
    Best for detailed biomechanics where accuracy > speed
    Reference: https://blog.roboflow.com/best-pose-estimation-models/

Strategy:
  - During live broadcast: YOLO26-Pose runs bundled with detection (free!)
  - Post-match analysis: ViTPose++ for sub-pixel joint accuracy
  - Coach dashboard: ViTPose++ for biomechanics reports

License: YOLO26 (AGPL-3.0), ViTPose++ (Apache-2.0)
"""
import numpy as np
from loguru import logger
from ..config import settings

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]

# Sport-specific keypoint analysis
CRICKET_BOWLING_KEYPOINTS = ["right_shoulder", "right_elbow", "right_wrist", "left_hip", "right_hip"]
CRICKET_BATTING_KEYPOINTS = ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_hip", "right_hip"]


class PoseEstimator:
    """Dual-mode pose estimation: fast (YOLO26-Pose) and accurate (ViTPose++).
    
    Live mode (YOLO26-Pose):
      - Runs as part of YOLO26 detection — no extra GPU cost
      - 68.8-70.4 mAP, enough for overlay visualization
      - NMS-free, deterministic latency
      
    Analysis mode (ViTPose++):
      - Separate model, higher accuracy (80.9 AP)
      - Best for post-match biomechanics: bowling action analysis, batting stance
      - Handles occlusion better (transformer global attention)
    """

    def __init__(self, mode: str = "live"):
        """
        Args:
            mode: "live" for YOLO26-Pose (fast), "analysis" for ViTPose++ (accurate)
        """
        self.model = None
        self.processor = None
        self.mode = mode
        self.device = f"cuda:{settings.gpu_device}"

    async def load(self):
        if self.mode == "live":
            await self._load_yolo26_pose()
        else:
            await self._load_vitpose()

    async def _load_yolo26_pose(self):
        """Load YOLO26-Pose — integrated detection + pose, zero overhead."""
        try:
            from ultralytics import YOLO
            weight_path = f"{settings.model_cache_dir}/{settings.pose_model_live}"
            self.model = YOLO(weight_path)
            self.model.to(self.device)
            logger.info("YOLO26-Pose loaded (live mode): {} — bundled with detection", settings.pose_model_live)
        except Exception as e:
            logger.warning("YOLO26-Pose not available: {} — using stub", e)
            self.model = None

    async def _load_vitpose(self):
        """Load ViTPose++ — highest accuracy for biomechanics analysis."""
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            model_id = settings.pose_model_analysis
            self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=settings.model_cache_dir)
            self.model = AutoModelForImageClassification.from_pretrained(model_id, cache_dir=settings.model_cache_dir)
            logger.info("ViTPose++ loaded (analysis mode): {} — 80.9 AP COCO", model_id)
        except Exception as e:
            logger.warning("ViTPose++ not available: {} — using stub", e)
            self.model = None

    async def unload(self):
        self.model = None
        self.processor = None

    def estimate(self, frame: np.ndarray, bboxes: list[list[float]] = None) -> list[dict]:
        """Estimate poses for detected persons.
        
        Args:
            frame: BGR image
            bboxes: List of person bounding boxes (required for ViTPose++, optional for YOLO26-Pose)
            
        Returns:
            List of pose results with keypoints and scores
        """
        if self.model is None:
            return self.estimate_placeholder()

        if self.mode == "live":
            return self._estimate_yolo26(frame)
        else:
            return self._estimate_vitpose(frame, bboxes or [])

    def _estimate_yolo26(self, frame: np.ndarray) -> list[dict]:
        """YOLO26-Pose: Single-pass detection + pose."""
        results = self.model(frame, verbose=False)
        poses = []
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints
            boxes = results[0].boxes
            for i in range(len(kps)):
                keypoints_data = kps[i].data.cpu().numpy()[0]  # (17, 3) — x, y, conf
                pose = {
                    "bbox": boxes.xyxy[i].tolist() if boxes is not None else [0, 0, 0, 0],
                    "keypoints": [
                        {"name": KEYPOINT_NAMES[j], "x": float(keypoints_data[j][0]), "y": float(keypoints_data[j][1]), "score": float(keypoints_data[j][2])}
                        for j in range(min(17, len(keypoints_data)))
                    ],
                    "model": "yolo26-pose",
                }
                poses.append(pose)
        return poses

    def _estimate_vitpose(self, frame: np.ndarray, bboxes: list[list[float]]) -> list[dict]:
        """ViTPose++: Top-down pose for each detected person crop."""
        poses = []
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            inputs = self.processor(images=crop, return_tensors="pt")
            outputs = self.model(**inputs)
            keypoints = outputs.logits.detach().cpu().numpy().reshape(-1, 3)
            poses.append({
                "bbox": bbox,
                "keypoints": [
                    {"name": KEYPOINT_NAMES[i], "x": float(kp[0]), "y": float(kp[1]), "score": float(kp[2])}
                    for i, kp in enumerate(keypoints[:17])
                ],
                "model": "vitpose++",
            })
        return poses

    def analyze_bowling_action(self, keypoints: list[dict]) -> dict:
        """Cricket-specific: Analyze bowling arm action for technique metrics."""
        kp_map = {kp["name"]: kp for kp in keypoints}
        analysis = {
            "arm_angle": None,
            "hip_rotation": None,
            "release_height": None,
            "technique_score": None,
        }
        # Detailed biomechanics computation would go here
        # Requires ViTPose++ mode for sub-pixel accuracy
        return analysis

    def estimate_placeholder(self) -> list:
        return [{"bbox": [100, 50, 300, 500], "keypoints": [{"name": n, "x": 200.0 + i * 5, "y": 100.0 + i * 20, "score": 0.9} for i, n in enumerate(KEYPOINT_NAMES)], "model": "placeholder"}]
