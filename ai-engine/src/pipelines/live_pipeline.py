"""Live Processing Pipeline — Real-time frame enhancement, tracking, and overlay composition."""
import time
import numpy as np
from loguru import logger
from ..models.registry import ModelRegistry

class LivePipeline:
    """Orchestrates real-time AI processing for a single camera stream."""

    def __init__(self, registry: ModelRegistry, stream_id: str, features: list[str]):
        self.registry = registry
        self.stream_id = stream_id
        self.features = features
        self.frame_count = 0
        self.total_processing_ms = 0.0

    async def process_frame(self, frame: np.ndarray, biometrics: dict | None = None) -> dict:
        """Process a single frame through the AI pipeline.
        
        Pipeline order: Stabilize → Super-Res → Detect → Track → Pose → Overlay → Encode
        """
        start = time.time()
        result = {"frame": frame, "metadata": {}}

        # 1. Video stabilization
        if "stabilization" in self.features and self.registry.is_loaded("stabilizer"):
            frame = self.registry.get_model("stabilizer").stabilize(frame)
            result["frame"] = frame

        # 2. Super-resolution
        if "super_resolution" in self.features and self.registry.is_loaded("super_resolution"):
            frame = self.registry.get_model("super_resolution").enhance(frame)
            result["frame"] = frame
            result["metadata"]["super_res"] = True

        # 3. Object detection + tracking
        detections = []
        if "tracking" in self.features and self.registry.is_loaded("tracker"):
            detections = self.registry.get_model("tracker").track(frame)
            result["metadata"]["detections"] = detections

        # 4. Pose estimation
        if "pose_estimation" in self.features and self.registry.is_loaded("pose"):
            person_bboxes = [d["bbox"] for d in detections if d.get("class_id") == 0]
            poses = self.registry.get_model("pose").estimate(frame, person_bboxes)
            result["metadata"]["poses"] = poses

        # 5. Highlight detection
        if "highlight_detection" in self.features and self.registry.is_loaded("highlight"):
            highlight = self.registry.get_model("highlight").detect(
                audio_level=0.5, imu_magnitude=1.0,
                hr_delta=biometrics.get("hr_delta", 0) if biometrics else 0,
                frame_detections=detections,
            )
            result["metadata"]["highlight"] = highlight

        # 6. Biometric overlay data
        if biometrics:
            result["metadata"]["biometrics"] = biometrics

        elapsed_ms = (time.time() - start) * 1000
        self.frame_count += 1
        self.total_processing_ms += elapsed_ms
        result["metadata"]["processing_ms"] = round(elapsed_ms, 2)
        result["metadata"]["avg_processing_ms"] = round(self.total_processing_ms / self.frame_count, 2)

        return result
