"""Model Registry — lazy-loads AI models with GPU management."""
from loguru import logger
from typing import Dict, Optional, Any
from .super_resolution import SuperResolutionModel
from .object_tracker import ObjectTracker
from .pose_estimator import PoseEstimator
from .highlight_detector import HighlightDetector
from .video_stabilizer import VideoStabilizer
from .audio_enhancer import AudioEnhancer

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.loaded_models: list = []
        self.available_models = {
            "super_resolution": SuperResolutionModel,
            "tracker": ObjectTracker,
            "pose": PoseEstimator,
            "highlight": HighlightDetector,
            "stabilizer": VideoStabilizer,
            "audio": AudioEnhancer,
        }

    async def load_models(self):
        """Load core models at startup."""
        priority_models = ["super_resolution", "tracker", "pose"]
        for name in priority_models:
            try:
                self.models[name] = self.available_models[name]()
                await self.models[name].load()
                self.loaded_models.append(name)
                logger.info("Loaded model: {}", name)
            except Exception as e:
                logger.warning("Failed to load {}: {} — running in CPU fallback", name, e)

    async def unload_models(self):
        for name, model in self.models.items():
            try:
                await model.unload()
                logger.info("Unloaded model: {}", name)
            except Exception:
                pass
        self.models.clear()
        self.loaded_models.clear()

    async def reload_model(self, name: str):
        if name in self.models:
            await self.models[name].unload()
        self.models[name] = self.available_models[name]()
        await self.models[name].load()
        if name not in self.loaded_models:
            self.loaded_models.append(name)

    def is_loaded(self, name: str) -> bool:
        return name in self.models

    def get_model(self, name: str) -> Any:
        return self.models.get(name)

    def get_model_info(self) -> list:
        info = []
        for name, cls in self.available_models.items():
            info.append({
                "name": name,
                "loaded": name in self.models,
                "class": cls.__name__,
                "description": cls.__doc__ or "",
            })
        return info
