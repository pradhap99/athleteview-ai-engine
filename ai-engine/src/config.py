from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    gpu_device: int = 0
    model_cache_dir: str = "/app/weights"
    tensorrt_enabled: bool = True
    kafka_brokers: str = "localhost:9092"
    max_batch_size: int = 4
    super_res_model: str = "realesr-general-x4v3"
    tracker_model: str = "yolo11x.pt"
    pose_model: str = "usyd-community/vitpose-plus-large"
    audio_model: str = "speechbrain/sepformer-wham16k-enhancement"
    inference_dtype: str = "float16"

    class Config:
        env_prefix = ""

settings = Settings()
