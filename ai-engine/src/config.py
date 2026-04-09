from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    gpu_device: int = 0
    model_cache_dir: str = "/app/weights"
    tensorrt_enabled: bool = True
    kafka_brokers: str = "localhost:9092"
    max_batch_size: int = 4

    # === MODEL STACK v2.0 (April 2026 Upgrade) ===

    # Super-Resolution: VPEG (recorded) + FlashVSR (live)
    # VPEG: AIM 2025 winner, 5x cheaper than Real-ESRGAN, better perceptual quality
    # FlashVSR: 17fps near-4K on A100, first real-time diffusion VSR
    super_res_model: str = "vpeg-x4"  # For recorded content
    super_res_live_model: str = "flashvsr-x2"  # For live streaming
    super_res_legacy_model: str = "realesr-general-x4v3"  # Fallback

    # Object Detection + Tracking: YOLO26 (Jan 2026)
    # NMS-free, 43% faster CPU inference, native end-to-end
    tracker_model: str = "yolo26x.pt"  # Production: highest accuracy (57.5 mAP)
    tracker_model_edge: str = "yolo26n.pt"  # Edge: nano (40.9 mAP at 1.7ms)
    tracker_config: str = "botsort.yaml"  # BoT-SORT still best MOT tracker

    # Pose Estimation: Dual strategy
    # YOLO26-Pose for live (68.8-70.4 mAP, bundled with detection)
    # ViTPose++ for post-match analysis (80.9 AP, highest accuracy)
    pose_model_live: str = "yolo26m-pose.pt"  # Live: integrated with detection
    pose_model_analysis: str = "usyd-community/vitpose-plus-large"  # Analysis: best accuracy

    # Depth Estimation: Dual strategy
    # Depth Pro for player segmentation (1.8x better boundary F1)
    # DAV2 Metric-Outdoor for field/stadium geometry (best outdoor metric)
    depth_model_segmentation: str = "apple/depth-pro"
    depth_model_geometry: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large"

    # 3D Reconstruction: Gaussian Splatting (validated by 2026 Olympics)
    splat_engine: str = "gsplat"

    # Intelligence Layer: Gemma 4 26B MoE (April 2026)
    # Native video understanding, 256K context, 4B active params, Apache 2.0
    intelligence_model: str = "google/gemma-4-27b-it"
    intelligence_context_window: int = 262144  # 256K tokens

    # Audio Enhancement: SpeechBrain (unchanged)
    audio_model: str = "speechbrain/sepformer-wham16k-enhancement"

    # Inference settings
    inference_dtype: str = "float16"

    class Config:
        env_prefix = ""

settings = Settings()
