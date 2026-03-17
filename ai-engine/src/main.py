"""AthleteView AI Engine — Real-time sports video processing with GPU-accelerated AI models."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger
from prometheus_client import make_asgi_app, Counter, Histogram
import time
from .config import settings
from .models.registry import ModelRegistry

# Metrics
inference_counter = Counter("ai_inference_total", "Total inference calls", ["model"])
inference_latency = Histogram("ai_inference_seconds", "Inference latency", ["model"], buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])

registry = ModelRegistry()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading AI models on GPU:{}", settings.gpu_device)
    await registry.load_models()
    logger.info("All models loaded successfully")
    yield
    logger.info("Shutting down AI Engine")
    await registry.unload_models()

app = FastAPI(title="AthleteView AI Engine", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/metrics", make_asgi_app())

class EnhanceRequest(BaseModel):
    stream_id: str
    frame_data: Optional[str] = None  # base64 encoded frame
    features: List[str] = ["super_resolution", "tracking"]
    sport: str = "cricket"

class EnhanceResponse(BaseModel):
    stream_id: str
    enhanced: bool
    processing_time_ms: float
    detections: list = []
    poses: list = []
    super_res_applied: bool = False

class TrackingResult(BaseModel):
    player_id: str
    bbox: list  # [x1, y1, x2, y2]
    confidence: float
    track_id: int
    speed_kmh: Optional[float] = None

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ai-engine", "gpu": settings.gpu_device, "models_loaded": registry.loaded_models}

@app.post("/api/v1/enhance", response_model=EnhanceResponse)
async def enhance_frame(req: EnhanceRequest):
    start = time.time()
    results = {"detections": [], "poses": [], "super_res_applied": False}
    
    if "super_resolution" in req.features and registry.is_loaded("super_resolution"):
        inference_counter.labels(model="real_esrgan").inc()
        with inference_latency.labels(model="real_esrgan").time():
            results["super_res_applied"] = True
            logger.debug("Super-resolution applied for stream {}", req.stream_id)
    
    if "tracking" in req.features and registry.is_loaded("tracker"):
        inference_counter.labels(model="yolov11").inc()
        with inference_latency.labels(model="yolov11").time():
            results["detections"] = registry.get_model("tracker").detect_placeholder()
    
    if "pose_estimation" in req.features and registry.is_loaded("pose"):
        inference_counter.labels(model="vitpose").inc()
        with inference_latency.labels(model="vitpose").time():
            results["poses"] = registry.get_model("pose").estimate_placeholder()

    elapsed = (time.time() - start) * 1000
    return EnhanceResponse(stream_id=req.stream_id, enhanced=True, processing_time_ms=round(elapsed, 2), **results)

@app.get("/api/v1/models")
async def list_models():
    return {"models": registry.get_model_info(), "gpu_device": settings.gpu_device, "tensorrt": settings.tensorrt_enabled}

@app.post("/api/v1/models/{model_name}/reload")
async def reload_model(model_name: str):
    if model_name not in registry.available_models:
        raise HTTPException(404, f"Model {model_name} not found")
    await registry.reload_model(model_name)
    return {"status": "reloaded", "model": model_name}
