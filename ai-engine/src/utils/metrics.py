"""Prometheus Metrics for AI Engine monitoring."""
from prometheus_client import Counter, Histogram, Gauge

frames_processed = Counter("ai_frames_processed_total", "Total frames processed", ["stream_id"])
model_inference_time = Histogram("ai_model_inference_seconds", "Per-model inference time", ["model_name"])
gpu_memory_usage = Gauge("ai_gpu_memory_bytes", "GPU memory usage", ["device"])
active_pipelines = Gauge("ai_active_pipelines", "Number of active processing pipelines")
