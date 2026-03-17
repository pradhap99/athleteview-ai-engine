"""GPU Utilization Monitor — Tracks VRAM, compute, temperature for resource management."""
from loguru import logger

def get_gpu_stats(device_id: int = 0) -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        props = torch.cuda.get_device_properties(device_id)
        return {
            "available": True,
            "name": props.name,
            "total_memory_gb": round(props.total_mem / 1e9, 2),
            "allocated_gb": round(torch.cuda.memory_allocated(device_id) / 1e9, 2),
            "cached_gb": round(torch.cuda.memory_reserved(device_id) / 1e9, 2),
            "utilization": None,  # Requires pynvml for real utilization
        }
    except Exception as e:
        return {"available": False, "error": str(e)}
