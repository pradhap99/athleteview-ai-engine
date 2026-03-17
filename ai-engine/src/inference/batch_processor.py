"""Batched GPU Inference — Processes multiple frames simultaneously for throughput."""
import numpy as np
from loguru import logger

class BatchProcessor:
    """Batches frames for efficient GPU inference."""

    def __init__(self, batch_size: int = 4, timeout_ms: float = 50.0):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.pending: list = []

    def add_frame(self, frame: np.ndarray, stream_id: str) -> bool:
        self.pending.append({"frame": frame, "stream_id": stream_id})
        return len(self.pending) >= self.batch_size

    def get_batch(self) -> list:
        batch = self.pending[:self.batch_size]
        self.pending = self.pending[self.batch_size:]
        return batch

    def flush(self) -> list:
        batch = self.pending
        self.pending = []
        return batch
