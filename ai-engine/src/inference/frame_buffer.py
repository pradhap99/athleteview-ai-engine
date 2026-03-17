"""Circular Frame Buffer — Thread-safe buffer for managing incoming video frames."""
import threading
import numpy as np
from collections import deque

class FrameBuffer:
    """Thread-safe circular buffer for video frames with metadata."""

    def __init__(self, max_size: int = 120):
        self.buffer: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.frame_count = 0

    def push(self, frame: np.ndarray, timestamp: float, metadata: dict | None = None):
        with self.lock:
            self.buffer.append({"frame": frame, "timestamp": timestamp, "metadata": metadata or {}, "index": self.frame_count})
            self.frame_count += 1

    def pop(self) -> dict | None:
        with self.lock:
            return self.buffer.popleft() if self.buffer else None

    def peek(self, n: int = 1) -> list[dict]:
        with self.lock:
            return list(self.buffer)[-n:]

    @property
    def size(self) -> int:
        return len(self.buffer)

    @property
    def is_empty(self) -> bool:
        return len(self.buffer) == 0
