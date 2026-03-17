"""SRT Stream Receiver — Accepts SRT connections from SmartPatch body cameras.

SRT (Secure Reliable Transport) is the optimal protocol for wearable cameras over 5G:
- ARQ-based error recovery (handles packet loss on cellular)
- UDP-based (lower overhead than TCP/RTMP)
- 0.5-2s latency achievable
- AES-128/256 encryption built-in
"""
import asyncio
from loguru import logger
from dataclasses import dataclass, field

@dataclass
class SRTStream:
    stream_id: str
    peer_address: str
    port: int = 9000
    latency_ms: int = 120
    passphrase: str = ""
    status: str = "connecting"
    frames_received: int = 0
    bytes_received: int = 0
    packet_loss_rate: float = 0.0

class SRTReceiver:
    """Manages SRT stream ingestion from multiple body cameras."""

    def __init__(self, listen_port: int = 9000, max_streams: int = 200):
        self.listen_port = listen_port
        self.max_streams = max_streams
        self.active_streams: dict[str, SRTStream] = {}

    async def start(self):
        logger.info("SRT receiver listening on port {} (max {} streams)", self.listen_port, self.max_streams)

    async def accept_stream(self, stream_id: str, peer_address: str) -> SRTStream:
        if len(self.active_streams) >= self.max_streams:
            raise RuntimeError(f"Max streams ({self.max_streams}) reached")
        stream = SRTStream(stream_id=stream_id, peer_address=peer_address, port=self.listen_port)
        self.active_streams[stream_id] = stream
        logger.info("SRT stream accepted: {} from {}", stream_id, peer_address)
        return stream

    async def remove_stream(self, stream_id: str):
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            logger.info("SRT stream removed: {}", stream_id)

    def get_stats(self) -> dict:
        return {"active_streams": len(self.active_streams), "max_streams": self.max_streams, "streams": {sid: {"frames": s.frames_received, "bytes": s.bytes_received, "loss": s.packet_loss_rate} for sid, s in self.active_streams.items()}}
