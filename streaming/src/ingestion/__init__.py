"""
AthleteView AI Platform - Stream Ingestion Module.

Provides SRT and RTMP receivers for video stream ingestion,
plus a device manager for camera registration and tracking.
"""

from src.ingestion.srt_receiver import SRTReceiver, SRTEndpoint, SRTStats
from src.ingestion.rtmp_receiver import RTMPReceiver, RTMPEndpoint, RTMPStats
from src.ingestion.device_manager import DeviceManager, DeviceInfo

__all__ = [
    "SRTReceiver",
    "SRTEndpoint",
    "SRTStats",
    "RTMPReceiver",
    "RTMPEndpoint",
    "RTMPStats",
    "DeviceManager",
    "DeviceInfo",
]
