"""
AthleteView AI Platform - Stream Processing Module.

Provides frame decoding, overlay rendering, and multi-view compositing
for live video streams.
"""

from src.processing.frame_decoder import FrameDecoder, DecodedFrame
from src.processing.overlay_renderer import OverlayRenderer, BiometricData, OverlayConfig
from src.processing.compositor import MultiViewCompositor, CompositorConfig

__all__ = [
    "FrameDecoder",
    "DecodedFrame",
    "OverlayRenderer",
    "BiometricData",
    "OverlayConfig",
    "MultiViewCompositor",
    "CompositorConfig",
]
