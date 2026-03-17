"""
AthleteView AI Platform - Frame Decoder.

Decodes video streams into raw frames using PyAV (FFmpeg bindings).
Supports hardware-accelerated decoding (NVDEC), color-space conversion,
and PTS correction for live streams.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import av
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DecodedFrame:
    """A single decoded video frame with metadata."""

    frame: np.ndarray  # RGB uint8 array, shape (H, W, 3)
    pts: int  # Presentation timestamp
    dts: Optional[int]  # Decoding timestamp
    keyframe: bool  # True if this is a keyframe / IDR
    width: int
    height: int
    codec: str
    timestamp_s: float  # Wall-clock capture time in seconds
    frame_number: int = 0


# ---------------------------------------------------------------------------
# Frame Decoder
# ---------------------------------------------------------------------------


class FrameDecoder:
    """
    Decodes video packets from a stream URL into numpy frames.

    Uses PyAV for FFmpeg-level decoding with optional hardware acceleration.
    Provides both synchronous single-frame decoding and an async generator
    for continuous stream decoding.
    """

    def __init__(
        self,
        codec: str = "h265",
        hw_accel: str = "none",
        target_format: str = "rgb24",
    ) -> None:
        self._codec = codec
        self._hw_accel = hw_accel
        self._target_format = target_format  # rgb24, bgr24
        self._frame_count = 0

    async def decode_stream(
        self,
        stream_url: str,
        max_frames: int = 0,
    ) -> AsyncIterator[DecodedFrame]:
        """
        Async generator that yields decoded frames from a stream URL.

        Runs the blocking PyAV decoder in a thread pool to avoid blocking
        the event loop.

        Args:
            stream_url: Input URL (SRT, RTMP, file, pipe, etc.)
            max_frames: Stop after this many frames (0 = unlimited).
        """
        loop = asyncio.get_running_loop()
        container = None

        try:
            container = await loop.run_in_executor(
                None, self._open_container, stream_url
            )

            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"

            # Set hardware decoder if available
            if self._hw_accel == "nvdec":
                video_stream.codec_context.options = {"hwaccel": "cuda"}

            codec_name = video_stream.codec_context.name
            self._frame_count = 0

            for packet in container.demux(video_stream):
                frames = packet.decode()
                for frame in frames:
                    decoded = await loop.run_in_executor(
                        None, self._convert_frame, frame, codec_name
                    )
                    if decoded is not None:
                        yield decoded

                    if 0 < max_frames <= self._frame_count:
                        return

        except av.error.EOFError:
            logger.info("frame_decoder.eof", url=stream_url)
        except av.error.InvalidDataError as exc:
            logger.error("frame_decoder.invalid_data", url=stream_url, error=str(exc))
        except Exception as exc:
            logger.error("frame_decoder.error", url=stream_url, error=str(exc))
            raise
        finally:
            if container is not None:
                container.close()

    def decode_frame(self, packet: bytes, codec: str = "h264") -> Optional[DecodedFrame]:
        """
        Decode a single raw packet into a frame.

        Useful for decoding individual NAL units or RTP payloads.
        """
        codec_ctx = None
        try:
            codec_obj = av.codec.Codec(codec, "r")
            codec_ctx = av.codec.CodecContext.create(codec_obj)
            codec_ctx.open()

            av_packet = av.Packet(packet)
            frames = codec_ctx.decode(av_packet)

            for frame in frames:
                return self._convert_frame(frame, codec)

            return None
        except Exception as exc:
            logger.error("frame_decoder.decode_frame_error", error=str(exc))
            return None
        finally:
            if codec_ctx is not None:
                codec_ctx.close()

    # -- internal ------------------------------------------------------------

    def _open_container(self, url: str) -> av.container.InputContainer:
        """Open a container with appropriate options for live streaming."""
        options = {
            "analyzeduration": "2000000",  # 2 seconds
            "probesize": "5000000",  # 5 MB
        }

        # SRT-specific options
        if url.startswith("srt://"):
            options["timeout"] = "5000000"  # 5 second timeout

        # RTMP-specific options
        if url.startswith("rtmp://"):
            options["live_start_index"] = "-1"

        container = av.open(url, options=options)
        return container

    def _convert_frame(
        self,
        frame: av.VideoFrame,
        codec_name: str,
    ) -> Optional[DecodedFrame]:
        """Convert an av.VideoFrame to a DecodedFrame with numpy array."""
        try:
            self._frame_count += 1

            # Convert to target pixel format
            if frame.format.name != self._target_format:
                frame = frame.reformat(format=self._target_format)

            # Convert to numpy array
            np_frame = frame.to_ndarray()

            return DecodedFrame(
                frame=np_frame,
                pts=frame.pts or 0,
                dts=frame.dts,
                keyframe=frame.key_frame,
                width=frame.width,
                height=frame.height,
                codec=codec_name,
                timestamp_s=time.time(),
                frame_number=self._frame_count,
            )
        except Exception as exc:
            logger.error(
                "frame_decoder.convert_error",
                error=str(exc),
                frame_number=self._frame_count,
            )
            return None

    @staticmethod
    def yuv420_to_rgb(yuv_frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Convert a YUV420p frame to RGB.

        Args:
            yuv_frame: Raw YUV420p data as 1D uint8 array.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            RGB frame as (H, W, 3) uint8 array.
        """
        # YUV420p layout: Y plane (W*H), U plane (W/2 * H/2), V plane (W/2 * H/2)
        y_size = width * height
        uv_size = (width // 2) * (height // 2)

        y = yuv_frame[:y_size].reshape((height, width)).astype(np.float32)
        u = yuv_frame[y_size : y_size + uv_size].reshape((height // 2, width // 2)).astype(np.float32)
        v = yuv_frame[y_size + uv_size : y_size + 2 * uv_size].reshape((height // 2, width // 2)).astype(np.float32)

        # Upsample U and V
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

        # BT.601 conversion
        r = y + 1.402 * (v - 128.0)
        g = y - 0.344136 * (u - 128.0) - 0.714136 * (v - 128.0)
        b = y + 1.772 * (u - 128.0)

        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    @staticmethod
    def nv12_to_rgb(nv12_frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Convert an NV12 frame to RGB.

        NV12 has a Y plane followed by interleaved UV plane.
        """
        y_size = width * height

        y = nv12_frame[:y_size].reshape((height, width)).astype(np.float32)
        uv = nv12_frame[y_size:].reshape((height // 2, width))

        u = uv[:, 0::2].astype(np.float32)
        v = uv[:, 1::2].astype(np.float32)

        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

        r = y + 1.402 * (v - 128.0)
        g = y - 0.344136 * (u - 128.0) - 0.714136 * (v - 128.0)
        b = y + 1.772 * (u - 128.0)

        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8)
