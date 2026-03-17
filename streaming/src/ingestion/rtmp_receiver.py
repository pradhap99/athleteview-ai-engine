"""
AthleteView AI Platform - RTMP Stream Receiver.

Manages RTMP ingest endpoints using FFmpeg subprocess. Each registered
stream gets a unique stream key; the single FFmpeg/RTMP listener validates
keys and routes incoming connections to the processing pipeline.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Coroutine, Optional

import structlog

from src.config import RTMPConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class RTMPStreamStatus(str, Enum):
    PENDING = "pending"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class RTMPEndpoint:
    """Represents a registered RTMP ingest endpoint."""

    url: str
    stream_id: str
    stream_key: str
    status: RTMPStreamStatus = RTMPStreamStatus.PENDING


@dataclass
class RTMPStats:
    """Real-time statistics for an active RTMP session."""

    bitrate_kbps: float = 0.0
    fps: float = 0.0
    connected_at: Optional[float] = None
    bytes_received: int = 0
    frames_received: int = 0
    dropped_frames: int = 0
    codec: str = ""
    resolution: str = ""
    duration_s: float = 0.0


@dataclass
class _RTMPSession:
    """Internal bookkeeping for a single RTMP stream session."""

    stream_id: str
    endpoint: RTMPEndpoint
    process: Optional[asyncio.subprocess.Process] = None
    stats: RTMPStats = field(default_factory=RTMPStats)
    monitor_task: Optional[asyncio.Task] = None
    reconnect_count: int = 0


# ---------------------------------------------------------------------------
# RTMP Receiver
# ---------------------------------------------------------------------------

class RTMPReceiver:
    """
    Manages RTMP ingest streams via FFmpeg sub-processes.

    Spawns FFmpeg processes that listen for RTMP streams on the configured
    port. Stream-key validation ensures only authorized streams are accepted.
    """

    def __init__(self, config: RTMPConfig) -> None:
        self._config = config
        self._sessions: dict[str, _RTMPSession] = {}
        self._stream_keys: dict[str, str] = {}  # stream_key -> stream_id
        self._on_connect_callbacks: list[Callable[..., Coroutine]] = []
        self._on_disconnect_callbacks: list[Callable[..., Coroutine]] = []
        self._running = False
        self._server_process: Optional[asyncio.subprocess.Process] = None
        self._ffmpeg_binary = "ffmpeg"

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the RTMP receiver, ready to accept stream registrations."""
        self._running = True
        logger.info(
            "rtmp_receiver.started",
            host=self._config.host,
            port=self._config.listen_port,
        )

    async def stop(self) -> None:
        """Gracefully stop all RTMP sessions and the listener."""
        self._running = False

        stream_ids = list(self._sessions.keys())
        for sid in stream_ids:
            await self._stop_session(sid)

        if self._server_process and self._server_process.returncode is None:
            self._server_process.terminate()
            try:
                await asyncio.wait_for(self._server_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._server_process.kill()
                await self._server_process.wait()

        logger.info("rtmp_receiver.stopped", streams_stopped=len(stream_ids))

    # -- public API ----------------------------------------------------------

    async def register_stream(
        self,
        stream_id: str,
        stream_key: Optional[str] = None,
    ) -> RTMPEndpoint:
        """
        Register a new RTMP ingest stream and return the connection endpoint.

        A unique stream key is generated if not provided. The endpoint URL
        includes the stream key for the sender to use in OBS/FFmpeg.
        """
        if stream_id in self._sessions:
            return self._sessions[stream_id].endpoint

        stream_key = stream_key or secrets.token_urlsafe(24)
        rtmp_url = (
            f"rtmp://{self._config.host}:{self._config.listen_port}"
            f"/live/{stream_key}"
        )

        endpoint = RTMPEndpoint(
            url=rtmp_url,
            stream_id=stream_id,
            stream_key=stream_key,
            status=RTMPStreamStatus.PENDING,
        )

        session = _RTMPSession(stream_id=stream_id, endpoint=endpoint)
        self._sessions[stream_id] = session
        self._stream_keys[stream_key] = stream_id

        # Spawn FFmpeg to listen for this particular stream
        await self._start_ffmpeg(session)

        logger.info(
            "rtmp_receiver.stream_registered",
            stream_id=stream_id,
        )
        return endpoint

    def on_connect(self, callback: Callable[..., Coroutine]) -> None:
        """Register a callback invoked when a new RTMP stream connects."""
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable[..., Coroutine]) -> None:
        """Register a callback invoked when an RTMP stream disconnects."""
        self._on_disconnect_callbacks.append(callback)

    async def get_stats(self, stream_id: str) -> RTMPStats:
        """Return real-time statistics for an active RTMP session."""
        session = self._sessions.get(stream_id)
        if session is None:
            raise KeyError(f"No RTMP session for stream_id={stream_id}")
        return session.stats

    def get_active_streams(self) -> dict[str, RTMPEndpoint]:
        """Return all currently registered RTMP endpoints."""
        return {sid: s.endpoint for sid, s in self._sessions.items()}

    def validate_stream_key(self, stream_key: str) -> Optional[str]:
        """Validate an incoming stream key. Returns stream_id or None."""
        return self._stream_keys.get(stream_key)

    # -- internal ------------------------------------------------------------

    async def _start_ffmpeg(self, session: _RTMPSession) -> None:
        """Spawn an FFmpeg process to listen for the RTMP stream."""
        cmd = self._build_ffmpeg_cmd(session)
        logger.debug(
            "rtmp_receiver.ffmpeg_start",
            stream_id=session.stream_id,
            cmd=" ".join(cmd),
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            session.process = process
            session.monitor_task = asyncio.create_task(
                self._monitor_process(session)
            )
        except FileNotFoundError:
            logger.error(
                "rtmp_receiver.ffmpeg_not_found",
                stream_id=session.stream_id,
                binary=self._ffmpeg_binary,
            )
            session.endpoint.status = RTMPStreamStatus.ERROR
        except Exception as exc:
            logger.error(
                "rtmp_receiver.ffmpeg_start_failed",
                stream_id=session.stream_id,
                error=str(exc),
            )
            session.endpoint.status = RTMPStreamStatus.ERROR

    def _build_ffmpeg_cmd(self, session: _RTMPSession) -> list[str]:
        """Build the FFmpeg command for RTMP ingest."""
        rtmp_url = session.endpoint.url
        return [
            self._ffmpeg_binary,
            "-y",
            "-loglevel", "warning",
            "-stats",
            # Listen mode for RTMP
            "-listen", "1",
            "-timeout", str(self._config.timeout_s),
            "-i", rtmp_url,
            # Copy codec (no transcoding on ingest)
            "-c", "copy",
            "-f", "mpegts",
            "pipe:1",
        ]

    async def _monitor_process(self, session: _RTMPSession) -> None:
        """Monitor an FFmpeg process for connection/disconnection events."""
        stream_id = session.stream_id
        process = session.process
        if process is None:
            return

        connected = False
        try:
            while self._running and process.returncode is None:
                try:
                    stderr_line = await asyncio.wait_for(
                        process.stderr.readline(),  # type: ignore[union-attr]
                        timeout=2.0,
                    )
                except asyncio.TimeoutError:
                    continue

                if not stderr_line:
                    break

                line_text = stderr_line.decode(errors="replace").strip()

                if not connected and line_text:
                    connected = True
                    session.endpoint.status = RTMPStreamStatus.CONNECTED
                    session.stats.connected_at = time.time()
                    await self._fire_on_connect(stream_id)

                self._parse_stats(session, line_text)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                "rtmp_receiver.monitor_error",
                stream_id=stream_id,
                error=str(exc),
            )
        finally:
            if connected:
                session.endpoint.status = RTMPStreamStatus.DISCONNECTED
                await self._fire_on_disconnect(stream_id)

            # Auto-reconnect
            if self._running and session.reconnect_count < 5:
                session.reconnect_count += 1
                logger.info(
                    "rtmp_receiver.reconnecting",
                    stream_id=stream_id,
                    attempt=session.reconnect_count,
                )
                await asyncio.sleep(2)
                await self._start_ffmpeg(session)

    def _parse_stats(self, session: _RTMPSession, line: str) -> None:
        """Parse FFmpeg stderr progress output for RTMP stats."""
        if "bitrate=" in line:
            try:
                parts = line.split()
                for part in parts:
                    if part.startswith("bitrate="):
                        bps_str = part.split("=")[1].replace("kbits/s", "")
                        session.stats.bitrate_kbps = float(bps_str)
                    elif part.startswith("size="):
                        size_str = part.split("=")[1].replace("kB", "")
                        session.stats.bytes_received = int(float(size_str) * 1024)
                    elif part.startswith("fps="):
                        fps_str = part.split("=")[1]
                        session.stats.fps = float(fps_str)
            except (ValueError, IndexError):
                pass

        if session.stats.connected_at:
            session.stats.duration_s = time.time() - session.stats.connected_at

    async def _stop_session(self, stream_id: str) -> None:
        """Terminate an FFmpeg process and clean up session state."""
        session = self._sessions.pop(stream_id, None)
        if session is None:
            return

        # Remove stream key mapping
        key_to_remove = None
        for key, sid in self._stream_keys.items():
            if sid == stream_id:
                key_to_remove = key
                break
        if key_to_remove:
            del self._stream_keys[key_to_remove]

        if session.monitor_task and not session.monitor_task.done():
            session.monitor_task.cancel()
            try:
                await session.monitor_task
            except asyncio.CancelledError:
                pass

        if session.process and session.process.returncode is None:
            session.process.terminate()
            try:
                await asyncio.wait_for(session.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                session.process.kill()
                await session.process.wait()

        logger.info("rtmp_receiver.session_stopped", stream_id=stream_id)

    async def _fire_on_connect(self, stream_id: str) -> None:
        for cb in self._on_connect_callbacks:
            try:
                await cb(stream_id)
            except Exception as exc:
                logger.error(
                    "rtmp_receiver.on_connect_callback_error",
                    stream_id=stream_id,
                    error=str(exc),
                )

    async def _fire_on_disconnect(self, stream_id: str) -> None:
        for cb in self._on_disconnect_callbacks:
            try:
                await cb(stream_id)
            except Exception as exc:
                logger.error(
                    "rtmp_receiver.on_disconnect_callback_error",
                    stream_id=stream_id,
                    error=str(exc),
                )
