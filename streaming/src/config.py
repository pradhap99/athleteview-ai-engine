"""
AthleteView AI Platform - Streaming Service Configuration.

Centralized configuration using Pydantic BaseSettings with environment variable support.
Covers SRT, RTMP, HLS, WebRTC, encoding, FFmpeg, Kafka, and Redis settings.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class H264Preset(str, Enum):
    ULTRAFAST = "ultrafast"
    SUPERFAST = "superfast"
    VERYFAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    VERYSLOW = "veryslow"


class H265Preset(str, Enum):
    ULTRAFAST = "ultrafast"
    SUPERFAST = "superfast"
    VERYFAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    VERYSLOW = "veryslow"


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

class SRTConfig(BaseSettings):
    """SRT (Secure Reliable Transport) ingest settings."""

    listen_port: int = Field(default=9000, description="UDP port for SRT listener")
    latency_ms: int = Field(default=120, description="SRT latency in milliseconds")
    max_bandwidth: int = Field(
        default=0, description="Max bandwidth in bytes/s (0=unlimited)"
    )
    encryption: bool = Field(default=True, description="Enable SRT AES encryption")
    passphrase_length: int = Field(default=16, description="Default passphrase length")
    peer_idle_timeout_ms: int = Field(
        default=5000, description="Peer idle timeout before disconnect"
    )
    payload_size: int = Field(default=1316, description="SRT payload size in bytes")
    host: str = Field(default="0.0.0.0", description="SRT listen address")

    model_config = {"env_prefix": "SRT_"}


class RTMPConfig(BaseSettings):
    """RTMP ingest settings."""

    listen_port: int = Field(default=1935, description="TCP port for RTMP listener")
    chunk_size: int = Field(default=4096, description="RTMP chunk size")
    host: str = Field(default="0.0.0.0", description="RTMP listen address")
    max_connections: int = Field(default=100, description="Max concurrent RTMP streams")
    timeout_s: int = Field(default=30, description="RTMP connection timeout in seconds")

    model_config = {"env_prefix": "RTMP_"}


class BitrateRung(BaseSettings):
    """Single rung in the ABR bitrate ladder."""

    height: int
    width: int
    video_bitrate_kbps: int
    audio_bitrate_kbps: int = 128
    framerate: int = 30
    label: str = ""

    model_config = {"env_prefix": ""}


class HLSConfig(BaseSettings):
    """HLS / LL-HLS packaging settings."""

    segment_duration: float = Field(default=4.0, description="Segment duration in seconds")
    playlist_size: int = Field(default=5, description="Number of segments in live playlist")
    output_dir: str = Field(default="/app/hls_output", description="HLS segment output directory")
    ll_hls_enabled: bool = Field(default=True, description="Enable Low-Latency HLS")
    partial_segment_duration: float = Field(
        default=0.5, description="LL-HLS partial segment duration"
    )
    delete_old_segments: bool = Field(default=True, description="Remove segments outside playlist window")
    cdn_origin_url: Optional[str] = Field(default=None, description="CDN origin base URL")

    model_config = {"env_prefix": "HLS_"}


class WebRTCConfig(BaseSettings):
    """WebRTC distribution settings."""

    stun_servers: list[str] = Field(
        default=["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"],
        description="STUN server URIs",
    )
    turn_servers: list[str] = Field(default=[], description="TURN server URIs")
    turn_username: Optional[str] = Field(default=None, description="TURN username")
    turn_password: Optional[str] = Field(default=None, description="TURN credential")
    port_range_min: int = Field(default=40000, description="Minimum UDP port for WebRTC")
    port_range_max: int = Field(default=41000, description="Maximum UDP port for WebRTC")
    max_viewers_per_stream: int = Field(default=100, description="Maximum WebRTC viewers per stream")

    model_config = {"env_prefix": "WEBRTC_"}


class EncodingConfig(BaseSettings):
    """Encoding / transcoding settings."""

    h264_preset: H264Preset = Field(default=H264Preset.VERYFAST, description="x264 preset")
    h265_preset: H265Preset = Field(default=H265Preset.FAST, description="x265 preset")
    hardware_accel: str = Field(
        default="none",
        description="Hardware acceleration: none, nvenc, vaapi, qsv",
    )
    keyint: int = Field(default=60, description="Keyframe interval in frames")
    bitrate_ladder: list[dict] = Field(
        default=[
            {"height": 360, "width": 640, "video_bitrate_kbps": 800, "audio_bitrate_kbps": 96, "framerate": 30, "label": "360p"},
            {"height": 480, "width": 854, "video_bitrate_kbps": 1400, "audio_bitrate_kbps": 128, "framerate": 30, "label": "480p"},
            {"height": 720, "width": 1280, "video_bitrate_kbps": 2800, "audio_bitrate_kbps": 128, "framerate": 30, "label": "720p"},
            {"height": 1080, "width": 1920, "video_bitrate_kbps": 5000, "audio_bitrate_kbps": 192, "framerate": 30, "label": "1080p"},
            {"height": 2160, "width": 3840, "video_bitrate_kbps": 16000, "audio_bitrate_kbps": 256, "framerate": 30, "label": "4K"},
        ],
        description="ABR bitrate ladder",
    )

    model_config = {"env_prefix": "ENCODING_"}


class FFmpegConfig(BaseSettings):
    """FFmpeg binary and default settings."""

    binary_path: str = Field(default="ffmpeg", description="Path to ffmpeg binary")
    ffprobe_path: str = Field(default="ffprobe", description="Path to ffprobe binary")
    loglevel: str = Field(default="warning", description="FFmpeg log level")
    threads: int = Field(default=0, description="FFmpeg threads (0=auto)")
    max_muxing_queue_size: int = Field(default=1024, description="Max muxing queue size")

    model_config = {"env_prefix": "FFMPEG_"}


class KafkaConfig(BaseSettings):
    """Kafka connection settings for stream events."""

    bootstrap_servers: str = Field(
        default="localhost:9092", description="Kafka bootstrap servers"
    )
    topic_stream_events: str = Field(
        default="streaming.events", description="Topic for stream lifecycle events"
    )
    topic_stream_stats: str = Field(
        default="streaming.stats", description="Topic for stream statistics"
    )
    topic_frame_events: str = Field(
        default="streaming.frames", description="Topic for frame-level events"
    )
    producer_acks: str = Field(default="1", description="Producer ack policy")
    client_id: str = Field(default="streaming-service", description="Kafka client ID")

    model_config = {"env_prefix": "KAFKA_"}


class RedisConfig(BaseSettings):
    """Redis connection settings."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    key_prefix: str = Field(default="streaming:", description="Key prefix for namespacing")
    socket_timeout: float = Field(default=5.0, description="Socket timeout in seconds")

    model_config = {"env_prefix": "REDIS_"}


class SRTBroadcastConfig(BaseSettings):
    """SRT broadcast/output settings."""

    default_bitrate_kbps: int = Field(
        default=20000, description="Default broadcast bitrate in kbps"
    )
    codec: str = Field(default="libx265", description="Broadcast codec (libx264/libx265)")
    encryption: bool = Field(default=True, description="Enable SRT encryption on output")

    model_config = {"env_prefix": "SRT_BROADCAST_"}


class RestreamConfig(BaseSettings):
    """Restreaming configuration."""

    max_targets_per_stream: int = Field(
        default=5, description="Max restream targets per source stream"
    )
    reconnect_delay_s: int = Field(
        default=5, description="Delay before reconnection attempt"
    )
    max_reconnect_attempts: int = Field(
        default=10, description="Max reconnect attempts before giving up"
    )

    model_config = {"env_prefix": "RESTREAM_"}


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------

class StreamingConfig(BaseSettings):
    """Root configuration aggregating all sub-configs for the Streaming Service."""

    # Service identity
    service_name: str = Field(default="athleteview-streaming", description="Service name")
    api_host: str = Field(default="0.0.0.0", description="API bind address")
    api_port: int = Field(default=8002, description="API port")
    log_level: str = Field(default="info", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")

    # Sub-configs
    srt: SRTConfig = Field(default_factory=SRTConfig)
    rtmp: RTMPConfig = Field(default_factory=RTMPConfig)
    hls: HLSConfig = Field(default_factory=HLSConfig)
    webrtc: WebRTCConfig = Field(default_factory=WebRTCConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    ffmpeg: FFmpegConfig = Field(default_factory=FFmpegConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    srt_broadcast: SRTBroadcastConfig = Field(default_factory=SRTBroadcastConfig)
    restream: RestreamConfig = Field(default_factory=RestreamConfig)

    model_config = {
        "env_prefix": "STREAMING_",
        "env_nested_delimiter": "__",
    }


def get_config() -> StreamingConfig:
    """Factory to build the configuration from environment variables."""
    return StreamingConfig()
