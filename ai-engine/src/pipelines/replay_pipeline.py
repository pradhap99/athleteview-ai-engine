"""
AthleteView AI Engine - Replay Pipeline
Offline replay generation with full-quality enhancement, stabilization,
slow-motion interpolation, and optional 3D reconstruction.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog

from src.config import Settings
from src.models.highlight_detector import HighlightEvent
from src.models.registry import ModelRegistry
from src.utils.metrics import get_metrics

logger = structlog.get_logger(__name__)


@dataclass
class EnhancedReplay:
    """Result container for a fully processed replay sequence."""

    frames: list[np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    format: str = "raw_frames"
    original_frame_count: int = 0
    output_frame_count: int = 0
    resolution: tuple[int, int] = (0, 0)
    fps: float = 30.0
    stages_applied: list[str] = field(default_factory=list)
    total_processing_seconds: float = 0.0


@dataclass
class ReplayProgress:
    """Progress information for async replay generation."""

    stage: str = "pending"
    stage_index: int = 0
    total_stages: int = 0
    percent_complete: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0


class ReplayPipeline:
    """
    Generates high-quality enhanced replays from raw frame sequences.

    Unlike the live pipeline, this runs asynchronously without real-time
    constraints, applying the full suite of enhancements:

        1. Video stabilization (all frames)
        2. Super-resolution upscaling
        3. Slow-motion frame interpolation
        4. 3D Gaussian splatting reconstruction (if multi-camera)

    Supports progress tracking via a callback mechanism.
    """

    def __init__(self, registry: ModelRegistry, config: Settings) -> None:
        self._registry = registry
        self._config = config
        self._metrics = get_metrics()
        self._progress = ReplayProgress()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_replay(
        self,
        frames: list[np.ndarray],
        highlight_event: HighlightEvent | None = None,
        config: dict[str, Any] | None = None,
    ) -> EnhancedReplay:
        """
        Generate a fully enhanced replay from a sequence of frames.

        Args:
            frames: Ordered list of BGR HWC uint8 frames.
            highlight_event: Optional highlight event that triggered this replay.
            config: Optional override configuration dict with keys:
                - stabilize (bool): Apply stabilization. Default True.
                - super_resolution (bool): Apply SR upscaling. Default True.
                - slow_motion_factor (int): Frame interpolation multiplier. Default 2.
                - reconstruct_3d (bool): Run 3D reconstruction. Default False.
                - camera_params (list): Camera parameters for 3D reconstruction.
                - output_fps (float): Desired output FPS. Default 60.0.

        Returns:
            EnhancedReplay with all enhanced frames and metadata.
        """
        if not frames:
            raise ValueError("At least one frame is required for replay generation")

        replay_config = config or {}
        do_stabilize = replay_config.get("stabilize", True)
        do_sr = replay_config.get("super_resolution", True)
        slow_motion_factor = replay_config.get("slow_motion_factor", 2)
        do_3d = replay_config.get("reconstruct_3d", False)
        camera_params = replay_config.get("camera_params", [])
        output_fps = replay_config.get("output_fps", 60.0)

        pipeline_start = time.perf_counter()
        stages_applied: list[str] = []
        working_frames = [f.copy() for f in frames]

        # Determine total stages
        total_stages = 0
        if do_stabilize and self._registry.is_loaded("video_stabilizer"):
            total_stages += 1
        if do_sr and self._registry.is_loaded("super_resolution"):
            total_stages += 1
        if slow_motion_factor > 1:
            total_stages += 1
        if do_3d and self._registry.is_loaded("gaussian_splat") and camera_params:
            total_stages += 1

        self._progress = ReplayProgress(total_stages=max(total_stages, 1))
        stage_idx = 0

        # --- Stage 1: Stabilization ---
        if do_stabilize and self._registry.is_loaded("video_stabilizer"):
            self._update_progress("stabilization", stage_idx, total_stages, pipeline_start)
            working_frames = await self._stabilize_all(working_frames)
            stages_applied.append("stabilization")
            stage_idx += 1

        # --- Stage 2: Super-resolution ---
        if do_sr and self._registry.is_loaded("super_resolution"):
            self._update_progress("super_resolution", stage_idx, total_stages, pipeline_start)
            working_frames = await self._upscale_all(working_frames)
            stages_applied.append("super_resolution")
            stage_idx += 1

        # --- Stage 3: Slow-motion interpolation ---
        if slow_motion_factor > 1:
            self._update_progress("slow_motion", stage_idx, total_stages, pipeline_start)
            working_frames = await self._interpolate_slow_motion(
                working_frames, slow_motion_factor
            )
            stages_applied.append(f"slow_motion_x{slow_motion_factor}")
            stage_idx += 1

        # --- Stage 4: 3D reconstruction ---
        rendered_views: list[np.ndarray] = []
        if do_3d and self._registry.is_loaded("gaussian_splat") and camera_params:
            self._update_progress("3d_reconstruction", stage_idx, total_stages, pipeline_start)
            rendered_views = await self._reconstruct_3d(working_frames, camera_params)
            stages_applied.append("3d_reconstruction")
            stage_idx += 1

        total_time = time.perf_counter() - pipeline_start
        self._update_progress("complete", stage_idx, total_stages, pipeline_start)

        h, w = working_frames[0].shape[:2] if working_frames else (0, 0)
        num_output_frames = len(working_frames)
        duration = num_output_frames / output_fps if output_fps > 0 else 0.0

        metadata: dict[str, Any] = {
            "original_frame_count": len(frames),
            "output_frame_count": num_output_frames,
            "stages": stages_applied,
            "slow_motion_factor": slow_motion_factor,
            "output_fps": output_fps,
        }
        if highlight_event is not None:
            metadata["highlight"] = highlight_event.to_dict()
        if rendered_views:
            metadata["rendered_3d_views"] = len(rendered_views)

        logger.info(
            "Replay generation complete",
            stages=stages_applied,
            input_frames=len(frames),
            output_frames=num_output_frames,
            duration_s=round(duration, 2),
            processing_s=round(total_time, 2),
        )

        return EnhancedReplay(
            frames=working_frames,
            metadata=metadata,
            duration_seconds=duration,
            format="raw_frames",
            original_frame_count=len(frames),
            output_frame_count=num_output_frames,
            resolution=(w, h),
            fps=output_fps,
            stages_applied=stages_applied,
            total_processing_seconds=round(total_time, 3),
        )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    async def _stabilize_all(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Stabilize all frames in sequence."""
        stabilizer = self._registry.get_model("video_stabilizer")
        stabilizer.reset()

        def _run() -> list[np.ndarray]:
            return stabilizer.stabilize_batch(frames)

        with self._metrics.track_pipeline_stage("replay", "stabilization"):
            result = await asyncio.get_event_loop().run_in_executor(None, _run)

        logger.debug("Stabilization complete", frame_count=len(result))
        return result

    async def _upscale_all(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply super-resolution to all frames."""
        sr_model = self._registry.get_model("super_resolution")

        def _run() -> list[np.ndarray]:
            return sr_model.enhance_batch(frames)

        with self._metrics.track_pipeline_stage("replay", "super_resolution"):
            result = await asyncio.get_event_loop().run_in_executor(None, _run)

        logger.debug("Super-resolution complete", frame_count=len(result))
        return result

    async def _interpolate_slow_motion(
        self, frames: list[np.ndarray], factor: int
    ) -> list[np.ndarray]:
        """
        Generate intermediate frames for slow-motion effect.

        Uses linear blending between consecutive frames. A production system
        would use RIFE or similar optical-flow-based interpolation.
        """
        import cv2

        if len(frames) < 2 or factor <= 1:
            return frames

        def _run() -> list[np.ndarray]:
            interpolated: list[np.ndarray] = []
            for i in range(len(frames) - 1):
                interpolated.append(frames[i])
                for step in range(1, factor):
                    alpha = step / factor
                    # Weighted blend for intermediate frame
                    blended = cv2.addWeighted(
                        frames[i], 1.0 - alpha, frames[i + 1], alpha, 0
                    )
                    interpolated.append(blended)
            interpolated.append(frames[-1])
            return interpolated

        with self._metrics.track_pipeline_stage("replay", "slow_motion"):
            result = await asyncio.get_event_loop().run_in_executor(None, _run)

        logger.debug(
            "Slow-motion interpolation complete",
            input_frames=len(frames),
            output_frames=len(result),
            factor=factor,
        )
        return result

    async def _reconstruct_3d(
        self,
        frames: list[np.ndarray],
        camera_params_raw: list[dict[str, Any]],
    ) -> list[np.ndarray]:
        """Run 3D Gaussian splatting and render novel views."""
        from src.models.gaussian_splat import CameraParams

        reconstructor = self._registry.get_model("gaussian_splat")

        camera_params: list[CameraParams] = []
        for cam in camera_params_raw:
            camera_params.append(
                CameraParams(
                    extrinsic=np.array(cam["extrinsic"], dtype=np.float64).reshape(4, 4),
                    intrinsic=np.array(cam["intrinsic"], dtype=np.float64).reshape(3, 3),
                    width=cam.get("width", frames[0].shape[1]),
                    height=cam.get("height", frames[0].shape[0]),
                )
            )

        # Use a subset of frames for reconstruction if the sequence is very long
        max_recon_frames = min(len(frames), len(camera_params))
        recon_frames = frames[:max_recon_frames]
        recon_cameras = camera_params[:max_recon_frames]

        def _run():
            scene = reconstructor.reconstruct(recon_frames, recon_cameras)
            rendered: list[np.ndarray] = []
            for cam in recon_cameras:
                rendered.append(reconstructor.render_novel_view(scene, cam))
            return rendered

        with self._metrics.track_pipeline_stage("replay", "3d_reconstruction"):
            result = await asyncio.get_event_loop().run_in_executor(None, _run)

        logger.debug("3D reconstruction complete", rendered_views=len(result))
        return result

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def _update_progress(
        self, stage: str, stage_idx: int, total_stages: int, start_time: float
    ) -> None:
        """Update progress information for external consumers."""
        elapsed = time.perf_counter() - start_time
        pct = (stage_idx / max(total_stages, 1)) * 100.0

        eta = 0.0
        if stage_idx > 0 and pct < 100.0:
            time_per_stage = elapsed / stage_idx
            remaining_stages = total_stages - stage_idx
            eta = time_per_stage * remaining_stages

        self._progress = ReplayProgress(
            stage=stage,
            stage_index=stage_idx,
            total_stages=total_stages,
            percent_complete=round(pct, 1),
            elapsed_seconds=round(elapsed, 2),
            eta_seconds=round(eta, 2),
        )

    @property
    def progress(self) -> ReplayProgress:
        """Return current replay generation progress."""
        return self._progress
