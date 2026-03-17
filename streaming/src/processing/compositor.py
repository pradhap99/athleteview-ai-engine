"""
AthleteView AI Platform - Multi-View Compositor.

Composites multiple video frames into grid, picture-in-picture, and
side-by-side layouts with smooth animated transitions. Used for
multi-camera views, tactical displays, and broadcast production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LayoutPreset(str, Enum):
    SINGLE = "single"
    DUAL = "dual"
    QUAD = "quad"
    MOSAIC = "mosaic"  # 1 large + 3 small
    PIP = "pip"
    SIDE_BY_SIDE = "side_by_side"


class TransitionType(str, Enum):
    CUT = "cut"
    CROSSFADE = "crossfade"
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    WIPE_UP = "wipe_up"
    WIPE_DOWN = "wipe_down"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"


@dataclass
class CompositorConfig:
    """Configuration for the multi-view compositor."""

    output_width: int = 1920
    output_height: int = 1080
    background_color: tuple[int, int, int] = (20, 20, 20)
    border_width: int = 2
    border_color: tuple[int, int, int] = (100, 100, 100)
    label_font_scale: float = 0.5
    label_color: tuple[int, int, int] = (255, 255, 255)
    pip_width_ratio: float = 0.25  # PiP size as fraction of output
    pip_height_ratio: float = 0.25
    pip_margin: int = 20
    pip_border_width: int = 2
    pip_border_color: tuple[int, int, int] = (0, 200, 255)
    transition_duration_frames: int = 15


# ---------------------------------------------------------------------------
# Multi-View Compositor
# ---------------------------------------------------------------------------


class MultiViewCompositor:
    """
    Composites multiple camera feeds into production layouts.

    Supports grid, picture-in-picture, side-by-side, and mosaic layouts
    with smooth animated transitions between layout changes.
    """

    def __init__(self, config: Optional[CompositorConfig] = None) -> None:
        self._config = config or CompositorConfig()
        self._current_layout = LayoutPreset.SINGLE
        self._transition_progress: float = 1.0  # 1.0 = no transition active

    @property
    def output_size(self) -> tuple[int, int]:
        return (self._config.output_width, self._config.output_height)

    # -- Grid layouts --------------------------------------------------------

    def compose_grid(
        self,
        frames: dict[str, np.ndarray],
        layout: str = "auto",
        labels: Optional[dict[str, str]] = None,
    ) -> np.ndarray:
        """
        Arrange frames in a grid layout.

        Supported layouts: "2x1", "2x2", "3x3", "1+3" (mosaic), "auto".
        'auto' selects based on the number of frames.
        """
        cfg = self._config
        canvas = self._blank_canvas()
        n = len(frames)

        if n == 0:
            return canvas

        frame_list = list(frames.items())

        if layout == "auto":
            layout = self._auto_layout(n)

        if layout == "1x1" or n == 1:
            sid, f = frame_list[0]
            resized = self._resize(f, cfg.output_width, cfg.output_height)
            canvas[:] = resized
            if labels and sid in labels:
                self._draw_label(canvas, labels[sid], 10, 30)

        elif layout == "2x1":
            cell_w = cfg.output_width // 2
            cell_h = cfg.output_height
            for i, (sid, f) in enumerate(frame_list[:2]):
                resized = self._resize(f, cell_w - cfg.border_width, cell_h)
                x0 = i * cell_w
                canvas[0:cell_h, x0 : x0 + resized.shape[1]] = resized
                if labels and sid in labels:
                    self._draw_label(canvas, labels[sid], x0 + 10, 30)
            # Vertical divider
            cv2.line(
                canvas, (cell_w, 0), (cell_w, cfg.output_height),
                cfg.border_color, cfg.border_width,
            )

        elif layout == "2x2":
            cell_w = cfg.output_width // 2
            cell_h = cfg.output_height // 2
            for i, (sid, f) in enumerate(frame_list[:4]):
                row, col = divmod(i, 2)
                resized = self._resize(
                    f, cell_w - cfg.border_width, cell_h - cfg.border_width
                )
                x0 = col * cell_w
                y0 = row * cell_h
                rh, rw = resized.shape[:2]
                canvas[y0 : y0 + rh, x0 : x0 + rw] = resized
                if labels and sid in labels:
                    self._draw_label(canvas, labels[sid], x0 + 10, y0 + 25)
            # Grid lines
            mid_x = cfg.output_width // 2
            mid_y = cfg.output_height // 2
            cv2.line(canvas, (mid_x, 0), (mid_x, cfg.output_height), cfg.border_color, cfg.border_width)
            cv2.line(canvas, (0, mid_y), (cfg.output_width, mid_y), cfg.border_color, cfg.border_width)

        elif layout == "3x3":
            cell_w = cfg.output_width // 3
            cell_h = cfg.output_height // 3
            for i, (sid, f) in enumerate(frame_list[:9]):
                row, col = divmod(i, 3)
                resized = self._resize(
                    f, cell_w - cfg.border_width, cell_h - cfg.border_width
                )
                x0 = col * cell_w
                y0 = row * cell_h
                rh, rw = resized.shape[:2]
                canvas[y0 : y0 + rh, x0 : x0 + rw] = resized
                if labels and sid in labels:
                    self._draw_label(canvas, labels[sid], x0 + 5, y0 + 20)
            # Grid lines
            for c in range(1, 3):
                x = c * cell_w
                cv2.line(canvas, (x, 0), (x, cfg.output_height), cfg.border_color, cfg.border_width)
            for r in range(1, 3):
                y = r * cell_h
                cv2.line(canvas, (0, y), (cfg.output_width, y), cfg.border_color, cfg.border_width)

        elif layout == "1+3":
            # Mosaic: 1 large (left 2/3) + 3 small stacked (right 1/3)
            main_w = int(cfg.output_width * 2 / 3)
            side_w = cfg.output_width - main_w
            side_h = cfg.output_height // 3

            if frame_list:
                sid, f = frame_list[0]
                resized = self._resize(f, main_w - cfg.border_width, cfg.output_height)
                canvas[0 : cfg.output_height, 0 : resized.shape[1]] = resized
                if labels and sid in labels:
                    self._draw_label(canvas, labels[sid], 10, 30)

            for i, (sid, f) in enumerate(frame_list[1:4]):
                resized = self._resize(f, side_w - cfg.border_width, side_h - cfg.border_width)
                y0 = i * side_h
                rh, rw = resized.shape[:2]
                canvas[y0 : y0 + rh, main_w : main_w + rw] = resized
                if labels and sid in labels:
                    self._draw_label(canvas, labels[sid], main_w + 5, y0 + 20)

            # Dividers
            cv2.line(canvas, (main_w, 0), (main_w, cfg.output_height), cfg.border_color, cfg.border_width)
            for r in range(1, 3):
                y = r * side_h
                cv2.line(canvas, (main_w, y), (cfg.output_width, y), cfg.border_color, cfg.border_width)

        return canvas

    # -- Picture-in-Picture --------------------------------------------------

    def compose_pip(
        self,
        main_frame: np.ndarray,
        pip_frames: list[np.ndarray],
        pip_positions: Optional[list[tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Overlay PiP thumbnails on top of a main frame.

        Default positions are bottom-right stacking leftward.
        """
        cfg = self._config
        canvas = self._resize(main_frame, cfg.output_width, cfg.output_height)

        pip_w = int(cfg.output_width * cfg.pip_width_ratio)
        pip_h = int(cfg.output_height * cfg.pip_height_ratio)

        for i, pf in enumerate(pip_frames):
            resized = self._resize(pf, pip_w, pip_h)

            if pip_positions and i < len(pip_positions):
                px, py = pip_positions[i]
            else:
                # Default: bottom-right, stacking leftward
                px = cfg.output_width - cfg.pip_margin - pip_w - i * (pip_w + cfg.pip_margin)
                py = cfg.output_height - cfg.pip_margin - pip_h

            # Clamp to canvas
            px = max(0, min(px, cfg.output_width - pip_w))
            py = max(0, min(py, cfg.output_height - pip_h))

            rh, rw = resized.shape[:2]
            canvas[py : py + rh, px : px + rw] = resized

            # PiP border
            cv2.rectangle(
                canvas,
                (px, py), (px + rw, py + rh),
                cfg.pip_border_color,
                cfg.pip_border_width,
            )

        return canvas

    # -- Side by side --------------------------------------------------------

    def compose_side_by_side(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        labels: Optional[tuple[str, str]] = None,
    ) -> np.ndarray:
        """Place two frames side by side."""
        cfg = self._config
        half_w = cfg.output_width // 2
        canvas = self._blank_canvas()

        r1 = self._resize(frame1, half_w - cfg.border_width, cfg.output_height)
        r2 = self._resize(frame2, half_w - cfg.border_width, cfg.output_height)

        canvas[0 : r1.shape[0], 0 : r1.shape[1]] = r1
        canvas[0 : r2.shape[0], half_w : half_w + r2.shape[1]] = r2

        # Divider
        cv2.line(
            canvas, (half_w, 0), (half_w, cfg.output_height),
            cfg.border_color, cfg.border_width,
        )

        if labels:
            self._draw_label(canvas, labels[0], 10, 30)
            self._draw_label(canvas, labels[1], half_w + 10, 30)

        return canvas

    # -- Transitions ---------------------------------------------------------

    def transition(
        self,
        from_frame: np.ndarray,
        to_frame: np.ndarray,
        transition_type: TransitionType = TransitionType.CROSSFADE,
        progress: float = 0.5,
    ) -> np.ndarray:
        """
        Apply a transition effect between two frames.

        Args:
            from_frame: Outgoing frame.
            to_frame: Incoming frame.
            transition_type: Type of transition effect.
            progress: 0.0 = fully from_frame, 1.0 = fully to_frame.
        """
        cfg = self._config
        progress = max(0.0, min(1.0, progress))

        f1 = self._resize(from_frame, cfg.output_width, cfg.output_height)
        f2 = self._resize(to_frame, cfg.output_width, cfg.output_height)

        if transition_type == TransitionType.CUT:
            return f2 if progress >= 0.5 else f1

        if transition_type == TransitionType.CROSSFADE:
            return self._crossfade(f1, f2, progress)

        if transition_type == TransitionType.WIPE_LEFT:
            return self._wipe(f1, f2, progress, direction="left")

        if transition_type == TransitionType.WIPE_RIGHT:
            return self._wipe(f1, f2, progress, direction="right")

        if transition_type == TransitionType.WIPE_UP:
            return self._wipe(f1, f2, progress, direction="up")

        if transition_type == TransitionType.WIPE_DOWN:
            return self._wipe(f1, f2, progress, direction="down")

        if transition_type == TransitionType.SLIDE_LEFT:
            return self._slide(f1, f2, progress, direction="left")

        if transition_type == TransitionType.SLIDE_RIGHT:
            return self._slide(f1, f2, progress, direction="right")

        # Fallback: crossfade
        return self._crossfade(f1, f2, progress)

    # -- Internal helpers ----------------------------------------------------

    def _blank_canvas(self) -> np.ndarray:
        """Create a blank canvas filled with the background color."""
        cfg = self._config
        canvas = np.full(
            (cfg.output_height, cfg.output_width, 3),
            cfg.background_color,
            dtype=np.uint8,
        )
        return canvas

    @staticmethod
    def _resize(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize frame to target dimensions, maintaining aspect ratio with letterbox."""
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox if needed
        if new_w == target_w and new_h == target_h:
            return resized

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        return canvas

    def _draw_label(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
    ) -> None:
        """Draw a text label with a dark background."""
        cfg = self._config
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, cfg.label_font_scale, 1
        )
        cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(
            img, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.label_font_scale,
            cfg.label_color, 1, cv2.LINE_AA,
        )

    @staticmethod
    def _crossfade(f1: np.ndarray, f2: np.ndarray, progress: float) -> np.ndarray:
        """Alpha-blend crossfade."""
        return cv2.addWeighted(f1, 1 - progress, f2, progress, 0)

    @staticmethod
    def _wipe(
        f1: np.ndarray,
        f2: np.ndarray,
        progress: float,
        direction: str,
    ) -> np.ndarray:
        """Hard-edge wipe transition."""
        h, w = f1.shape[:2]
        result = f1.copy()

        if direction == "left":
            split = int(w * progress)
            result[:, :split] = f2[:, :split]
        elif direction == "right":
            split = int(w * (1 - progress))
            result[:, split:] = f2[:, split:]
        elif direction == "up":
            split = int(h * progress)
            result[:split, :] = f2[:split, :]
        elif direction == "down":
            split = int(h * (1 - progress))
            result[split:, :] = f2[split:, :]

        return result

    @staticmethod
    def _slide(
        f1: np.ndarray,
        f2: np.ndarray,
        progress: float,
        direction: str,
    ) -> np.ndarray:
        """Sliding transition where both frames move."""
        h, w = f1.shape[:2]
        result = np.zeros_like(f1)
        offset = int(w * progress)

        if direction == "left":
            # f1 slides left, f2 slides in from right
            if w - offset > 0:
                result[:, : w - offset] = f1[:, offset:]
            if offset > 0:
                result[:, w - offset :] = f2[:, : offset]
        elif direction == "right":
            # f1 slides right, f2 slides in from left
            if w - offset > 0:
                result[:, offset:] = f1[:, : w - offset]
            if offset > 0:
                result[:, :offset] = f2[:, w - offset :]

        return result

    def _auto_layout(self, n: int) -> str:
        """Select layout based on number of frames."""
        if n <= 1:
            return "1x1"
        elif n == 2:
            return "2x1"
        elif n <= 4:
            return "2x2"
        else:
            return "3x3"
