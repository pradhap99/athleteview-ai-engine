"""
AthleteView — Video Stabilization Module
Uses OpenCV feature matching + affine transform for stabilization.
"""

import cv2
import numpy as np


class VideoStabilizer:
    """Stabilize video using feature-based motion estimation."""

    def __init__(self, smoothing_radius=15):
        self.smoothing_radius = smoothing_radius
        self.prev_gray = None
        self.transforms = []
        self.trajectory = []

    def reset(self):
        """Reset stabilizer state."""
        self.prev_gray = None
        self.transforms = []
        self.trajectory = []

    def estimate_motion(self, frame):
        """
        Estimate motion between current and previous frame.
        Returns: (dx, dy, da) translation and rotation delta
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0, 0, 0

        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=200, qualityLevel=0.01,
            minDistance=30, blockSize=3
        )

        if prev_pts is None or len(prev_pts) < 10:
            self.prev_gray = gray
            return 0, 0, 0

        # Track features to current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None
        )

        # Filter valid points
        idx = np.where(status == 1)[0]
        if len(idx) < 10:
            self.prev_gray = gray
            return 0, 0, 0

        prev_valid = prev_pts[idx]
        curr_valid = curr_pts[idx]

        # Estimate affine transform
        m, _ = cv2.estimateAffinePartial2D(prev_valid, curr_valid)

        if m is None:
            self.prev_gray = gray
            return 0, 0, 0

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        self.prev_gray = gray
        return dx, dy, da

    def stabilize_frame(self, frame, dx, dy, da, smooth_dx, smooth_dy, smooth_da):
        """Apply stabilization transform to frame."""
        h, w = frame.shape[:2]

        # Compute difference between smoothed and original trajectory
        diff_dx = smooth_dx - dx
        diff_dy = smooth_dy - dy
        diff_da = smooth_da - da

        # Build correction transform
        cos_a = np.cos(diff_da)
        sin_a = np.sin(diff_da)
        m = np.array([
            [cos_a, -sin_a, diff_dx],
            [sin_a, cos_a, diff_dy]
        ], dtype=np.float64)

        # Apply transform
        stabilized = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return stabilized


def two_pass_stabilize(input_path, output_path, smoothing=15):
    """
    Two-pass video stabilization.

    Pass 1: Compute motion trajectory
    Pass 2: Apply smoothed corrections

    Args:
        input_path: path to input video
        output_path: path to output stabilized video
        smoothing: smoothing window radius
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stab = VideoStabilizer(smoothing_radius=smoothing)

    # Pass 1: Compute transforms
    print(f"[Stabilizer] Pass 1: Computing motion for {n_frames} frames...")
    transforms = []
    trajectory = []
    cum_x, cum_y, cum_a = 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dx, dy, da = stab.estimate_motion(frame)
        transforms.append((dx, dy, da))

        cum_x += dx
        cum_y += dy
        cum_a += da
        trajectory.append((cum_x, cum_y, cum_a))

    cap.release()

    if len(transforms) == 0:
        print("[Stabilizer] No frames found!")
        return

    # Smooth trajectory
    trajectory = np.array(trajectory)
    smoothed = np.copy(trajectory)

    for i in range(3):  # x, y, angle
        kernel = np.ones(2 * smoothing + 1) / (2 * smoothing + 1)
        padded = np.pad(trajectory[:, i], smoothing, mode='edge')
        smoothed[:, i] = np.convolve(padded, kernel, mode='valid')[:len(trajectory)]

    # Pass 2: Apply corrections
    print(f"[Stabilizer] Pass 2: Applying stabilization...")
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i in range(len(transforms)):
        ret, frame = cap.read()
        if not ret:
            break

        dx, dy, da = transforms[i]
        cum_x_orig = trajectory[i, 0]
        cum_y_orig = trajectory[i, 1]
        cum_a_orig = trajectory[i, 2]

        smooth_x = smoothed[i, 0]
        smooth_y = smoothed[i, 1]
        smooth_a = smoothed[i, 2]

        diff_x = smooth_x - cum_x_orig
        diff_y = smooth_y - cum_y_orig
        diff_a = smooth_a - cum_a_orig

        cos_a = np.cos(diff_a)
        sin_a = np.sin(diff_a)
        m = np.array([
            [cos_a, -sin_a, diff_x],
            [sin_a, cos_a, diff_y]
        ], dtype=np.float64)

        stabilized = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        out.write(stabilized)

    cap.release()
    out.release()
    print(f"[Stabilizer] Saved stabilized video to {output_path}")
