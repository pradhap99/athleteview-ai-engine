#!/usr/bin/env python3
"""
AthleteView — AI Demo Pipeline
================================
Full video processing pipeline:
1. Input video → 
2. Super-resolution (OpenCV DNN / bicubic upscale) →
3. Player detection + tracking (YOLOv8-nano / HOG fallback) →
4. Biometric HUD overlay →
5. Video stabilization →
6. Before/After comparison output

Runs entirely on CPU. No GPU required.
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse

from overlay_engine import BiometricSimulator, draw_biometric_hud, draw_tracking_overlay
from tracker import PlayerTracker
from stabilizer import VideoStabilizer


def upscale_frame(frame, scale=2, method="bicubic"):
    """
    Super-resolution upscale of a frame.
    Uses bicubic interpolation (fast, CPU-friendly).
    For production, swap with Real-ESRGAN or EDSR.
    """
    h, w = frame.shape[:2]
    new_w, new_h = w * scale, h * scale

    if method == "bicubic":
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif method == "lanczos":
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def create_comparison_frame(original, processed, target_w=1920, target_h=540):
    """Create a side-by-side before/after comparison frame."""
    half_w = target_w // 2
    
    left = cv2.resize(original, (half_w, target_h))
    right = cv2.resize(processed, (half_w, target_h))

    # Labels
    cv2.putText(left, "ORIGINAL", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    cv2.putText(right, "ATHLETEVIEW AI", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 229, 0), 2)

    # Divider line
    comparison = np.hstack([left, right])
    cv2.line(comparison, (half_w, 0), (half_w, target_h), (255, 229, 0), 2)

    return comparison


def process_video(input_path, output_dir="output", scale=1, use_yolo=True,
                  athletes=None, stabilize=True):
    """
    Main pipeline: process a video through the full AthleteView AI stack.

    Args:
        input_path: path to input video
        output_dir: output directory
        scale: super-resolution scale factor (1 = no upscale, 2 = 2x)
        use_yolo: whether to use YOLO for detection
        athletes: list of athlete name/sport dicts
        stabilize: whether to apply stabilization
    """
    os.makedirs(output_dir, exist_ok=True)

    # Default athletes
    if athletes is None:
        athletes = [
            {"name": "Virat Kohli", "sport": "Cricket"},
            {"name": "MS Dhoni", "sport": "Cricket"},
            {"name": "Rohit Sharma", "sport": "Cricket"},
        ]

    print("=" * 60)
    print("  ATHLETEVIEW AI DEMO PIPELINE")
    print("  See the Game Through Their Eyes")
    print("=" * 60)

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = in_w * scale
    out_h = in_h * scale
    comp_w = 1920
    comp_h = out_h if scale == 1 else out_h // 2

    print(f"\n[Pipeline] Input: {input_path}")
    print(f"[Pipeline] Resolution: {in_w}x{in_h} → {out_w}x{out_h}")
    print(f"[Pipeline] Frames: {n_frames} @ {fps:.1f} FPS")
    print(f"[Pipeline] Scale: {scale}x | YOLO: {use_yolo} | Stabilize: {stabilize}")

    # Initialize components
    print("\n[1/5] Initializing tracker...")
    tracker = PlayerTracker(use_yolo=use_yolo)

    print("[2/5] Initializing biometric simulators...")
    simulators = [BiometricSimulator(a["name"], a["sport"]) for a in athletes]
    active_sim_idx = 0

    print("[3/5] Initializing stabilizer...")
    stab = VideoStabilizer(smoothing_radius=10)

    # Output writers
    enhanced_path = os.path.join(output_dir, "enhanced.mp4")
    comparison_path = os.path.join(output_dir, "comparison.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_enhanced = cv2.VideoWriter(enhanced_path, fourcc, fps, (out_w, out_h))
    writer_comparison = cv2.VideoWriter(comparison_path, fourcc, fps, (comp_w, comp_h))

    print(f"[4/5] Processing {n_frames} frames...")
    start_time = time.time()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()

        # Step 1: Super-resolution upscale
        if scale > 1:
            frame = upscale_frame(frame, scale=scale)

        # Step 2: Player detection & tracking
        tracked = tracker.track(frame)

        # Cycle through athlete simulators based on tracked players
        active_sim = simulators[frame_idx % len(simulators)]

        # Step 3: Update biometrics
        biometrics = active_sim.update(frame_idx, fps)

        # Step 4: Draw tracking overlay
        frame = draw_tracking_overlay(frame, tracked, frame_idx)

        # Step 5: Draw biometric HUD
        frame = draw_biometric_hud(frame, biometrics)

        # Step 6: Simple per-frame stabilization
        if stabilize:
            dx, dy, da = stab.estimate_motion(frame)
            # Apply light smoothing directly
            smooth_factor = 0.3
            cos_a = np.cos(-da * smooth_factor)
            sin_a = np.sin(-da * smooth_factor)
            M = np.array([
                [cos_a, -sin_a, -dx * smooth_factor],
                [sin_a, cos_a, -dy * smooth_factor]
            ], dtype=np.float64)
            frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                                    borderMode=cv2.BORDER_REPLICATE)

        # Write enhanced frame
        writer_enhanced.write(frame)

        # Create and write comparison
        comparison = create_comparison_frame(original, frame, comp_w, comp_h)
        writer_comparison.write(comparison)

        frame_idx += 1

        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            progress = (frame_idx / n_frames) * 100 if n_frames > 0 else 0
            print(f"  Frame {frame_idx}/{n_frames} ({progress:.0f}%) | {fps_actual:.1f} fps")

    cap.release()
    writer_enhanced.release()
    writer_comparison.release()

    elapsed = time.time() - start_time

    print(f"\n[5/5] Pipeline complete!")
    print(f"  Processed: {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"  Enhanced:  {enhanced_path} ({os.path.getsize(enhanced_path) / 1024 / 1024:.1f} MB)")
    print(f"  Comparison: {comparison_path} ({os.path.getsize(comparison_path) / 1024 / 1024:.1f} MB)")
    print("=" * 60)

    return enhanced_path, comparison_path


def main():
    parser = argparse.ArgumentParser(description="AthleteView AI Demo Pipeline")
    parser.add_argument("input", nargs="?", default="sample_input.mp4",
                        help="Input video path (default: sample_input.mp4)")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--scale", "-s", type=int, default=1, choices=[1, 2],
                        help="Super-resolution scale (1=none, 2=2x)")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Disable YOLO, use HOG fallback")
    parser.add_argument("--no-stabilize", action="store_true",
                        help="Disable video stabilization")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input video not found: {args.input}")
        print("  Run 'python generate_sample.py' first to create a test video.")
        sys.exit(1)

    process_video(
        input_path=args.input,
        output_dir=args.output,
        scale=args.scale,
        use_yolo=not args.no_yolo,
        stabilize=not args.no_stabilize,
    )


if __name__ == "__main__":
    main()
