"""
AthleteView — Sample Video Generator
Creates a synthetic cricket-field test video with moving players.
"""

import cv2
import numpy as np
import random
import math
import os


def generate_sample_video(output_path="sample_input.mp4", duration=10, fps=30, width=1280, height=720):
    """
    Generate a synthetic cricket-field video with moving colored player rectangles.

    Args:
        output_path: output video file path
        duration: video duration in seconds
        fps: frames per second
        width: frame width
        height: frame height
    """
    print(f"[Generator] Creating {duration}s sample video at {width}x{height}@{fps}fps...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    n_frames = duration * fps

    # Define players with initial positions, colors, and movement patterns
    players = [
        {"name": "Batsman", "color": (255, 255, 255), "x": 500, "y": 400, "w": 40, "h": 80,
         "vx": 2.5, "vy": 0.5, "phase": 0},
        {"name": "Bowler", "color": (0, 200, 255), "x": 800, "y": 300, "w": 35, "h": 75,
         "vx": -3, "vy": 1, "phase": 1.5},
        {"name": "Fielder1", "color": (100, 255, 100), "x": 200, "y": 200, "w": 35, "h": 70,
         "vx": 1, "vy": 2, "phase": 0.8},
        {"name": "Fielder2", "color": (100, 255, 100), "x": 1000, "y": 500, "w": 35, "h": 70,
         "vx": -1.5, "vy": -1, "phase": 2.2},
        {"name": "Keeper", "color": (255, 200, 0), "x": 600, "y": 450, "w": 38, "h": 75,
         "vx": 0.8, "vy": -0.3, "phase": 3.0},
    ]

    # Camera shake parameters (simulating body-worn camera)
    shake_amp_x = 3
    shake_amp_y = 2
    shake_freq = 4.0

    for frame_idx in range(n_frames):
        t = frame_idx / fps

        # Green cricket field background with gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            green_val = int(80 + 40 * (y / height))
            frame[y, :] = (30, green_val, 20)

        # Cricket pitch (lighter strip in center)
        pitch_x1, pitch_x2 = width // 2 - 30, width // 2 + 30
        pitch_y1, pitch_y2 = height // 4, 3 * height // 4
        cv2.rectangle(frame, (pitch_x1, pitch_y1), (pitch_x2, pitch_y2), (60, 160, 100), -1)

        # Boundary circle
        cv2.ellipse(frame, (width // 2, height // 2), (width // 2 - 50, height // 2 - 30),
                     0, 0, 360, (40, 100, 40), 2)

        # Draw crease lines
        cv2.line(frame, (pitch_x1 - 20, pitch_y1 + 40), (pitch_x2 + 20, pitch_y1 + 40), (200, 200, 200), 2)
        cv2.line(frame, (pitch_x1 - 20, pitch_y2 - 40), (pitch_x2 + 20, pitch_y2 - 40), (200, 200, 200), 2)

        # Move and draw players
        for p in players:
            # Sinusoidal movement
            p["x"] += p["vx"] * math.cos(t * 0.5 + p["phase"])
            p["y"] += p["vy"] * math.sin(t * 0.7 + p["phase"])

            # Keep in bounds
            p["x"] = max(50, min(width - 50, p["x"]))
            p["y"] = max(50, min(height - 100, p["y"]))

            px, py = int(p["x"]), int(p["y"])
            pw, ph = p["w"], p["h"]

            # Player body (rectangle)
            cv2.rectangle(frame, (px - pw // 2, py - ph // 2),
                          (px + pw // 2, py + ph // 2), p["color"], -1)

            # Head (circle)
            cv2.circle(frame, (px, py - ph // 2 - 10), 12, p["color"], -1)

            # Shadow
            shadow_pts = np.array([
                [px - pw // 3, py + ph // 2],
                [px + pw // 3, py + ph // 2],
                [px + pw // 2 + 10, py + ph // 2 + 8],
                [px - pw // 2 - 10, py + ph // 2 + 8]
            ])
            cv2.fillPoly(frame, [shadow_pts], (20, 50, 15))

        # Apply camera shake
        shake_x = int(shake_amp_x * math.sin(t * shake_freq * 2 * math.pi + 0.3 * math.sin(t * 2.7)))
        shake_y = int(shake_amp_y * math.cos(t * shake_freq * 1.7 * math.pi + 0.5 * math.cos(t * 3.1)))

        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

        # Frame counter (small, top-left)
        cv2.putText(frame, f"Frame {frame_idx}/{n_frames}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        out.write(frame)

    out.release()
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Generator] Saved {output_path} ({file_size:.1f} MB, {n_frames} frames)")
    return output_path


if __name__ == "__main__":
    generate_sample_video()
