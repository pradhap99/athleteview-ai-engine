"""
AthleteView — Biometric HUD Overlay Engine
Renders real-time biometric data overlay on video frames.
"""

import cv2
import numpy as np
import random
import math
import time


class BiometricSimulator:
    """Simulates realistic athlete biometric data with smooth fluctuations."""

    def __init__(self, athlete_name="Player 1", sport="Cricket"):
        self.athlete_name = athlete_name
        self.sport = sport
        self.start_time = time.time()

        # Base values (athletic range)
        self.hr_base = random.uniform(140, 165)
        self.spo2_base = random.uniform(96, 98)
        self.temp_base = random.uniform(37.2, 37.8)
        self.hydration_base = random.uniform(75, 90)

        # Current values
        self.hr = self.hr_base
        self.spo2 = self.spo2_base
        self.temp = self.temp_base
        self.hydration = self.hydration_base

    def update(self, frame_idx, fps=30):
        """Update biometric values with realistic fluctuation."""
        t = frame_idx / fps

        # Heart rate: sinusoidal variation + random noise (120-185 BPM)
        self.hr = self.hr_base + 15 * math.sin(t * 0.3) + random.gauss(0, 2)
        self.hr = max(120, min(185, self.hr))

        # SpO2: slow drift with occasional dips (93-99%)
        self.spo2 = self.spo2_base + 1.5 * math.sin(t * 0.1) + random.gauss(0, 0.3)
        self.spo2 = max(93, min(99, self.spo2))

        # Body temp: gradual increase during activity (37.0-38.5°C)
        self.temp = self.temp_base + 0.3 * math.sin(t * 0.05) + 0.001 * t + random.gauss(0, 0.05)
        self.temp = max(37.0, min(38.5, self.temp))

        # Hydration: slow decrease over time (60-95%)
        self.hydration = self.hydration_base - 0.02 * t + 2 * math.sin(t * 0.08) + random.gauss(0, 0.5)
        self.hydration = max(60, min(95, self.hydration))

        return {
            "hr": round(self.hr),
            "spo2": round(self.spo2, 1),
            "temp": round(self.temp, 1),
            "hydration": round(self.hydration, 1),
            "athlete_name": self.athlete_name,
            "sport": self.sport,
        }


def get_alert_color(metric, value):
    """Return color based on alert thresholds. BGR format."""
    if metric == "hr":
        if value > 175:
            return (0, 0, 255)  # Red — danger
        elif value > 165:
            return (0, 165, 255)  # Orange — warning
        return (0, 255, 200)  # Teal — normal
    elif metric == "spo2":
        if value < 94:
            return (0, 0, 255)  # Red
        elif value < 95:
            return (0, 165, 255)  # Orange
        return (0, 255, 200)  # Teal
    elif metric == "temp":
        if value > 38.2:
            return (0, 0, 255)
        elif value > 38.0:
            return (0, 165, 255)
        return (0, 255, 200)
    elif metric == "hydration":
        if value < 65:
            return (0, 0, 255)
        elif value < 70:
            return (0, 165, 255)
        return (0, 255, 200)
    return (0, 255, 200)


def draw_biometric_hud(frame, biometrics, opacity=0.7):
    """
    Draw the biometric HUD overlay on a video frame.

    Args:
        frame: OpenCV BGR frame
        biometrics: dict from BiometricSimulator.update()
        opacity: overlay transparency (0-1)

    Returns:
        Frame with HUD overlay drawn
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel dimensions
    panel_h = int(h * 0.18)
    panel_y = h - panel_h
    panel_margin = 10

    # Draw semi-transparent dark panel at bottom
    cv2.rectangle(overlay, (0, panel_y), (w, h), (20, 20, 30), -1)

    # Top accent line (teal)
    cv2.line(overlay, (0, panel_y), (w, panel_y), (255, 229, 0), 2)  # BGR teal

    # Blend overlay
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    # Athlete name and sport badge
    name_y = panel_y + 25
    cv2.putText(frame, biometrics["athlete_name"].upper(),
                (20, name_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Sport badge
    sport_text = biometrics["sport"].upper()
    sport_size = cv2.getTextSize(sport_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    badge_x = 20 + cv2.getTextSize(biometrics["athlete_name"].upper(),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] + 15
    cv2.rectangle(frame, (badge_x, name_y - 15), (badge_x + sport_size[0] + 10, name_y + 3), (255, 229, 0), -1)
    cv2.putText(frame, sport_text, (badge_x + 5, name_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 30), 1)

    # AthleteView logo text (right side)
    logo_text = "ATHLETEVIEW"
    logo_size = cv2.getTextSize(logo_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.putText(frame, logo_text, (w - logo_size[0] - 20, name_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 229, 0), 2)

    # Metrics row
    metrics = [
        ("HR", f"{biometrics['hr']}", "BPM", "hr", biometrics["hr"]),
        ("SpO2", f"{biometrics['spo2']}", "%", "spo2", biometrics["spo2"]),
        ("TEMP", f"{biometrics['temp']}", "°C", "temp", biometrics["temp"]),
        ("HYDRATION", f"{biometrics['hydration']}", "%", "hydration", biometrics["hydration"]),
    ]

    metric_w = (w - 40) // len(metrics)
    metric_y = name_y + 35

    for i, (label, value, unit, key, raw_val) in enumerate(metrics):
        x = 20 + i * metric_w
        color = get_alert_color(key, raw_val)

        # Label
        cv2.putText(frame, label, (x, metric_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Value (large)
        cv2.putText(frame, value, (x, metric_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Unit
        val_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame, unit, (x + val_size[0] + 5, metric_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

        # Mini bar indicator
        bar_y = metric_y + 40
        bar_w = metric_w - 20
        bar_h = 4

        # Background bar
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

        # Filled portion based on value range
        if key == "hr":
            fill = (raw_val - 100) / 100  # 100-200 range
        elif key == "spo2":
            fill = (raw_val - 90) / 10  # 90-100 range
        elif key == "temp":
            fill = (raw_val - 36) / 3  # 36-39 range
        else:
            fill = raw_val / 100  # 0-100 range

        fill = max(0, min(1, fill))
        cv2.rectangle(frame, (x, bar_y), (x + int(bar_w * fill), bar_y + bar_h), color, -1)

    return frame


def draw_tracking_overlay(frame, detections, frame_idx):
    """
    Draw player tracking boxes and IDs on frame.

    Args:
        frame: OpenCV BGR frame
        detections: list of (x1, y1, x2, y2, track_id, confidence) tuples
        frame_idx: current frame number

    Returns:
        Frame with tracking overlay
    """
    colors = [
        (255, 229, 0),   # Teal
        (0, 165, 255),   # Orange
        (200, 100, 255), # Pink
        (100, 255, 100), # Green
        (255, 100, 100), # Light blue
    ]

    for det in detections:
        x1, y1, x2, y2, track_id, conf = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = colors[track_id % len(colors)]

        # Bounding box with rounded feel (draw 4 corner brackets)
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 2

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

        # ID label background
        label = f"P{track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), (x1 + label_size[0] + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 30), 1)

        # Confidence
        conf_text = f"{conf:.0%}"
        cv2.putText(frame, conf_text, (x1 + label_size[0] + 14, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return frame
