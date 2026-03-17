"""Overlay Renderer — Composites biometric data, tracking boxes, and sport-specific graphics onto video frames."""
import numpy as np
from loguru import logger

class OverlayRenderer:
    """Renders broadcast-quality overlays on video frames."""

    def __init__(self):
        self.font_scale = 0.7
        self.colors = {"cyan": (255, 229, 0), "orange": (53, 107, 255), "green": (136, 255, 0), "red": (0, 0, 255), "white": (255, 255, 255)}

    def render_tracking_boxes(self, frame: np.ndarray, detections: list) -> np.ndarray:
        import cv2
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            color = self.colors["cyan"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{det.get('track_id', '?')} {det.get('speed_kmh', 0):.1f} km/h"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 2)
        return frame

    def render_biometrics(self, frame: np.ndarray, bio: dict, position: str = "right") -> np.ndarray:
        import cv2
        h, w = frame.shape[:2]
        x = w - 280 if position == "right" else 20
        y_start = 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y_start - 10), (x + 260, y_start + 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        hr = bio.get("heart_rate", 0)
        hr_color = self.colors["green"] if hr < 160 else self.colors["orange"] if hr < 180 else self.colors["red"]
        cv2.putText(frame, f"HR: {hr} BPM", (x, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hr_color, 2)
        cv2.putText(frame, f"SpO2: {bio.get('spo2', 0)}%", (x, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["cyan"], 2)
        cv2.putText(frame, f"Temp: {bio.get('body_temp', 0):.1f}C", (x, y_start + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["white"], 2)
        cv2.putText(frame, f"Fatigue: {bio.get('fatigue_index', 0)}%", (x, y_start + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["orange"], 2)
        cv2.putText(frame, f"Speed: {bio.get('sprint_speed', 0):.1f} km/h", (x, y_start + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["green"], 2)
        return frame

    def render_pose_skeleton(self, frame: np.ndarray, poses: list) -> np.ndarray:
        import cv2
        SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for pose in poses:
            kps = pose.get("keypoints", [])
            for i, kp in enumerate(kps):
                if kp.get("score", 0) > 0.3:
                    cx, cy = int(kp["x"]), int(kp["y"])
                    cv2.circle(frame, (cx, cy), 4, self.colors["cyan"], -1)
            for a, b in SKELETON:
                if a < len(kps) and b < len(kps) and kps[a].get("score", 0) > 0.3 and kps[b].get("score", 0) > 0.3:
                    pt1 = (int(kps[a]["x"]), int(kps[a]["y"]))
                    pt2 = (int(kps[b]["x"]), int(kps[b]["y"]))
                    cv2.line(frame, pt1, pt2, self.colors["green"], 2)
        return frame
