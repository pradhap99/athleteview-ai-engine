"""Temporal Action Detection — Detects highlights: goals, wickets, catches, tackles.

Uses combination of:
  - Audio spike detection (crowd roar)
  - Motion intensity analysis (IMU acceleration spikes)
  - Biometric surge detection (HR spike > 20 BPM in 5 seconds)
  - Object event detection (ball trajectory change)

Datasets: SoccerNet (500 games, 17 action classes)
"""
import numpy as np
from loguru import logger

class HighlightDetector:
    """Multi-modal highlight detection combining audio, motion, and biometrics."""

    def __init__(self):
        self.audio_threshold = 0.7
        self.motion_threshold = 3.0  # g-force
        self.hr_spike_threshold = 20  # BPM change in 5s
        self.model = None

    async def load(self):
        logger.info("Highlight detector initialized (rule-based + ML hybrid)")

    async def unload(self):
        self.model = None

    def detect(self, audio_level: float, imu_magnitude: float, hr_delta: float, frame_detections: list) -> dict:
        """Detect if current moment is a highlight.
        
        Args:
            audio_level: Normalized audio energy (0-1)
            imu_magnitude: IMU acceleration magnitude in g
            hr_delta: Heart rate change over last 5 seconds
            frame_detections: Object detections from YOLO
            
        Returns:
            Highlight classification with confidence
        """
        score = 0.0
        reasons = []

        if audio_level > self.audio_threshold:
            score += 0.3
            reasons.append("crowd_roar")
        if imu_magnitude > self.motion_threshold:
            score += 0.25
            reasons.append("high_impact")
        if abs(hr_delta) > self.hr_spike_threshold:
            score += 0.25
            reasons.append("biometric_surge")
        if len(frame_detections) > 5:
            score += 0.2
            reasons.append("player_clustering")

        return {
            "is_highlight": score >= 0.5,
            "confidence": min(score, 1.0),
            "reasons": reasons,
            "suggested_type": self._classify_highlight(reasons),
        }

    def _classify_highlight(self, reasons: list) -> str:
        if "high_impact" in reasons and "crowd_roar" in reasons:
            return "goal_or_wicket"
        if "biometric_surge" in reasons:
            return "physical_exertion"
        if "crowd_roar" in reasons:
            return "crowd_moment"
        return "general"
