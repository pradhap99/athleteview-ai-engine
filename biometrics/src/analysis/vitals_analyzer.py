"""Vitals Analyzer — Computes composite metrics from raw biometric data."""
import numpy as np

class VitalsAnalyzer:
    """Computes fatigue index, injury risk, composure index from sensor fusion."""

    HR_ZONES = {"rest": (40, 100), "easy": (100, 140), "tempo": (140, 160), "threshold": (160, 180), "max": (180, 220)}

    def compute(self, hr: int, spo2: float, hrv: float, temp: float, imu: list[float]) -> dict:
        accel_mag = np.sqrt(sum(x**2 for x in imu))
        fatigue = self._fatigue_index(hr, hrv, spo2, temp)
        risk = self._injury_risk(fatigue, hr, temp, accel_mag)
        composure = self._composure_index(hr, hrv)
        speed = self._estimate_speed(accel_mag)

        return {
            "heart_rate": hr, "spo2": spo2, "hrv_rmssd": hrv,
            "body_temp": round(temp, 1), "fatigue_index": round(fatigue, 1),
            "sprint_speed": round(speed, 1), "injury_risk": risk,
            "composure_index": round(composure, 1),
        }

    def _fatigue_index(self, hr: int, hrv: float, spo2: float, temp: float) -> float:
        hr_score = min(100, max(0, (hr - 60) / 1.4))
        hrv_score = max(0, 100 - hrv)  # Lower HRV = more fatigue
        spo2_score = max(0, (100 - spo2) * 10)
        temp_score = max(0, (temp - 37.5) * 30) if temp > 37.5 else 0
        return hr_score * 0.35 + hrv_score * 0.30 + spo2_score * 0.20 + temp_score * 0.15

    def _injury_risk(self, fatigue: float, hr: int, temp: float, accel: float) -> str:
        score = fatigue * 0.4 + (1 if hr > 185 else 0) * 20 + (1 if temp > 38.5 else 0) * 20 + min(accel / 5, 1) * 20
        if score > 70: return "critical"
        if score > 50: return "high"
        if score > 30: return "medium"
        return "low"

    def _composure_index(self, hr: int, hrv: float) -> float:
        hr_component = max(0, 100 - (hr - 70) * 0.5)
        hrv_component = min(100, hrv * 1.5)
        return hr_component * 0.4 + hrv_component * 0.6

    def _estimate_speed(self, accel_magnitude: float) -> float:
        if accel_magnitude < 10: return 0.0
        return min(35.0, (accel_magnitude - 9.8) * 3.5)
