"""Temperature processor for MAX30208 high-accuracy clinical-grade sensor.

Implements Kalman-filtered noise reduction, core body temperature estimation
from skin temperature via a dual-heat-flux model, and hyper/hypothermia
detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np


class TempTrend(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"


class AlertLevel(str, Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(slots=True)
class TempResult:
    """Output of the temperature processing pipeline."""

    skin_temp_c: float = 0.0
    estimated_core_temp_c: float = 0.0
    trend: str = "stable"
    alert_level: str = "normal"


class TemperatureProcessor:
    """Process raw MAX30208 skin-temperature readings.

    Parameters
    ----------
    sample_rate : float
        Sensor sample rate in Hz (default 1 Hz for MAX30208).
    """

    # Kalman filter state
    _kf_x: float  # state estimate
    _kf_p: float  # estimate covariance
    _kf_q: float  # process noise
    _kf_r: float  # measurement noise

    def __init__(self, sample_rate: float = 1.0) -> None:
        self.sample_rate = sample_rate

        # Kalman filter initialisation
        self._kf_x = 33.0  # initial skin temp guess
        self._kf_p = 1.0
        self._kf_q = 0.001  # slow-changing process
        self._kf_r = 0.05   # sensor noise (MAX30208 ± 0.1 C)

        # Sliding window for trend analysis (last 120 samples = 2 min at 1 Hz)
        self._history: list[float] = []
        self._history_max = 120

        # Thresholds
        self._hyper_threshold_c = 39.5   # core temp
        self._hypo_threshold_c = 35.0    # core temp
        self._caution_high_c = 38.5
        self._caution_low_c = 35.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_raw(
        self,
        raw_data: np.ndarray,
        ambient_temp: float = 25.0,
    ) -> TempResult:
        """Process a window (or single sample) of raw temperature readings.

        Parameters
        ----------
        raw_data : ndarray
            Array of skin temperature readings in Celsius.
        ambient_temp : float
            Ambient temperature for core-temp estimation.

        Returns
        -------
        TempResult
        """
        raw_data = np.asarray(raw_data, dtype=np.float64).ravel()

        # Apply Kalman filter to each sample
        filtered_values: list[float] = []
        for sample in raw_data:
            fv = self._kalman_update(sample)
            filtered_values.append(fv)

        skin_temp = filtered_values[-1] if filtered_values else 0.0

        # Core temp estimation
        core_temp = self.compute_core_temp(skin_temp, ambient_temp)

        # Update history
        self._history.extend(filtered_values)
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]

        # Trend
        trend = self._compute_trend()

        # Alert level based on core temp
        alert_level = self._assess_alert(core_temp)

        return TempResult(
            skin_temp_c=round(skin_temp, 2),
            estimated_core_temp_c=round(core_temp, 2),
            trend=trend,
            alert_level=alert_level,
        )

    def compute_core_temp(
        self, skin_temp: float, ambient_temp: float
    ) -> float:
        """Estimate core body temperature from skin temperature.

        Uses the dual-heat-flux model:
            T_core = T_skin + k * (T_skin - T_ambient)

        where k is a thermal resistance ratio derived from Kitamura (2010).
        Typical k ~ 0.7 – 0.9 for clothed torso sensor placement.

        Parameters
        ----------
        skin_temp : float
            Filtered skin temperature (C).
        ambient_temp : float
            Ambient/environmental temperature (C).

        Returns
        -------
        float
            Estimated core temperature (C).
        """
        k = 0.79  # Empirical coefficient for torso placement
        core = skin_temp + k * (skin_temp - ambient_temp)
        # Clamp to physiologically plausible range
        return float(np.clip(core, 30.0, 44.0))

    def detect_hyperthermia(self, temp_history: np.ndarray) -> bool:
        """Detect hyperthermia risk from a history of core temperatures.

        Returns True if the core temperature exceeds 39.5 C or has been
        rising above 38.5 C for > 5 consecutive samples.
        """
        arr = np.asarray(temp_history, dtype=np.float64).ravel()
        if len(arr) == 0:
            return False

        if arr[-1] >= self._hyper_threshold_c:
            return True

        # Sustained elevation check
        above_caution = arr >= self._caution_high_c
        if len(above_caution) >= 5 and np.all(above_caution[-5:]):
            return True

        return False

    def detect_hypothermia(self, temp_history: np.ndarray) -> bool:
        """Detect hypothermia risk from a history of core temperatures."""
        arr = np.asarray(temp_history, dtype=np.float64).ravel()
        if len(arr) == 0:
            return False

        if arr[-1] <= self._hypo_threshold_c:
            return True

        below_caution = arr <= self._caution_low_c
        if len(below_caution) >= 5 and np.all(below_caution[-5:]):
            return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _kalman_update(self, measurement: float) -> float:
        """Single-step scalar Kalman filter update."""
        # Predict
        x_pred = self._kf_x
        p_pred = self._kf_p + self._kf_q

        # Update
        k_gain = p_pred / (p_pred + self._kf_r)
        self._kf_x = x_pred + k_gain * (measurement - x_pred)
        self._kf_p = (1.0 - k_gain) * p_pred

        return self._kf_x

    def _compute_trend(self) -> str:
        """Determine temperature trend from sliding window via linear regression."""
        if len(self._history) < 10:
            return TempTrend.STABLE.value

        window = np.array(self._history[-30:])
        x = np.arange(len(window))
        slope = np.polyfit(x, window, 1)[0]  # deg/sample

        # Convert to degrees per minute
        slope_per_min = slope * self.sample_rate * 60.0

        if slope_per_min > 0.05:
            return TempTrend.RISING.value
        elif slope_per_min < -0.05:
            return TempTrend.FALLING.value
        return TempTrend.STABLE.value

    def _assess_alert(self, core_temp: float) -> str:
        """Classify alert level based on estimated core temperature."""
        if core_temp >= self._hyper_threshold_c or core_temp <= self._hypo_threshold_c:
            return AlertLevel.CRITICAL.value
        if core_temp >= self._caution_high_c or core_temp <= self._caution_low_c:
            return AlertLevel.WARNING.value
        if core_temp >= 38.0 or core_temp <= 36.0:
            return AlertLevel.CAUTION.value
        return AlertLevel.NORMAL.value
