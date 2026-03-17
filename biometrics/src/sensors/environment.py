"""
Environmental sensor processor for the BME280 (humidity, pressure, temperature).

Computes WBGT estimation, altitude from barometric pressure, heat index,
and environmental comfort scoring for athlete safety monitoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EnvResult:
    """Output of the environmental processing pipeline."""

    ambient_temp_c: float = 0.0
    relative_humidity_pct: float = 0.0
    pressure_hpa: float = 0.0
    altitude_m: float = 0.0
    heat_index_c: float = 0.0
    wbgt_c: float = 0.0
    dew_point_c: float = 0.0
    comfort_level: str = "comfortable"  # comfortable / caution / warning / danger


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class EnvironmentProcessor:
    """Process BME280 environmental sensor readings.

    Provides WBGT estimation, altitude calculation, heat index, and
    environmental comfort classification for sports safety monitoring.

    Parameters
    ----------
    pressure_sea_level_hpa : float
        Sea-level reference pressure for altitude calculation (default 1013.25).
    humidity_offset_pct : float
        Calibration offset for humidity sensor.
    temperature_offset_c : float
        Calibration offset for the BME280 on-board thermistor.
    """

    def __init__(
        self,
        pressure_sea_level_hpa: float = 1013.25,
        humidity_offset_pct: float = 0.0,
        temperature_offset_c: float = 0.0,
    ) -> None:
        self.pressure_sea_level_hpa = pressure_sea_level_hpa
        self.humidity_offset_pct = humidity_offset_pct
        self.temperature_offset_c = temperature_offset_c

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_raw(
        self,
        temperature_c: float,
        humidity_pct: float,
        pressure_hpa: float,
    ) -> EnvResult:
        """Process a single set of BME280 readings.

        Parameters
        ----------
        temperature_c : float
            Ambient temperature in Celsius.
        humidity_pct : float
            Relative humidity as a percentage (0-100).
        pressure_hpa : float
            Barometric pressure in hectopascals.

        Returns
        -------
        EnvResult
        """
        # Apply calibration offsets
        temp = temperature_c + self.temperature_offset_c
        rh = np.clip(humidity_pct + self.humidity_offset_pct, 0.0, 100.0)
        pressure = pressure_hpa

        altitude = self.estimate_altitude(pressure)
        heat_index = self.compute_heat_index(temp, rh)
        wbgt = self.estimate_wbgt(temp, rh)
        dew_point = self.compute_dew_point(temp, rh)
        comfort = self._classify_comfort(heat_index, rh)

        return EnvResult(
            ambient_temp_c=round(float(temp), 2),
            relative_humidity_pct=round(float(rh), 1),
            pressure_hpa=round(float(pressure), 2),
            altitude_m=round(altitude, 1),
            heat_index_c=round(heat_index, 1),
            wbgt_c=round(wbgt, 1),
            dew_point_c=round(dew_point, 1),
            comfort_level=comfort,
        )

    # ------------------------------------------------------------------
    # Altitude estimation (hypsometric formula)
    # ------------------------------------------------------------------

    def estimate_altitude(self, pressure_hpa: float) -> float:
        """Estimate altitude from barometric pressure using the international
        barometric formula.

        altitude = 44330 * (1 - (P / P_0)^(1/5.255))

        Parameters
        ----------
        pressure_hpa : float
            Measured barometric pressure in hPa.

        Returns
        -------
        float
            Estimated altitude in metres above sea level.
        """
        if pressure_hpa <= 0 or self.pressure_sea_level_hpa <= 0:
            return 0.0

        ratio = pressure_hpa / self.pressure_sea_level_hpa
        altitude = 44330.0 * (1.0 - math.pow(ratio, 1.0 / 5.255))
        return float(altitude)

    # ------------------------------------------------------------------
    # Heat index (Rothfusz regression)
    # ------------------------------------------------------------------

    def compute_heat_index(
        self, temp_c: float, humidity_pct: float
    ) -> float:
        """Calculate heat index using the Rothfusz regression equation
        (NWS / NOAA method).

        The equation uses Fahrenheit internally and converts back to Celsius.

        Parameters
        ----------
        temp_c : float
            Ambient temperature in Celsius.
        humidity_pct : float
            Relative humidity (0-100 %).

        Returns
        -------
        float
            Heat index in Celsius.
        """
        t_f = temp_c * 9.0 / 5.0 + 32.0
        rh = float(humidity_pct)

        # Simple formula for lower temperatures
        if t_f < 80.0:
            hi_f = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)
            return (hi_f - 32.0) * 5.0 / 9.0

        # Full Rothfusz regression
        hi_f = (
            -42.379
            + 2.04901523 * t_f
            + 10.14333127 * rh
            - 0.22475541 * t_f * rh
            - 6.83783e-3 * t_f ** 2
            - 5.481717e-2 * rh ** 2
            + 1.22874e-3 * t_f ** 2 * rh
            + 8.5282e-4 * t_f * rh ** 2
            - 1.99e-6 * t_f ** 2 * rh ** 2
        )

        # Adjustment for low humidity
        if rh < 13.0 and 80.0 <= t_f <= 112.0:
            adj = -((13.0 - rh) / 4.0) * math.sqrt(
                (17.0 - abs(t_f - 95.0)) / 17.0
            )
            hi_f += adj
        # Adjustment for high humidity
        elif rh > 85.0 and 80.0 <= t_f <= 87.0:
            adj = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
            hi_f += adj

        return (hi_f - 32.0) * 5.0 / 9.0

    # ------------------------------------------------------------------
    # WBGT estimation (Liljegren simplified model)
    # ------------------------------------------------------------------

    def estimate_wbgt(
        self,
        temp_c: float,
        humidity_pct: float,
        solar_radiation_w_m2: float = 0.0,
        wind_speed_m_s: float = 1.0,
    ) -> float:
        """Estimate Wet Bulb Globe Temperature (WBGT) for outdoor heat stress.

        Uses the Liljegren simplified outdoor model. When no globe or solar
        data are available, falls back to the indoor approximation:

            WBGT_indoor ~ 0.7 * T_wb + 0.3 * T_db

        For outdoor with solar radiation:

            WBGT_outdoor ~ 0.7 * T_wb + 0.2 * T_g + 0.1 * T_db

        Parameters
        ----------
        temp_c : float
            Dry-bulb temperature (Celsius).
        humidity_pct : float
            Relative humidity (%).
        solar_radiation_w_m2 : float
            Incident solar radiation (W/m^2). 0 for indoor.
        wind_speed_m_s : float
            Wind speed (m/s).

        Returns
        -------
        float
            Estimated WBGT in Celsius.
        """
        t_wb = self._wet_bulb_temperature(temp_c, humidity_pct)

        if solar_radiation_w_m2 <= 0:
            # Indoor WBGT approximation
            return 0.7 * t_wb + 0.3 * temp_c

        # Estimate globe temperature (simplified)
        t_globe = temp_c + 0.01 * solar_radiation_w_m2 - 0.5 * wind_speed_m_s
        return 0.7 * t_wb + 0.2 * t_globe + 0.1 * temp_c

    # ------------------------------------------------------------------
    # Dew point (Magnus formula)
    # ------------------------------------------------------------------

    def compute_dew_point(
        self, temp_c: float, humidity_pct: float
    ) -> float:
        """Calculate dew point temperature using the Magnus-Tetens approximation.

        Parameters
        ----------
        temp_c : float
            Temperature in Celsius.
        humidity_pct : float
            Relative humidity (%).

        Returns
        -------
        float
            Dew point in Celsius.
        """
        if humidity_pct <= 0:
            return temp_c - 30.0  # Very rough lower bound

        a = 17.27
        b = 237.7
        rh_frac = humidity_pct / 100.0

        alpha = (a * temp_c) / (b + temp_c) + math.log(rh_frac)
        dew_point = (b * alpha) / (a - alpha)
        return float(dew_point)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wet_bulb_temperature(
        self, temp_c: float, humidity_pct: float
    ) -> float:
        """Estimate wet-bulb temperature using the Stull (2011) empirical
        regression, valid for RH > 5% and -20 < T < 50 C.

        T_wb = T * atan(0.151977 * sqrt(RH + 8.313659))
             + atan(T + RH) - atan(RH - 1.676331)
             + 0.00391838 * RH^1.5 * atan(0.023101 * RH) - 4.686035
        """
        t = temp_c
        rh = max(humidity_pct, 5.0)

        t_wb = (
            t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
            + math.atan(t + rh)
            - math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
            - 4.686035
        )
        return float(t_wb)

    def _classify_comfort(
        self, heat_index_c: float, humidity_pct: float
    ) -> str:
        """Classify environmental comfort level for athlete safety.

        Levels
        ------
        comfortable : HI < 27 C
        caution     : 27 <= HI < 32 C
        warning     : 32 <= HI < 39 C
        danger      : HI >= 39 C
        """
        if heat_index_c >= 39.0:
            return "danger"
        if heat_index_c >= 32.0:
            return "warning"
        if heat_index_c >= 27.0:
            return "caution"
        return "comfortable"
