"""
Colorimetric sweat analysis processor.

Processes sweat-patch sensor data to estimate electrolyte concentrations
(sodium, potassium), glucose levels, dehydration risk, and electrolyte
balance assessment for real-time athlete monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------


class DehydrationRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class ElectrolyteStatus(str, Enum):
    BALANCED = "balanced"
    LOW = "low"
    HIGH = "high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"


@dataclass(slots=True)
class SweatResult:
    """Aggregated output of the sweat analysis pipeline."""

    sodium_mmol_l: float = 0.0
    potassium_mmol_l: float = 0.0
    glucose_mmol_l: float = 0.0
    sweat_rate_l_hr: float = 0.0
    total_fluid_loss_ml: float = 0.0
    dehydration_risk: str = "low"
    dehydration_score: float = 0.0  # 0-100
    electrolyte_status: str = "balanced"
    sodium_status: str = "balanced"
    potassium_status: str = "balanced"
    replacement_sodium_mg: float = 0.0
    replacement_fluid_ml: float = 0.0


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class SweatAnalyzer:
    """Colorimetric sweat analysis processor.

    Converts raw colorimetric sensor readings (RGB absorbance values) into
    electrolyte concentrations, monitors fluid loss via estimated sweat rate,
    and provides dehydration / electrolyte-balance assessments.

    Parameters
    ----------
    sodium_low_mmol : float
        Normal low sodium concentration threshold.
    sodium_high_mmol : float
        Normal high sodium concentration threshold.
    potassium_low_mmol : float
        Normal low potassium concentration threshold.
    potassium_high_mmol : float
        Normal high potassium concentration threshold.
    glucose_low_mmol : float
        Normal low glucose concentration threshold.
    glucose_high_mmol : float
        Normal high glucose concentration threshold.
    dehydration_sweat_rate_threshold_l_hr : float
        Sweat rate above which dehydration risk increases.
    athlete_weight_kg : float
        Athlete body weight for fluid-loss calculations.
    """

    def __init__(
        self,
        sodium_low_mmol: float = 20.0,
        sodium_high_mmol: float = 80.0,
        potassium_low_mmol: float = 2.0,
        potassium_high_mmol: float = 10.0,
        glucose_low_mmol: float = 0.1,
        glucose_high_mmol: float = 1.0,
        dehydration_sweat_rate_threshold_l_hr: float = 1.5,
        athlete_weight_kg: float = 75.0,
    ) -> None:
        self.sodium_low = sodium_low_mmol
        self.sodium_high = sodium_high_mmol
        self.potassium_low = potassium_low_mmol
        self.potassium_high = potassium_high_mmol
        self.glucose_low = glucose_low_mmol
        self.glucose_high = glucose_high_mmol
        self.sweat_rate_threshold = dehydration_sweat_rate_threshold_l_hr
        self.athlete_weight_kg = athlete_weight_kg

        # Cumulative tracking
        self._cumulative_fluid_loss_ml: float = 0.0
        self._measurement_count: int = 0
        self._sodium_history: List[float] = []
        self._sweat_rate_history: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_raw(
        self,
        rgb_sodium: tuple[float, float, float] | None = None,
        rgb_potassium: tuple[float, float, float] | None = None,
        rgb_glucose: tuple[float, float, float] | None = None,
        sweat_rate_l_hr: float = 0.0,
        elapsed_minutes: float = 1.0,
        sodium_mmol_l: Optional[float] = None,
        potassium_mmol_l: Optional[float] = None,
        glucose_mmol_l: Optional[float] = None,
    ) -> SweatResult:
        """Analyse a sweat sensor reading.

        Accepts either raw RGB absorbance tuples (from colorimetric sensors)
        or pre-computed concentrations.  At least a sweat rate is required
        for dehydration assessment.

        Parameters
        ----------
        rgb_sodium : tuple of 3 floats, optional
            (R, G, B) absorbance for sodium colorimetric assay.
        rgb_potassium : tuple of 3 floats, optional
            (R, G, B) absorbance for potassium colorimetric assay.
        rgb_glucose : tuple of 3 floats, optional
            (R, G, B) absorbance for glucose colorimetric assay.
        sweat_rate_l_hr : float
            Estimated whole-body sweat rate in litres per hour.
        elapsed_minutes : float
            Time elapsed since last measurement for cumulative tracking.
        sodium_mmol_l : float, optional
            Pre-computed sodium concentration (skips colorimetric conversion).
        potassium_mmol_l : float, optional
            Pre-computed potassium concentration.
        glucose_mmol_l : float, optional
            Pre-computed glucose concentration.

        Returns
        -------
        SweatResult
        """
        # Determine concentrations
        na = sodium_mmol_l if sodium_mmol_l is not None else (
            self._rgb_to_sodium(rgb_sodium) if rgb_sodium else 0.0
        )
        k = potassium_mmol_l if potassium_mmol_l is not None else (
            self._rgb_to_potassium(rgb_potassium) if rgb_potassium else 0.0
        )
        glu = glucose_mmol_l if glucose_mmol_l is not None else (
            self._rgb_to_glucose(rgb_glucose) if rgb_glucose else 0.0
        )

        # Cumulative fluid loss
        fluid_loss_ml = sweat_rate_l_hr * (elapsed_minutes / 60.0) * 1000.0
        self._cumulative_fluid_loss_ml += fluid_loss_ml
        self._sweat_rate_history.append(sweat_rate_l_hr)
        self._sodium_history.append(na)
        self._measurement_count += 1

        # Dehydration assessment
        dehydration_score = self._compute_dehydration_score(
            sweat_rate_l_hr, self._cumulative_fluid_loss_ml
        )
        dehydration_risk = self._classify_dehydration(dehydration_score)

        # Electrolyte assessment
        na_status = self._classify_electrolyte(na, self.sodium_low, self.sodium_high)
        k_status = self._classify_electrolyte(k, self.potassium_low, self.potassium_high)
        overall_status = self._overall_electrolyte_status(na_status, k_status)

        # Replacement recommendations
        replacement_na_mg = self._recommend_sodium_replacement(na, sweat_rate_l_hr)
        replacement_fluid_ml = self._recommend_fluid_replacement(
            sweat_rate_l_hr, self._cumulative_fluid_loss_ml
        )

        return SweatResult(
            sodium_mmol_l=round(na, 1),
            potassium_mmol_l=round(k, 1),
            glucose_mmol_l=round(glu, 2),
            sweat_rate_l_hr=round(sweat_rate_l_hr, 2),
            total_fluid_loss_ml=round(self._cumulative_fluid_loss_ml, 0),
            dehydration_risk=dehydration_risk,
            dehydration_score=round(dehydration_score, 1),
            electrolyte_status=overall_status,
            sodium_status=na_status,
            potassium_status=k_status,
            replacement_sodium_mg=round(replacement_na_mg, 0),
            replacement_fluid_ml=round(replacement_fluid_ml, 0),
        )

    def reset(self) -> None:
        """Reset cumulative tracking state."""
        self._cumulative_fluid_loss_ml = 0.0
        self._measurement_count = 0
        self._sodium_history.clear()
        self._sweat_rate_history.clear()

    # ------------------------------------------------------------------
    # Colorimetric conversion models
    # ------------------------------------------------------------------

    def _rgb_to_sodium(self, rgb: tuple[float, float, float]) -> float:
        """Convert RGB absorbance from a sodium-specific colorimetric assay
        to sodium concentration in mmol/L.

        Uses a polynomial calibration model fitted to reference solutions.
        The green channel provides the primary signal for sodium indicators
        (chromoionophore-based).
        """
        r, g, b = rgb
        # Primary: ratio of green to (red + blue)
        denominator = r + b + 1e-6
        ratio = g / denominator

        # Quadratic calibration: Na = a * ratio^2 + b * ratio + c
        # Fitted to calibration solutions (20, 40, 60, 80 mmol/L)
        na = 120.0 * ratio ** 2 + 15.0 * ratio + 5.0
        return float(np.clip(na, 0.0, 150.0))

    def _rgb_to_potassium(self, rgb: tuple[float, float, float]) -> float:
        """Convert RGB absorbance to potassium concentration (mmol/L).

        Potassium indicators (valinomycin-based) primarily affect the
        red channel.
        """
        r, g, b = rgb
        denominator = g + b + 1e-6
        ratio = r / denominator

        k = 15.0 * ratio ** 2 + 3.0 * ratio + 0.5
        return float(np.clip(k, 0.0, 25.0))

    def _rgb_to_glucose(self, rgb: tuple[float, float, float]) -> float:
        """Convert RGB absorbance to glucose concentration (mmol/L).

        Glucose oxidase / peroxidase colorimetric assay produces a colour
        shift primarily in the blue channel.
        """
        r, g, b = rgb
        denominator = r + g + 1e-6
        ratio = b / denominator

        glu = 2.0 * ratio ** 2 + 0.3 * ratio + 0.05
        return float(np.clip(glu, 0.0, 5.0))

    # ------------------------------------------------------------------
    # Dehydration assessment
    # ------------------------------------------------------------------

    def _compute_dehydration_score(
        self, current_sweat_rate: float, total_loss_ml: float
    ) -> float:
        """Compute a composite dehydration risk score (0-100).

        Factors:
        1. Current sweat rate relative to threshold  (40% weight)
        2. Cumulative fluid loss as % of body weight (40% weight)
        3. Sweat rate trend (accelerating = worse)    (20% weight)
        """
        # Factor 1: Sweat rate ratio
        rate_ratio = current_sweat_rate / self.sweat_rate_threshold if self.sweat_rate_threshold > 0 else 0.0
        rate_score = min(rate_ratio * 50.0, 50.0)

        # Factor 2: Cumulative loss as % body weight
        body_water_ml = self.athlete_weight_kg * 1000.0 * 0.60  # ~60% body water
        loss_pct = (total_loss_ml / body_water_ml) * 100.0 if body_water_ml > 0 else 0.0
        # 2% body weight loss is significant; map to 0-50 score
        loss_score = min(loss_pct / 2.0 * 50.0, 50.0)

        # Factor 3: Trend
        trend_score = 0.0
        if len(self._sweat_rate_history) >= 3:
            recent = self._sweat_rate_history[-3:]
            if recent[-1] > recent[0]:
                trend_score = min((recent[-1] - recent[0]) / self.sweat_rate_threshold * 25.0, 25.0)

        score = 0.4 * rate_score + 0.4 * loss_score + 0.2 * trend_score
        return float(np.clip(score, 0.0, 100.0))

    def _classify_dehydration(self, score: float) -> str:
        """Map dehydration score to risk level."""
        if score >= 75:
            return DehydrationRisk.SEVERE.value
        if score >= 50:
            return DehydrationRisk.HIGH.value
        if score >= 25:
            return DehydrationRisk.MODERATE.value
        return DehydrationRisk.LOW.value

    # ------------------------------------------------------------------
    # Electrolyte assessment
    # ------------------------------------------------------------------

    def _classify_electrolyte(
        self, value: float, low: float, high: float
    ) -> str:
        """Classify an electrolyte concentration relative to normal range."""
        critical_low = low * 0.5
        critical_high = high * 1.5

        if value <= critical_low:
            return ElectrolyteStatus.CRITICAL_LOW.value
        if value <= low:
            return ElectrolyteStatus.LOW.value
        if value >= critical_high:
            return ElectrolyteStatus.CRITICAL_HIGH.value
        if value >= high:
            return ElectrolyteStatus.HIGH.value
        return ElectrolyteStatus.BALANCED.value

    def _overall_electrolyte_status(self, na_status: str, k_status: str) -> str:
        """Determine overall electrolyte balance from individual statuses."""
        critical_states = {
            ElectrolyteStatus.CRITICAL_LOW.value,
            ElectrolyteStatus.CRITICAL_HIGH.value,
        }
        abnormal_states = {
            ElectrolyteStatus.LOW.value,
            ElectrolyteStatus.HIGH.value,
        }

        if na_status in critical_states or k_status in critical_states:
            return "critical"
        if na_status in abnormal_states or k_status in abnormal_states:
            return "imbalanced"
        return "balanced"

    # ------------------------------------------------------------------
    # Replacement recommendations
    # ------------------------------------------------------------------

    def _recommend_sodium_replacement(
        self, sodium_mmol_l: float, sweat_rate_l_hr: float
    ) -> float:
        """Estimate sodium replacement needed (mg).

        Sodium lost = sweat_rate * [Na+] * molecular_weight
        1 mmol Na = 23 mg
        """
        na_loss_mmol_hr = sodium_mmol_l * sweat_rate_l_hr
        na_loss_mg_hr = na_loss_mmol_hr * 23.0
        return max(na_loss_mg_hr, 0.0)

    def _recommend_fluid_replacement(
        self, sweat_rate_l_hr: float, total_loss_ml: float
    ) -> float:
        """Recommend fluid intake (ml) to offset losses.

        Target: replace ~80% of sweat losses (stomach can absorb ~800-1000 ml/hr).
        """
        target_rate_ml_hr = min(sweat_rate_l_hr * 1000.0 * 0.80, 1000.0)
        deficit = total_loss_ml * 0.80  # Target 80% replacement
        return max(target_rate_ml_hr, deficit * 0.1)  # At least 10% of deficit per cycle
