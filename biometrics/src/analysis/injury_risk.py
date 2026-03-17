"""
Injury risk assessment model.

Evaluates multi-factor injury risk from biometric vitals, biomechanical data,
and session history. Produces per-body-region risk scores, contributing factor
analysis, and actionable recommendations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Sport-specific injury patterns
# ---------------------------------------------------------------------------

SPORT_INJURY_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "soccer": {
        "high_risk_regions": ["knee", "ankle", "hamstring", "groin"],
        "specific_risks": ["acl_tear", "ankle_sprain", "hamstring_strain", "groin_pull"],
        "fatigue_sensitive_regions": ["knee", "hamstring"],
    },
    "football": {
        "high_risk_regions": ["knee", "ankle", "shoulder", "head"],
        "specific_risks": ["acl_tear", "ankle_sprain", "shoulder_dislocation", "concussion"],
        "fatigue_sensitive_regions": ["knee", "ankle"],
    },
    "basketball": {
        "high_risk_regions": ["knee", "ankle", "achilles", "wrist"],
        "specific_risks": ["acl_tear", "ankle_sprain", "achilles_rupture", "wrist_fracture"],
        "fatigue_sensitive_regions": ["ankle", "achilles"],
    },
    "cricket": {
        "high_risk_regions": ["shoulder", "lower_back", "knee", "elbow"],
        "specific_risks": ["shoulder_impingement", "lower_back_stress", "knee_strain", "elbow_tendinopathy"],
        "fatigue_sensitive_regions": ["shoulder", "lower_back"],
    },
    "baseball": {
        "high_risk_regions": ["shoulder", "elbow", "lower_back", "knee"],
        "specific_risks": ["rotator_cuff_tear", "ucl_tear", "lower_back_strain", "knee_strain"],
        "fatigue_sensitive_regions": ["shoulder", "elbow"],
    },
    "kabaddi": {
        "high_risk_regions": ["knee", "ankle", "shoulder", "wrist"],
        "specific_risks": ["knee_ligament", "ankle_sprain", "shoulder_dislocation", "wrist_fracture"],
        "fatigue_sensitive_regions": ["knee", "ankle"],
    },
}

DEFAULT_INJURY_PATTERN: Dict[str, List[str]] = {
    "high_risk_regions": ["knee", "ankle", "lower_back"],
    "specific_risks": ["general_strain", "overuse_injury"],
    "fatigue_sensitive_regions": ["knee", "lower_back"],
}

# Body region baseline risk weights (0-1)
BODY_REGION_WEIGHTS: Dict[str, float] = {
    "head": 0.05,
    "neck": 0.05,
    "shoulder": 0.10,
    "elbow": 0.06,
    "wrist": 0.04,
    "lower_back": 0.12,
    "hip": 0.08,
    "groin": 0.07,
    "hamstring": 0.10,
    "knee": 0.15,
    "achilles": 0.06,
    "ankle": 0.08,
    "foot": 0.04,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RiskFactor:
    """A single contributing factor to injury risk."""

    name: str
    score: float  # 0-100
    weight: float  # relative weight in overall score
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "weight": round(self.weight, 2),
            "description": self.description,
        }


@dataclass
class InjuryRiskResult:
    """Complete injury risk assessment result."""

    overall_risk: float = 0.0  # 0-100
    risk_factors: List[RiskFactor] = field(default_factory=list)
    body_regions: Dict[str, float] = field(default_factory=dict)  # region -> risk 0-100
    recommendations: List[str] = field(default_factory=list)
    sport: str = "generic"
    timestamp_ms: int = 0


# ---------------------------------------------------------------------------
# Risk assessor
# ---------------------------------------------------------------------------


class InjuryRiskAssessor:
    """Multi-factor injury risk assessment engine.

    Evaluates injury risk by combining physiological stress indicators,
    biomechanical asymmetry data, fatigue accumulation, recovery status,
    and sport-specific injury patterns.

    Parameters
    ----------
    sport : str
        Default sport for injury-pattern lookup.
    max_hr : float
        Maximum heart rate for normalisation.
    resting_hrv_rmssd : float
        Baseline HRV RMSSD for recovery assessment.
    """

    def __init__(
        self,
        sport: str = "generic",
        max_hr: float = 200.0,
        resting_hrv_rmssd: float = 50.0,
    ) -> None:
        self.sport = sport
        self.max_hr = max_hr
        self.resting_hrv_rmssd = resting_hrv_rmssd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        vitals: Dict[str, Any],
        biomechanics: Dict[str, Any],
        history: List[Dict[str, Any]],
        sport: Optional[str] = None,
    ) -> InjuryRiskResult:
        """Perform a comprehensive injury risk assessment.

        Parameters
        ----------
        vitals : dict
            Current vitals snapshot with keys: heart_rate_bpm, spo2_pct,
            hrv_rmssd_ms, core_temp_c, activity_level, fatigue_score,
            dehydration_risk.
        biomechanics : dict
            Biomechanical data with optional keys: asymmetry_pct,
            ground_contact_time_ms, stride_length_cm, impact_force_g,
            left_right_balance, pose_data.
        history : list of dict
            Recent vitals history for trend analysis.

        Returns
        -------
        InjuryRiskResult
        """
        active_sport = sport or self.sport
        pattern = SPORT_INJURY_PATTERNS.get(active_sport, DEFAULT_INJURY_PATTERN)

        # Compute individual risk factors
        factors = self._compute_risk_factors(vitals, biomechanics, history, pattern)

        # Compute weighted overall score
        total_weight = sum(f.weight for f in factors) or 1.0
        overall_risk = sum(f.score * f.weight for f in factors) / total_weight
        overall_risk = float(np.clip(overall_risk, 0.0, 100.0))

        # Compute per-body-region risk
        body_regions = self._compute_body_region_risk(
            vitals, biomechanics, factors, pattern, overall_risk
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_risk, factors, body_regions, active_sport
        )

        return InjuryRiskResult(
            overall_risk=round(overall_risk, 1),
            risk_factors=factors,
            body_regions={k: round(v, 1) for k, v in body_regions.items()},
            recommendations=recommendations,
            sport=active_sport,
            timestamp_ms=int(time.time() * 1000),
        )

    # ------------------------------------------------------------------
    # Risk factor computation
    # ------------------------------------------------------------------

    def _compute_risk_factors(
        self,
        vitals: Dict[str, Any],
        biomechanics: Dict[str, Any],
        history: List[Dict[str, Any]],
        pattern: Dict[str, Any],
    ) -> List[RiskFactor]:
        """Compute individual risk factor scores."""
        factors: List[RiskFactor] = []

        # 1. Asymmetric loading (from biomechanics)
        factors.append(self._assess_asymmetric_loading(biomechanics))

        # 2. Fatigue accumulation
        factors.append(self._assess_fatigue_accumulation(vitals, history))

        # 3. Insufficient recovery
        factors.append(self._assess_insufficient_recovery(vitals, history))

        # 4. Environmental stress
        factors.append(self._assess_environmental_stress(vitals))

        # 5. Rapid HR elevation
        factors.append(self._assess_rapid_hr_elevation(vitals, history))

        # 6. Dehydration
        factors.append(self._assess_dehydration(vitals))

        return factors

    def _assess_asymmetric_loading(self, biomechanics: Dict[str, Any]) -> RiskFactor:
        """Assess risk from asymmetric force distribution.

        Asymmetry > 15% significantly increases injury risk, particularly
        for lower-extremity injuries (ACL, ankle).
        """
        asymmetry = biomechanics.get("asymmetry_pct", 0.0)
        left_right_balance = biomechanics.get("left_right_balance", 50.0)

        # Balance deviation from 50/50
        balance_deviation = abs(left_right_balance - 50.0) * 2.0

        # Impact force contribution
        impact = biomechanics.get("impact_force_g", 0.0)
        impact_score = min(impact / 5.0 * 30.0, 30.0)  # 5g is high-risk threshold

        # Combined asymmetry score
        asym_score = min(asymmetry / 15.0 * 40.0, 40.0)
        balance_score = min(balance_deviation / 20.0 * 30.0, 30.0)

        score = asym_score + balance_score + impact_score
        score = float(np.clip(score, 0.0, 100.0))

        return RiskFactor(
            name="asymmetric_loading",
            score=score,
            weight=0.20,
            description=f"Force distribution asymmetry: {asymmetry:.1f}%",
        )

    def _assess_fatigue_accumulation(
        self, vitals: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> RiskFactor:
        """Assess risk from accumulated fatigue.

        Fatigue score > 70 is a significant risk multiplier. Sustained high
        fatigue (>60 for >10 min) compounds injury risk exponentially.
        """
        fatigue = vitals.get("fatigue_score", 0.0)

        # Direct fatigue score contribution
        fatigue_direct = min(fatigue / 85.0 * 60.0, 60.0)

        # Duration at high fatigue from history
        duration_factor = 0.0
        if history:
            high_activity_count = sum(
                1 for h in history
                if h.get("activity_level", 0) > 0.6
            )
            total = max(len(history), 1)
            duration_factor = min(high_activity_count / total * 40.0, 40.0)

        score = fatigue_direct + duration_factor
        score = float(np.clip(score, 0.0, 100.0))

        return RiskFactor(
            name="fatigue_accumulation",
            score=score,
            weight=0.25,
            description=f"Current fatigue score: {fatigue:.1f}",
        )

    def _assess_insufficient_recovery(
        self, vitals: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> RiskFactor:
        """Assess risk from insufficient recovery.

        Low HRV relative to baseline indicates incomplete autonomic recovery.
        This is a strong predictor of overuse injuries.
        """
        hrv_rmssd = vitals.get("hrv_rmssd_ms", 0.0)

        # HRV depression relative to baseline
        if self.resting_hrv_rmssd > 0 and hrv_rmssd > 0:
            hrv_ratio = hrv_rmssd / self.resting_hrv_rmssd
            recovery_score = max(0.0, (1.0 - hrv_ratio) * 80.0)
        elif hrv_rmssd <= 0:
            recovery_score = 50.0  # Unable to assess, moderate default
        else:
            recovery_score = 0.0

        # Check for declining HRV trend
        trend_penalty = 0.0
        if len(history) >= 5:
            recent_hrv = [
                h.get("hrv_rmssd_ms", 0.0) for h in history[-10:]
                if h.get("hrv_rmssd_ms", 0.0) > 0
            ]
            if len(recent_hrv) >= 3:
                early = np.mean(recent_hrv[: len(recent_hrv) // 2])
                late = np.mean(recent_hrv[len(recent_hrv) // 2:])
                if early > 0 and late < early * 0.85:
                    trend_penalty = 20.0

        score = float(np.clip(recovery_score + trend_penalty, 0.0, 100.0))

        return RiskFactor(
            name="insufficient_recovery",
            score=score,
            weight=0.20,
            description=f"HRV RMSSD: {hrv_rmssd:.1f} ms (baseline: {self.resting_hrv_rmssd:.1f} ms)",
        )

    def _assess_environmental_stress(self, vitals: Dict[str, Any]) -> RiskFactor:
        """Assess risk from environmental conditions.

        Heat stress increases injury risk through:
        - Impaired neuromuscular function (core temp > 39 C)
        - Dehydration-related muscle cramping
        - Reduced reaction time
        """
        core_temp = vitals.get("core_temp_c", 37.0)

        # Core temperature risk
        if core_temp >= 40.0:
            temp_score = 100.0
        elif core_temp >= 39.5:
            temp_score = 80.0
        elif core_temp >= 39.0:
            temp_score = 50.0
        elif core_temp >= 38.5:
            temp_score = 25.0
        else:
            temp_score = max(0.0, (core_temp - 37.0) / 1.5 * 15.0)

        score = float(np.clip(temp_score, 0.0, 100.0))

        return RiskFactor(
            name="environmental_stress",
            score=score,
            weight=0.10,
            description=f"Core temperature: {core_temp:.1f} C",
        )

    def _assess_rapid_hr_elevation(
        self, vitals: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> RiskFactor:
        """Assess risk from rapid heart rate elevation.

        Sudden HR spikes without corresponding increase in workload may
        indicate cardiac drift or autonomic stress.
        """
        hr = vitals.get("heart_rate_bpm", 0.0)
        score = 0.0

        if hr > 0 and self.max_hr > 0:
            # Current HR as fraction of max
            hr_pct = hr / self.max_hr

            # Sustained high HR is risky
            if hr_pct >= 0.95:
                score = 80.0
            elif hr_pct >= 0.90:
                score = 50.0
            elif hr_pct >= 0.85:
                score = 25.0

            # Check for rapid elevation from history
            if len(history) >= 3:
                recent_hr = [
                    h.get("heart_rate_bpm", 0.0) for h in history[-5:]
                    if h.get("heart_rate_bpm", 0.0) > 0
                ]
                if len(recent_hr) >= 2:
                    hr_change = recent_hr[-1] - recent_hr[0]
                    if hr_change > 40:  # >40 bpm jump
                        score = max(score, 60.0)
                    elif hr_change > 25:
                        score = max(score, 35.0)

        score = float(np.clip(score, 0.0, 100.0))

        return RiskFactor(
            name="rapid_hr_elevation",
            score=score,
            weight=0.10,
            description=f"Heart rate: {hr:.0f} bpm ({hr / self.max_hr * 100:.0f}% max)" if hr > 0 else "No HR data",
        )

    def _assess_dehydration(self, vitals: Dict[str, Any]) -> RiskFactor:
        """Assess risk from dehydration.

        Dehydration impairs thermoregulation and increases muscle injury risk.
        """
        dehydration_risk = vitals.get("dehydration_risk", "low")

        risk_map = {
            "low": 5.0,
            "moderate": 30.0,
            "high": 60.0,
            "severe": 90.0,
        }
        score = risk_map.get(dehydration_risk, 5.0)

        # SpO2 contribution (low SpO2 compounds dehydration risk)
        spo2 = vitals.get("spo2_pct", 98.0)
        if spo2 > 0 and spo2 < 93.0:
            score = min(score + 15.0, 100.0)

        score = float(np.clip(score, 0.0, 100.0))

        return RiskFactor(
            name="dehydration",
            score=score,
            weight=0.15,
            description=f"Dehydration risk level: {dehydration_risk}",
        )

    # ------------------------------------------------------------------
    # Body region risk
    # ------------------------------------------------------------------

    def _compute_body_region_risk(
        self,
        vitals: Dict[str, Any],
        biomechanics: Dict[str, Any],
        factors: List[RiskFactor],
        pattern: Dict[str, Any],
        overall_risk: float,
    ) -> Dict[str, float]:
        """Compute per-body-region risk scores.

        Distributes the overall risk across body regions based on sport-specific
        patterns and available biomechanical data.
        """
        regions: Dict[str, float] = {}
        high_risk_regions = pattern.get("high_risk_regions", [])
        fatigue_regions = pattern.get("fatigue_sensitive_regions", [])

        # Get fatigue factor score
        fatigue_factor_score = 0.0
        for f in factors:
            if f.name == "fatigue_accumulation":
                fatigue_factor_score = f.score
                break

        for region, base_weight in BODY_REGION_WEIGHTS.items():
            region_risk = overall_risk * base_weight * 2.0  # Scale to use full range

            # Boost high-risk regions for this sport
            if region in high_risk_regions:
                region_risk *= 1.5

            # Apply fatigue multiplier to sensitive regions
            if region in fatigue_regions and fatigue_factor_score > 50:
                fatigue_mult = 1.0 + (fatigue_factor_score - 50) / 100.0
                region_risk *= fatigue_mult

            # Apply pose/biomechanics data if available
            pose_data = biomechanics.get("pose_data", {})
            if region in pose_data:
                region_specific = pose_data[region]
                region_risk = max(region_risk, region_specific.get("risk", 0.0))

            regions[region] = float(np.clip(region_risk, 0.0, 100.0))

        return regions

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        overall_risk: float,
        factors: List[RiskFactor],
        body_regions: Dict[str, float],
        sport: str,
    ) -> List[str]:
        """Generate actionable recommendations based on risk assessment."""
        recommendations: List[str] = []

        # Overall risk-level recommendations
        if overall_risk >= 80:
            recommendations.append(
                "CRITICAL: Immediate activity reduction recommended. "
                "Consider substitution or mandatory rest period."
            )
        elif overall_risk >= 60:
            recommendations.append(
                "HIGH RISK: Reduce training intensity. "
                "Active monitoring required with 5-minute reassessment intervals."
            )
        elif overall_risk >= 40:
            recommendations.append(
                "MODERATE RISK: Monitor closely. Consider reducing high-intensity intervals."
            )

        # Factor-specific recommendations
        for f in factors:
            if f.name == "fatigue_accumulation" and f.score > 50:
                recommendations.append(
                    "Fatigue is elevated. Implement active recovery protocols and "
                    "ensure adequate hydration."
                )
            elif f.name == "insufficient_recovery" and f.score > 40:
                recommendations.append(
                    "HRV indicates incomplete recovery. Consider extending warm-down "
                    "periods and monitoring sleep quality."
                )
            elif f.name == "environmental_stress" and f.score > 40:
                recommendations.append(
                    "Environmental heat stress detected. Increase fluid intake, "
                    "implement cooling strategies, and schedule rest in shaded areas."
                )
            elif f.name == "asymmetric_loading" and f.score > 40:
                recommendations.append(
                    "Significant movement asymmetry detected. Review technique and "
                    "consider biomechanical assessment."
                )
            elif f.name == "dehydration" and f.score > 40:
                recommendations.append(
                    "Dehydration risk is elevated. Increase fluid and electrolyte intake immediately."
                )
            elif f.name == "rapid_hr_elevation" and f.score > 50:
                recommendations.append(
                    "Rapid heart rate elevation detected. Allow for gradual intensity "
                    "increases and monitor cardiac response."
                )

        # Body-region specific recommendations
        high_risk_regions = [
            r for r, risk in body_regions.items() if risk > 60
        ]
        if high_risk_regions:
            region_str = ", ".join(high_risk_regions)
            recommendations.append(
                f"Elevated risk in: {region_str}. Apply targeted warm-up and "
                f"protective equipment for these areas."
            )

        # Sport-specific recommendations
        pattern = SPORT_INJURY_PATTERNS.get(sport, DEFAULT_INJURY_PATTERN)
        specific_risks = pattern.get("specific_risks", [])
        if overall_risk > 50 and specific_risks:
            recommendations.append(
                f"Sport-specific concerns ({sport}): Monitor for early signs of "
                f"{', '.join(specific_risks[:2])}."
            )

        if not recommendations:
            recommendations.append("Risk levels are within normal range. Continue current activity.")

        return recommendations
