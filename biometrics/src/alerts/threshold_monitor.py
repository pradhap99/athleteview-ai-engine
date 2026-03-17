"""Threshold Monitor — Configurable alerts for biometric thresholds."""
from dataclasses import dataclass
from loguru import logger

@dataclass
class AlertRule:
    name: str
    metric: str
    operator: str  # "gt", "lt", "eq"
    threshold: float
    severity: str = "warning"  # "info", "warning", "critical"

DEFAULT_RULES = [
    AlertRule("High HR", "heart_rate", "gt", 190, "critical"),
    AlertRule("Low HR", "heart_rate", "lt", 40, "critical"),
    AlertRule("Low SpO2", "spo2", "lt", 92, "critical"),
    AlertRule("High Temp", "body_temp", "gt", 39.5, "warning"),
    AlertRule("Extreme Fatigue", "fatigue_index", "gt", 85, "warning"),
    AlertRule("Injury Risk", "injury_risk", "eq", "critical", "critical"),
]

class ThresholdMonitor:
    def __init__(self, rules: list[AlertRule] = None):
        self.rules = rules or DEFAULT_RULES

    def check(self, vitals: dict) -> list[dict]:
        alerts = []
        for rule in self.rules:
            value = vitals.get(rule.metric)
            if value is None: continue
            triggered = False
            if rule.operator == "gt" and isinstance(value, (int, float)): triggered = value > rule.threshold
            elif rule.operator == "lt" and isinstance(value, (int, float)): triggered = value < rule.threshold
            elif rule.operator == "eq": triggered = str(value) == str(rule.threshold)
            if triggered:
                alert = {"rule": rule.name, "metric": rule.metric, "value": value, "threshold": rule.threshold, "severity": rule.severity}
                alerts.append(alert)
                logger.warning("Alert: {} — {} = {} (threshold: {})", rule.name, rule.metric, value, rule.threshold)
        return alerts
