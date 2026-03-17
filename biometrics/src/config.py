"""
Biometrics Processing Service configuration.

Pydantic settings for sensor calibration, alert thresholds,
ML model paths, and infrastructure (Kafka / Redis) connectivity.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Sensor calibration
# ---------------------------------------------------------------------------


class PPGCalibration(BaseSettings):
    """MAX86141 PPG sensor calibration parameters."""

    model_config = {"env_prefix": "PPG_"}

    # Sampling & filtering
    sample_rate_hz: int = Field(default=100, description="MAX86141 ADC sampling rate")
    bandpass_low_hz: float = Field(default=0.5, description="HR bandpass lower cutoff")
    bandpass_high_hz: float = Field(default=4.0, description="HR bandpass upper cutoff")
    filter_order: int = Field(default=4, description="Butterworth filter order")

    # Peak detection
    peak_min_distance_samples: int = Field(
        default=30,
        description="Minimum samples between R-R peaks (~0.3 s at 100 Hz)",
    )
    peak_min_height_percentile: float = Field(
        default=60.0,
        description="Minimum peak height as percentile of signal amplitude",
    )

    # SpO2 calibration (empirical Beer-Lambert coefficients)
    spo2_coeff_a: float = Field(default=110.0, description="SpO2 = a - b * R")
    spo2_coeff_b: float = Field(default=25.0, description="SpO2 = a - b * R")

    # Adaptive noise cancellation (LMS)
    anc_mu: float = Field(default=0.01, description="LMS adaptive filter step size")
    anc_filter_length: int = Field(default=32, description="LMS filter tap count")


class TemperatureCalibration(BaseSettings):
    """MAX30208 body-temperature sensor calibration."""

    model_config = {"env_prefix": "TEMP_"}

    offset_c: float = Field(default=0.0, description="Calibration offset in Celsius")
    gain: float = Field(default=1.0, description="Calibration gain multiplier")
    smoothing_window: int = Field(default=10, description="Moving-average window size")
    skin_to_core_offset_c: float = Field(
        default=0.75,
        description="Additive adjustment from skin temp to estimated core temp",
    )
    heat_stress_threshold_c: float = Field(
        default=39.5,
        description="Core-temperature threshold for heat-stress alert",
    )
    sample_rate_hz: int = Field(default=1, description="MAX30208 sampling rate")


class EnvironmentCalibration(BaseSettings):
    """BME280 environmental sensor calibration."""

    model_config = {"env_prefix": "ENV_"}

    pressure_sea_level_hpa: float = Field(
        default=1013.25, description="Sea-level reference pressure in hPa"
    )
    humidity_offset_pct: float = Field(
        default=0.0, description="Relative-humidity calibration offset"
    )
    temperature_offset_c: float = Field(
        default=0.0, description="Ambient temperature calibration offset"
    )


class SweatCalibration(BaseSettings):
    """Colorimetric sweat-analysis calibration."""

    model_config = {"env_prefix": "SWEAT_"}

    sodium_low_mmol: float = Field(default=20.0)
    sodium_high_mmol: float = Field(default=80.0)
    potassium_low_mmol: float = Field(default=2.0)
    potassium_high_mmol: float = Field(default=10.0)
    glucose_low_mmol: float = Field(default=0.1)
    glucose_high_mmol: float = Field(default=1.0)
    dehydration_sweat_rate_threshold_l_hr: float = Field(
        default=1.5,
        description="Sweat-rate threshold for dehydration risk (L / hr)",
    )


# ---------------------------------------------------------------------------
# Sensor sample-rate defaults
# ---------------------------------------------------------------------------


class SensorRates(BaseSettings):
    """Default sensor sample rates in Hz."""

    ppg_hz: float = 100.0
    temperature_hz: float = 1.0
    imu_hz: float = 200.0
    environment_hz: float = 0.1

    model_config = {"env_prefix": "SENSOR_RATE_"}


# ---------------------------------------------------------------------------
# Processing window durations (seconds)
# ---------------------------------------------------------------------------


class ProcessingWindows(BaseSettings):
    """Sliding-window lengths for each derived metric."""

    heart_rate_s: float = 10.0
    hrv_s: float = 60.0
    spo2_s: float = 30.0
    respiration_s: float = 30.0
    temperature_s: float = 60.0

    model_config = {"env_prefix": "PROC_WIN_"}


# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------


SPORT_ALERT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "generic": {
        "max_heart_rate_bpm": 200,
        "min_heart_rate_bpm": 35,
        "min_spo2_pct": 90.0,
        "max_core_temp_c": 39.5,
        "max_fatigue_score": 85,
        "max_injury_risk": 80,
    },
    "soccer": {
        "max_heart_rate_bpm": 195,
        "min_heart_rate_bpm": 38,
        "min_spo2_pct": 91.0,
        "max_core_temp_c": 39.5,
        "max_fatigue_score": 85,
        "max_injury_risk": 75,
    },
    "football": {
        "max_heart_rate_bpm": 195,
        "min_heart_rate_bpm": 38,
        "min_spo2_pct": 91.0,
        "max_core_temp_c": 39.5,
        "max_fatigue_score": 85,
        "max_injury_risk": 75,
    },
    "basketball": {
        "max_heart_rate_bpm": 200,
        "min_heart_rate_bpm": 38,
        "min_spo2_pct": 90.0,
        "max_core_temp_c": 39.5,
        "max_fatigue_score": 85,
        "max_injury_risk": 75,
    },
    "cricket": {
        "max_heart_rate_bpm": 185,
        "min_heart_rate_bpm": 40,
        "min_spo2_pct": 92.0,
        "max_core_temp_c": 39.5,
        "max_fatigue_score": 80,
        "max_injury_risk": 70,
    },
    "kabaddi": {
        "max_heart_rate_bpm": 200,
        "min_heart_rate_bpm": 38,
        "min_spo2_pct": 90.0,
        "max_core_temp_c": 39.5,
        "max_fatigue_score": 85,
        "max_injury_risk": 75,
    },
}


class AlertThresholds(BaseSettings):
    """Default alert thresholds (overridden per-athlete at runtime)."""

    max_heart_rate_bpm: int = 200
    min_heart_rate_bpm: int = 35
    min_spo2_pct: float = 90.0
    spo2_warning_pct: float = 93.0
    max_core_temp_c: float = 40.0
    temp_warning_c: float = 39.0
    hrv_rmssd_min_ms: float = 10.0
    max_fatigue_score: int = 85
    fatigue_warning: int = 70
    max_injury_risk: int = 80
    injury_warning: int = 60
    alert_cooldown_s: int = 60
    escalation_delay_s: int = 300

    model_config = {"env_prefix": "ALERT_"}


# ---------------------------------------------------------------------------
# ML model configuration
# ---------------------------------------------------------------------------


class ModelConfig(BaseSettings):
    """Paths and hyper-parameters for ML models."""

    model_config = {"env_prefix": "MODEL_"}

    fatigue_model_path: str = Field(
        default="models/fatigue_lstm.pt",
        description="Path to serialised fatigue LSTM weights",
    )
    fatigue_input_size: int = Field(
        default=5, description="Number of input features for fatigue model"
    )
    fatigue_hidden_size: int = Field(default=64, description="LSTM hidden-layer size")
    fatigue_num_layers: int = Field(default=2, description="Number of LSTM layers")
    fatigue_sequence_length: int = Field(
        default=60, description="Sliding-window length (time-steps)"
    )

    injury_risk_model_path: str = Field(
        default="models/injury_risk_rf.pkl",
        description="Path to injury risk model",
    )


# ---------------------------------------------------------------------------
# Sport-specific performance-index weights
# ---------------------------------------------------------------------------


SPORT_PERFORMANCE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "cricket": {
        "cardio_efficiency": 0.25,
        "recovery_speed": 0.30,
        "sustained_output": 0.25,
        "peak_performance": 0.20,
    },
    "football": {
        "cardio_efficiency": 0.30,
        "recovery_speed": 0.20,
        "sustained_output": 0.30,
        "peak_performance": 0.20,
    },
    "basketball": {
        "cardio_efficiency": 0.25,
        "recovery_speed": 0.25,
        "sustained_output": 0.20,
        "peak_performance": 0.30,
    },
    "kabaddi": {
        "cardio_efficiency": 0.20,
        "recovery_speed": 0.25,
        "sustained_output": 0.25,
        "peak_performance": 0.30,
    },
}

ZONE_HR_PERCENTAGES: Dict[str, tuple] = {
    "rest": (0.0, 0.50),
    "warm_up": (0.50, 0.60),
    "fat_burn": (0.60, 0.70),
    "cardio": (0.70, 0.80),
    "hard": (0.80, 0.90),
    "max_effort": (0.90, 1.00),
}


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


class KafkaConfig(BaseSettings):
    """Kafka connectivity."""

    model_config = {"env_prefix": "KAFKA_"}

    bootstrap_servers: str = Field(default="localhost:9092")
    raw_topic: str = Field(default="biometrics.raw")
    processed_topic: str = Field(default="biometrics.processed")
    alerts_topic: str = Field(default="biometrics.alerts")
    consumer_group: str = Field(default="biometrics-processor")
    enable: bool = Field(default=False, description="Enable Kafka integration")


class RedisConfig(BaseSettings):
    """Redis connectivity."""

    model_config = {"env_prefix": "REDIS_"}

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    vitals_ttl_seconds: int = Field(default=3600)
    enable: bool = Field(default=False, description="Enable Redis caching")

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


# ---------------------------------------------------------------------------
# Aggregate root settings
# ---------------------------------------------------------------------------


class BiometricsConfig(BaseSettings):
    """Root configuration aggregating all sub-configs and infra settings."""

    # Sub-configs
    sensor_rates: SensorRates = Field(default_factory=SensorRates)
    processing_windows: ProcessingWindows = Field(default_factory=ProcessingWindows)
    alert_defaults: AlertThresholds = Field(default_factory=AlertThresholds)
    ppg: PPGCalibration = Field(default_factory=PPGCalibration)
    temperature: TemperatureCalibration = Field(default_factory=TemperatureCalibration)
    environment: EnvironmentCalibration = Field(default_factory=EnvironmentCalibration)
    sweat: SweatCalibration = Field(default_factory=SweatCalibration)
    models: ModelConfig = Field(default_factory=ModelConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    # Service identity
    service_name: str = "biometrics-service"
    service_version: str = "1.0.0"
    service_port: int = 8003
    log_level: str = "INFO"
    debug: bool = False

    # TimescaleDB
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "athleteview_biometrics"
    db_user: str = "biometrics"
    db_password: str = "biometrics"
    db_pool_min: int = 2
    db_pool_max: int = 10

    @property
    def db_dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # Athlete defaults
    default_max_hr: int = 200
    default_resting_hr: int = 60

    model_config = {"env_prefix": "BIO_", "env_nested_delimiter": "__"}


def get_settings() -> BiometricsConfig:
    """Return a settings instance (call per-request or cache in app state)."""
    return BiometricsConfig()
