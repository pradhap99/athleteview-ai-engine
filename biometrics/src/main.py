"""AthleteView Biometrics Service — Real-time processing of SmartPatch sensor data."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from loguru import logger
from prometheus_client import make_asgi_app

app = FastAPI(title="AthleteView Biometrics", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/metrics", make_asgi_app())

class SensorReading(BaseModel):
    athlete_id: str
    timestamp: float
    ppg_raw: Optional[list[float]] = None
    temperature_raw: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    imu_accel: Optional[list[float]] = None
    imu_gyro: Optional[list[float]] = None

class ProcessedVitals(BaseModel):
    athlete_id: str
    heart_rate: int
    spo2: float
    hrv_rmssd: float
    body_temp: float
    fatigue_index: float
    sprint_speed: float
    injury_risk: str
    composure_index: float

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "biometrics"}

@app.post("/api/v1/process", response_model=ProcessedVitals)
async def process_reading(reading: SensorReading):
    from .sensors.ppg_processor import PPGProcessor
    from .analysis.vitals_analyzer import VitalsAnalyzer
    ppg = PPGProcessor()
    analyzer = VitalsAnalyzer()
    hr, spo2, hrv = ppg.process(reading.ppg_raw or [])
    vitals = analyzer.compute(hr=hr, spo2=spo2, hrv=hrv, temp=reading.temperature_raw or 37.0, imu=reading.imu_accel or [0, 0, 9.8])
    return ProcessedVitals(athlete_id=reading.athlete_id, **vitals)
