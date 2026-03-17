"""Device Manager — Tracks connected SmartPatch cameras and their capabilities."""
from dataclasses import dataclass, field
from loguru import logger

@dataclass
class SmartPatchDevice:
    device_id: str
    athlete_id: str
    camera_position: str
    firmware_version: str
    battery_level: int = 100
    signal_strength: int = -50
    sensors_active: list = field(default_factory=lambda: ["camera", "ppg", "imu", "temp", "mic"])
    connected_at: float = 0.0

class DeviceManager:
    def __init__(self):
        self.devices: dict[str, SmartPatchDevice] = {}

    def register(self, device: SmartPatchDevice):
        self.devices[device.device_id] = device
        logger.info("Device registered: {} (athlete: {}, position: {})", device.device_id, device.athlete_id, device.camera_position)

    def unregister(self, device_id: str):
        if device_id in self.devices:
            del self.devices[device_id]

    def get_athlete_devices(self, athlete_id: str) -> list[SmartPatchDevice]:
        return [d for d in self.devices.values() if d.athlete_id == athlete_id]

    def get_all(self) -> list[SmartPatchDevice]:
        return list(self.devices.values())
