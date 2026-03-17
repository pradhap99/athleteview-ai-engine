# AthleteView Architecture

## System Overview

AthleteView is a distributed microservices platform for real-time sports video streaming with AI enhancement.

### Data Flow

1. **SmartPatch** → SRT/5G → **Ingestion Server** (MediaMTX)
2. **Ingestion** → Kafka → **AI Engine** (GPU processing)
3. **AI Engine** → Enhanced frames → **Streaming Service**
4. **Streaming** → LL-HLS / WebRTC / RTMP / SRT → **Viewers & Broadcasters**
5. **Biometrics** → TimescaleDB → **Real-time overlay** + **API**

### Service Communication

| From | To | Protocol | Purpose |
|------|-----|----------|---------|
| SmartPatch | Streaming | SRT (UDP) | Video ingestion |
| Streaming | AI Engine | gRPC / Kafka | Frame processing |
| Biometrics | Gateway | Kafka | Real-time vitals |
| Gateway | Clients | REST / WebSocket | API + live data |
| Streaming | CDN | HLS / SRT | Video distribution |

### GPU Resource Management

- Each AI Engine instance handles ~4 concurrent streams on NVIDIA L40S
- TensorRT FP16 for Real-ESRGAN: ~15ms/frame at 1080p→4K
- YOLOv11 + BoT-SORT: ~8ms/frame for 20 tracked objects
- ViTPose++: ~12ms/frame for 5 persons

### Scaling Strategy

| Phase | Cameras | GPU Servers | CDN | Monthly Cost |
|-------|---------|------------|-----|-------------|
| Pilot (10) | 10 | 1× L40S | CloudFront | $2,400 |
| MVP (50) | 50 | 4× L40S | CloudFront | $12,000 |
| Scale (200) | 200 | 12× L40S | Akamai | $45,000 |
| Full (1000+) | 1000 | 50× L40S | Multi-CDN | $180,000 |
