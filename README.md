<div align="center">

```
     _   _   _     _      _     __     ___               
    / \ | |_| |__ | | ___| |_ __\ \   / (_) _____      __
   / _ \| __| '_ \| |/ _ \ __/ _ \ \ / /| |/ _ \ \ /\ / /
  / ___ \ |_| | | | |  __/ ||  __/\ V / | |  __/\ V  V / 
 /_/   \_\__|_| |_|_|\___|\__\___| \_/  |_|\___| \_/\_/  
                                                          
         ╔═══════════════════════════════════════╗
         ║  AI-Powered Sports Streaming Platform  ║
         ║     See the Game Through Their Eyes    ║
         ╚═══════════════════════════════════════╝
```

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://python.org)
[![Node 20+](https://img.shields.io/badge/Node.js-20+-339933.svg)](https://nodejs.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](docker-compose.yml)
[![CI](https://github.com/pradhap99/athleteview-ai-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/pradhap99/athleteview-ai-engine/actions)

</div>

---

## Overview

AthleteView AI Engine is a real-time sports streaming platform that ingests live video from wearable body cameras (SmartPatch), enhances it through an AI pipeline, overlays biometric data, and distributes to multiple destinations simultaneously.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ATHLETEVIEW AI ENGINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────┐    ┌──────────────┐    ┌─────────────────────────┐  │
│  │ SmartPatch │───▶│  SRT/RTMP    │───▶│    AI PROCESSING GPU    │  │
│  │ Body Cams  │    │  Ingestion   │    │                         │  │
│  │ (5G/Wi-Fi) │    │  (MediaMTX)  │    │  Real-ESRGAN ──┐       │  │
│  └───────────┘    └──────────────┘    │  YOLOv11 ──────┤       │  │
│                                        │  ViTPose++ ────┤       │  │
│  ┌───────────┐    ┌──────────────┐    │  SpeechBrain ──┤       │  │
│  │ MAX86141  │───▶│  Biometrics  │───▶│  Compositor ◀──┘       │  │
│  │ MAX30208  │    │  Processing  │    │       │                 │  │
│  │ BME280    │    │  Service     │    └───────┼─────────────────┘  │
│  │ ICM-42688 │    └──────────────┘            │                    │
│  └───────────┘                                ▼                    │
│                              ┌─────────────────────────────┐       │
│  ┌───────────┐              │      DISTRIBUTION            │       │
│  │  Kafka    │◀─────────────│                              │       │
│  │  Events   │              │  LL-HLS ──▶ Our Platform     │       │
│  └─────┬─────┘              │  WebRTC ──▶ Ultra-Low Lat    │       │
│        │                    │  RTMP ────▶ YouTube/Twitch   │       │
│  ┌─────▼─────┐              │  SRT ─────▶ TV Broadcast     │       │
│  │TimescaleDB│              └─────────────────────────────┘       │
│  │  + Redis  │                                                     │
│  └───────────┘                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Stack | Description |
|---------|------|-------|-------------|
| **Gateway** | 3000 | Node.js + TypeScript | API gateway, WebSocket, auth |
| **AI Engine** | 8001 | Python + FastAPI | Real-ESRGAN, YOLO, ViTPose++ |
| **Streaming** | 8002 | Python + FastAPI | SRT ingestion, HLS/WebRTC dist |
| **Biometrics** | 8003 | Python + FastAPI | PPG processing, vitals analysis |
| **Training** | — | Python + PyTorch | Model fine-tuning pipelines |
| **Firmware** | — | C (RV1106) | SmartPatch embedded firmware |

## AI Models

| Model | Task | Pretrained Weights | License |
|-------|------|--------------------|---------|
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Super-Resolution | [x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) | BSD-3 |
| [YOLOv11](https://github.com/ultralytics/ultralytics) | Object Tracking | [yolo11x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | AGPL-3.0 |
| [ViTPose++](https://huggingface.co/usyd-community/vitpose-plus-large) | Pose Estimation | HuggingFace | Apache-2.0 |
| [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus) | Temporal SR | [MMEditing](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/) | Apache-2.0 |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | 3D Reconstruction | — | Apache-2.0 |
| [SpeechBrain](https://huggingface.co/speechbrain/sepformer-wham16k-enhancement) | Audio Enhancement | HuggingFace | Apache-2.0 |

## Quick Start

```bash
# Clone
git clone https://github.com/pradhap99/athleteview-ai-engine.git
cd athleteview-ai-engine

# Environment
cp .env.example .env

# Start all services
make up

# Or with Docker directly
docker compose up -d

# Download pretrained models
make download-models

# Run tests
make test
```

## API Examples

### Start a stream
```bash
curl -X POST http://localhost:3000/api/v1/streams \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "athlete_id": "kohli_18",
    "camera_position": "chest",
    "sport": "cricket",
    "match_id": "ipl_2026_mi_csk",
    "resolution": "4K",
    "ai_features": ["super_resolution", "tracking", "biometrics"]
  }'
```

### Get live biometrics
```bash
curl http://localhost:3000/api/v1/biometrics/kohli_18/live \
  -H "Authorization: Bearer $TOKEN"
```

### WebSocket — Real-time events
```javascript
const ws = new WebSocket('ws://localhost:3000/ws');
ws.send(JSON.stringify({ 
  type: 'subscribe', 
  channels: ['biometrics:kohli_18', 'highlights:ipl_2026_mi_csk'] 
}));
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

## Project Structure

```
athleteview-ai-engine/
├── gateway/          # API Gateway (Node.js/TS + Express + Socket.io)
├── ai-engine/        # AI Processing (FastAPI + PyTorch + TensorRT)
├── streaming/        # Stream Management (FastAPI + FFmpeg + SRT)
├── biometrics/       # Biometric Processing (FastAPI + SciPy)
├── training/         # Model Training (PyTorch + Ultralytics)
├── firmware/         # SmartPatch Firmware (C for RV1106)
├── shared/           # Protobuf schemas, JSON schemas, constants
├── infra/            # Kubernetes, Terraform, monitoring
└── docs/             # Architecture, API, deployment guides
```

## Infrastructure

- **GPU**: NVIDIA L40S (AI inference) + Jetson AGX Orin (edge)
- **Streaming**: MediaMTX (SRT ingestion) + Ant Media (WebRTC)
- **Message Bus**: Apache Kafka (Confluent)
- **Time-Series**: TimescaleDB (biometric data)
- **Cache**: Redis (session, real-time state)
- **CDN**: CloudFront (HLS delivery)
- **Monitoring**: Prometheus + Grafana

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>AthleteView</b> — See the Game Through Their Eyes<br>
  Built in India, for the world 🏏⚽🏀
</div>
