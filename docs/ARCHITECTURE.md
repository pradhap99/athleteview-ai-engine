# AthleteView Architecture

## Model Stack v2.0 (April 2026)

| Task | Model | Why | License |
|------|-------|-----|---------|
| Super-Resolution (recorded) | **VPEG** | AIM 2025 winner, 5x cheaper than Real-ESRGAN, better perceptual quality | Open (AIM 2025) |
| Super-Resolution (live) | **FlashVSR** | 17fps near-4K on A100, first real-time diffusion VSR | Open |
| Player Detection | **YOLO26** | NMS-free, 43% faster CPU, 57.5 mAP (Jan 2026) | AGPL-3.0 |
| Player Tracking | **BoT-SORT** | Still best multi-object tracker for sports | MIT |
| Pose (live) | **YOLO26-Pose** | Bundled with detection, zero overhead (68.8-70.4 mAP) | AGPL-3.0 |
| Pose (analysis) | **ViTPose++** | Highest accuracy for biomechanics (80.9 AP) | Apache-2.0 |
| Depth (players) | **Depth Pro** | 1.8x better boundary F1 than DAV2, metric depth | Apple Open |
| Depth (field) | **DAV2 Metric-Outdoor** | Best outdoor metric depth (0.045 AbsRel KITTI) | MIT |
| 3D Reconstruction | **gsplat (3DGS)** | 100+ fps rendering, validated by 2026 Olympics MUCAR | MIT |
| Intelligence | **Gemma 4 26B MoE** | 256K context, native video, Apache 2.0, 4B active params | Apache-2.0 |
| Audio | **SpeechBrain** | Commentary isolation + enhancement | Apache-2.0 |

## System Overview

AthleteView is a distributed microservices platform for real-time sports video streaming with AI enhancement.

### Data Flow

1. **SmartPatch** → SRT/5G → **Ingestion Server** (MediaMTX)
2. **Ingestion** → Kafka → **AI Engine** (GPU processing)
3. **AI Engine** → Enhanced frames → **Streaming Service**
4. **Streaming** → LL-HLS / WebRTC / RTMP / SRT → **Viewers & Broadcasters**
5. **Biometrics** → TimescaleDB → **Real-time overlay** + **API**
6. **Gemma 4** → Commentary, insights, coach queries → **Intelligence API**

### Service Communication

| From | To | Protocol | Purpose |
|------|-----|----------|---------|
| SmartPatch | Streaming | SRT (UDP) | Video ingestion |
| Streaming | AI Engine | gRPC / Kafka | Frame processing |
| Biometrics | Gateway | Kafka | Real-time vitals |
| Gateway | Clients | REST / WebSocket | API + live data |
| Streaming | CDN | HLS / SRT | Video distribution |
| AI Engine | Gemma 4 | gRPC | Commentary + insights |

### GPU Resource Management (Updated for v2.0)

- Each AI Engine instance handles ~6 concurrent streams on NVIDIA L40S (was ~4 with Real-ESRGAN)
- TensorRT FP16 for VPEG: ~6ms/frame at 1080p→4K (was ~15ms with Real-ESRGAN)
- FlashVSR live: ~58ms/frame at 768×1408 on A100 (17fps real-time)
- YOLO26 + BoT-SORT: ~5ms/frame for 20 tracked objects (was ~8ms with YOLOv11)
- YOLO26-Pose (live): 0ms additional — bundled with detection
- ViTPose++ (analysis): ~12ms/frame for 5 persons (post-match only)
- Depth Pro: ~100ms/frame (not real-time, used for 3D reconstruction pass)
- gsplat rendering: 100+ fps (runs on viewer GPU, not server)

### Per-Match GPU Cost (3 hours, 6 camera feeds)

| Component | Cost (v2.0) | Cost (v1.0) | Savings |
|-----------|-------------|-------------|---------|
| VPEG/FlashVSR (6 feeds) | ~$15 | ~$27 | -44% |
| YOLO26 + tracking | $0 (CPU) | $0 (CPU) | — |
| YOLO26-Pose (live) | $0 (bundled) | ~$27 | -100% |
| ViTPose++ (post-match) | ~$5 | — | New |
| 3D Gaussian Splatting | ~$18 | ~$18 | — |
| Gemma 4 commentary | ~$2 | ~$2 | — |
| **Total per match** | **~$40-60** | **~$75-100** | **-40%** |

### Scaling Strategy

| Phase | Cameras | GPU Servers | CDN | Monthly Cost |
|-------|---------|------------|-----|-------------|
| Pilot (10) | 10 | 1× L40S | CloudFront | $1,500 |
| MVP (50) | 50 | 3× L40S | CloudFront | $8,000 |
| Scale (200) | 200 | 8× L40S | Akamai | $30,000 |
| Full (1000+) | 1000 | 35× L40S | Multi-CDN | $120,000 |

*Costs reduced ~30-40% from v1.0 due to VPEG efficiency and YOLO26-Pose bundling.*
