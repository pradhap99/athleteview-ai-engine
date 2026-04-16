# CLAUDE.md — AthleteView AI Engine

## Project Overview

AthleteView AI Engine is a real-time sports streaming platform that ingests live video from wearable body cameras (SmartPatch), enhances it through a GPU-accelerated AI pipeline, overlays biometric data, and distributes to multiple destinations (LL-HLS, WebRTC, RTMP, SRT). Built as a distributed microservices architecture.

## Repository Structure

```
athleteview-ai-engine/
├── gateway/             # API Gateway — Node.js + TypeScript + Express + Socket.io (port 3000)
├── ai-engine/           # AI Processing — Python + FastAPI + PyTorch + TensorRT (port 8001)
├── streaming/           # Stream Mgmt — Python + FastAPI + FFmpeg + SRT (port 8002)
├── biometrics/          # Biometric Processing — Python + FastAPI + SciPy (port 8003)
├── training/            # Model Training — PyTorch + Ultralytics (offline)
├── firmware/            # SmartPatch Firmware — C for RV1106 SoC (embedded)
├── shared/              # Protobuf schemas, JSON schemas, constants
│   ├── proto/           # gRPC .proto files (stream, biometrics, events)
│   └── schemas/         # JSON Schema definitions (stream, athlete, biometrics)
├── config/              # Hardware specs (SmartPatch v3.0)
├── demo-pipeline/       # Standalone demo pipeline (Python)
├── infra/               # Infrastructure-as-code
│   ├── kubernetes/      # K8s deployments, ingress, namespace
│   ├── terraform/       # AWS: EKS, CloudFront, RDS, ElastiCache
│   └── monitoring/      # Prometheus + Grafana dashboards
├── docs/                # Architecture, API docs, contributing guide
├── docker-compose.yml   # Production compose (all services + Kafka + Redis + TimescaleDB)
├── docker-compose.dev.yml # Dev compose (hot-reload, volume mounts)
└── Makefile             # Common commands (up, down, dev, test, lint, clean)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Gateway | Node.js 20+, TypeScript 5.3, Express 4, Socket.io 4, KafkaJS, ioredis |
| Python services | Python 3.11+, FastAPI 0.109, Pydantic v2, Loguru, Prometheus client |
| AI/ML | PyTorch 2.2+, Ultralytics (YOLO26), Transformers (ViTPose++), TensorRT, ONNX Runtime |
| Messaging | Apache Kafka (Confluent 7.6) |
| Databases | TimescaleDB (PostgreSQL 16), Redis 7 |
| Infra | Docker, Kubernetes (EKS 1.28), Terraform (AWS), Prometheus + Grafana |
| Firmware | C (RV1106 SoC), CMake |
| Schemas | Protocol Buffers 3, JSON Schema draft-07 |

## Build & Run Commands

```bash
make up                  # Start all services (docker compose up -d)
make down                # Stop all services
make dev                 # Start dev mode (hot-reload, volume mounts)
make test                # Run all tests (gateway + ai-engine + streaming + biometrics)
make lint                # Run all linters (eslint + ruff)
make download-models     # Download pretrained model weights
make clean               # Stop services, remove volumes, clear caches
```

### Running Individual Service Tests

```bash
cd gateway && npm test                          # Gateway tests (Jest)
cd ai-engine && python -m pytest tests/ -v      # AI Engine tests (pytest)
cd streaming && python -m pytest tests/ -v      # Streaming tests (pytest)
cd biometrics && python -m pytest tests/ -v     # Biometrics tests (pytest)
```

### Running Individual Service Linters

```bash
cd gateway && npm run lint                      # ESLint for TypeScript
cd ai-engine && ruff check src/                 # Ruff for Python
cd streaming && ruff check src/                 # Ruff for Python
cd biometrics && ruff check src/                # Ruff for Python
```

## Code Style & Conventions

### Python (ai-engine, streaming, biometrics, training)
- **Linter/Formatter**: Ruff + Black, line length 120
- **Type hints**: Use throughout; Pydantic v2 models for request/response schemas
- **Settings**: Use `pydantic-settings` `BaseSettings` for configuration (see `src/config.py` in each service)
- **Logging**: Use `loguru` (not stdlib `logging`). Use `logger.info("message {}", var)` format
- **Metrics**: Prometheus via `prometheus_client`. Mount at `/metrics` in each FastAPI app
- **Health checks**: Every service exposes `GET /health`
- **API prefix**: All routes under `/api/v1/`
- **Imports**: Relative imports within service packages (e.g., `from .config import settings`)
- **Async**: FastAPI endpoints use `async def`

### TypeScript (gateway)
- **Linter/Formatter**: ESLint + Prettier
- **Runtime**: Node.js 20+ with `tsx` for dev, compiled via `tsc` for production
- **Testing**: Jest with `ts-jest` preset. Tests in `tests/` directory, named `*.test.ts`
- **Logging**: Winston (JSON format with timestamps)
- **Metrics**: `prom-client` with custom registry
- **Validation**: Zod for runtime schema validation
- **Auth**: JWT-based with middleware in `src/middleware/auth.ts`

### C (firmware)
- **Style**: Linux kernel style — tabs for indentation, 80-column lines
- **Build**: CMake (`CMakeLists.txt` in firmware root)
- **Headers**: Single header `include/athleteview.h` for all public types and prototypes

### General
- **Docker**: Multi-stage builds with `dev` and `production` targets
- **Schemas**: Protobuf for inter-service communication, JSON Schema for API validation
- **Architecture decisions**: Major changes require an ADR in `docs/adr/`

## Service Architecture

### Communication Flow
```
SmartPatch → SRT/5G → Streaming Service (ingestion)
Streaming → Kafka → AI Engine (frame processing)
AI Engine → Enhanced frames → Streaming Service (distribution)
Biometrics → TimescaleDB → Real-time overlay + API
Gateway → REST/WebSocket → Clients
```

### Service Ports
| Service | Port | Protocol |
|---------|------|----------|
| Gateway | 3000 | HTTP/WS |
| AI Engine | 8001 | HTTP |
| Streaming | 8002 | HTTP |
| Streaming (SRT) | 9000 | UDP |
| Streaming (RTMP) | 1935 | TCP |
| Biometrics | 8003 | HTTP |
| Kafka | 9092 | TCP |
| Redis | 6379 | TCP |
| TimescaleDB | 5432 | TCP |

## AI Model Stack (v2.0)

| Task | Model | Notes |
|------|-------|-------|
| Super-Resolution (recorded) | VPEG | AIM 2025 winner, 5x cheaper than Real-ESRGAN |
| Super-Resolution (live) | FlashVSR | 17fps near-4K on A100 |
| Player Detection | YOLO26 | NMS-free, 43% faster CPU |
| Player Tracking | BoT-SORT | Multi-object tracker |
| Pose (live) | YOLO26-Pose | Bundled with detection, zero overhead |
| Pose (analysis) | ViTPose++ | Highest accuracy, post-match only |
| Depth (players) | Depth Pro | 1.8x better boundary F1 |
| Depth (field) | DAV2 Metric-Outdoor | Best outdoor metric depth |
| 3D Reconstruction | gsplat (3DGS) | 100+ fps rendering |
| Intelligence | Gemma 4 26B MoE | 256K context, native video |
| Audio | SpeechBrain | Commentary isolation |

Model configuration is in `ai-engine/src/config.py` (Settings class). Training configs in `training/configs/*.yaml`.

### AI Pipeline Order (live_pipeline.py)
Stabilize → Super-Res → Detect → Track → Pose → Overlay → Encode

## Environment Variables

Copy `.env.example` to `.env` before running. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | 3000 | Gateway port |
| `AI_ENGINE_PORT` | 8001 | AI Engine port |
| `GPU_DEVICE` | 0 | CUDA device index |
| `TENSORRT_ENABLED` | true | Enable TensorRT acceleration |
| `MODEL_CACHE_DIR` | /app/weights | Model weights directory |
| `KAFKA_BROKERS` | kafka:9092 | Kafka bootstrap servers |
| `REDIS_URL` | redis://redis:6379 | Redis connection |
| `TIMESCALEDB_URL` | postgresql://... | TimescaleDB connection |
| `SRT_LISTEN_PORT` | 9000 | SRT ingestion port |
| `PPG_SAMPLE_RATE` | 100 | Biometric PPG sample rate (Hz) |

## CI/CD

GitHub Actions workflow in `.github/workflows/ci.yml`:
- **Triggers**: Push to `main`, PRs targeting `main`
- **Jobs**: `gateway`, `ai-engine`, `streaming`, `biometrics` (run in parallel), then `docker` build
- Each service job: checkout → setup runtime → install deps → lint/test
- Docker job depends on all service jobs passing

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Add a new API endpoint | `gateway/src/routes/*.ts`, register in `gateway/src/index.ts` |
| Add a new AI model | `ai-engine/src/models/`, register in `ai-engine/src/models/registry.py` |
| Modify the live pipeline | `ai-engine/src/pipelines/live_pipeline.py` |
| Add training config | `training/configs/*.yaml` |
| Change biometric processing | `biometrics/src/sensors/`, `biometrics/src/analysis/` |
| Add streaming protocol | `streaming/src/ingestion/`, `streaming/src/distribution/` |
| Update protobuf schemas | `shared/proto/*.proto` |
| Update JSON schemas | `shared/schemas/*.json` |
| Modify infrastructure | `infra/terraform/`, `infra/kubernetes/` |
| Add monitoring dashboards | `infra/monitoring/grafana-dashboards/` |
| Firmware changes | `firmware/src/`, header: `firmware/include/athleteview.h` |
| Hardware spec updates | `config/hardware_spec_v3.json` |

## Testing Guidelines

- Tests live in `tests/` directories within each service
- Python tests use `pytest` — run with `python -m pytest tests/ -v`
- Gateway tests use Jest — run with `npm test`
- CI runs tests for all services on every PR
- Include tests with every feature PR

## Pull Request Conventions

- One feature per PR
- Include tests for new functionality
- Update docs if API surface changes
- Squash commits before merge
- Major architectural changes require an ADR in `docs/adr/`

## Important Notes

- **GPU required**: AI Engine needs NVIDIA GPU with CUDA for production inference. Falls back to CPU with warnings.
- **Model weights**: Not committed to git (`.gitignore` excludes `.pth`, `.pt`, `.onnx`, `.engine`). Run `make download-models` to fetch.
- **Secrets**: Never commit `.env` files. Use `.env.example` as template.
- **Media files**: Video files (`.mp4`, `.avi`, `.mkv`) are gitignored.
- **Real-ESRGAN is legacy**: v2.0 uses VPEG/FlashVSR. Real-ESRGAN remains as fallback only.
