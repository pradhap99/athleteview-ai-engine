---
name: Bug Report
about: Report a bug in the AthleteView AI Platform
title: "[BUG] "
labels: bug, triage
assignees: ""
---

## Description

A clear and concise description of the bug.

## Service Affected

Which service(s) does this bug affect? Check all that apply:

- [ ] Gateway (Node.js API)
- [ ] AI Engine (Python/FastAPI)
- [ ] Streaming Service (Python/FastAPI)
- [ ] Biometrics Service (Python/FastAPI)
- [ ] Training Pipeline (Python/PyTorch)
- [ ] Infrastructure (Docker, Kafka, Redis, TimescaleDB)
- [ ] Monitoring (Prometheus, Grafana)
- [ ] Other: ___

## Environment

- **OS**: (e.g., Ubuntu 22.04, macOS 14.2, Windows 11)
- **Docker version**: (e.g., 24.0.7)
- **Docker Compose version**: (e.g., 2.23.3)
- **Node.js version** (if gateway): (e.g., 20.10.0)
- **Python version** (if Python service): (e.g., 3.11.7)
- **GPU model** (if AI engine): (e.g., NVIDIA RTX 4090)
- **CUDA version** (if AI engine): (e.g., 12.3)
- **Deployment method**: (e.g., docker compose, bare metal, Kubernetes)
- **Branch/Commit SHA**: (e.g., main / abc1234)

## Steps to Reproduce

1. Start the platform with `make dev`
2. Send a request to `POST /api/...`
3. Observe the error in the logs

Please include any relevant API requests (cURL commands, Postman exports, etc.):

```bash
curl -X POST http://localhost:3000/api/... \
  -H "Content-Type: application/json" \
  -d '{"key": "value"}'
```

## Expected Behavior

A clear description of what you expected to happen.

## Actual Behavior

A clear description of what actually happened, including any error messages.

## Logs

Paste relevant log output here. Use `make logs` or `make logs-<service>` to collect logs.

```
PASTE LOGS HERE
```

## Screenshots

If applicable, add screenshots to help explain the problem (especially for Grafana dashboards or streaming output issues).

## Additional Context

- Is this a regression? (Did it work in a previous version?)
- How frequently does this occur? (Always, intermittently, once)
- Any workarounds discovered?

## Possible Fix

If you have a suggestion for what might be causing this or how to fix it, please describe it here.
