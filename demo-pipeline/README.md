# AthleteView AI Demo Pipeline

Working demonstration of the AthleteView video processing stack. Runs entirely on CPU — no GPU required.

## What It Does

Takes input video → applies the full AthleteView AI pipeline → outputs enhanced video with:

1. **Super-Resolution** — Bicubic upscaling (production: VPEG + FlashVSR)
2. **Player Detection & Tracking** — Background subtraction + centroid tracking (production: YOLO26 + BoT-SORT)
3. **Biometric HUD Overlay** — Real-time heart rate, SpO2, body temperature, hydration display
4. **Video Stabilization** — Feature-based optical flow stabilization
5. **Before/After Comparison** — Side-by-side original vs enhanced output

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample test video (synthetic cricket field)
python generate_sample.py

# Run the full pipeline
python demo_pipeline.py sample_input.mp4

# Output files:
#   output/enhanced.mp4     — Full enhanced video with overlays
#   output/comparison.mp4   — Side-by-side before/after
```

## Options

```bash
python demo_pipeline.py input.mp4 --scale 2         # 2x super-resolution
python demo_pipeline.py input.mp4 --no-yolo          # Use HOG detector (no YOLO download)
python demo_pipeline.py input.mp4 --no-stabilize      # Skip stabilization
python demo_pipeline.py input.mp4 -o my_output/       # Custom output directory
```

## Module Overview

| Module | Purpose | Production Equivalent |
|--------|---------|----------------------|
| `demo_pipeline.py` | Main orchestrator | Gateway + AI Engine services |
| `overlay_engine.py` | Biometric HUD rendering | Biometrics service + Frontend |
| `tracker.py` | Player detection + tracking | YOLO26 + BoT-SORT pipeline |
| `stabilizer.py` | Video stabilization | BasicVSR++ / optical flow |
| `generate_sample.py` | Test video generator | Real camera input |

## Architecture (Production vs Demo)

```
Demo (this repo):
  Video File → OpenCV Read → Bicubic Upscale → BG Subtraction → HUD Overlay → Stabilize → MP4

Production (athleteview-ai-engine, Model Stack v2.0 April 2026):
  5G Stream → SRT Ingest → VPEG/FlashVSR → YOLO26+BoT-SORT → YOLO26-Pose (live) / ViTPose++ (analysis) →
  Depth Pro → gsplat 3D → Gemma 4 Commentary → LL-HLS/WebRTC/RTMP Distribution
```

## Requirements

- Python 3.8+
- ~500 MB disk (YOLOv8n model download)
- Works on any machine with CPU (no GPU needed)
- Tested on Linux, macOS, Windows

## License

Proprietary — AthleteView Technologies
