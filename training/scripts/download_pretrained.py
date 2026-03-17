"""Download all pretrained model weights for AthleteView AI Engine."""
import os, urllib.request, sys
from pathlib import Path

WEIGHTS_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "weights"))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1.0/RealESRGAN_x2plus.pth",
    "yolo11x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}

def download(name: str, url: str):
    dest = WEIGHTS_DIR / name
    if dest.exists():
        print(f"[SKIP] {name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[DOWNLOAD] {name} from {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[DONE] {name} ({dest.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    print(f"Downloading pretrained weights to {WEIGHTS_DIR}")
    for name, url in MODELS.items():
        try:
            download(name, url)
        except Exception as e:
            print(f"[ERROR] {name}: {e}", file=sys.stderr)
    print("\nViTPose++ and SpeechBrain models are downloaded automatically from HuggingFace on first use.")
    print("Done!")
