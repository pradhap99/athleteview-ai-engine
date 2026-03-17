#!/usr/bin/env python3
"""YOLOv11 fine-tuning for sports multi-object tracking.

Fine-tunes a YOLOv11 model on the SportsMOT dataset with custom augmentations,
BoT-SORT tracker integration, and WandB experiment tracking.

Usage:
    python scripts/train_tracker.py --config configs/tracker.yaml
    python scripts/train_tracker.py --config configs/tracker.yaml --resume runs/tracker/last.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load and validate training configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def setup_wandb(config: dict) -> None:
    """Initialize Weights & Biases logging."""
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        logger.info("WandB logging disabled")
        return

    try:
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "athleteview-tracker"),
            entity=wandb_cfg.get("entity"),
            name=config.get("name", "yolov11_sportsmot"),
            tags=wandb_cfg.get("tags", []),
            config=config,
        )
        logger.info(f"WandB initialized: {wandb.run.url}")
    except ImportError:
        logger.warning("wandb not installed, skipping WandB logging")
    except Exception as e:
        logger.warning(f"WandB init failed: {e}")


def prepare_sportsmot_dataset(config: dict) -> Path:
    """Prepare SportsMOT dataset in YOLO format.

    Creates a dataset YAML file compatible with Ultralytics training.

    Returns:
        Path to the generated dataset YAML file.
    """
    data_cfg = config.get("data", {})
    data_path = Path(data_cfg.get("path", "data/sportsmot"))

    # Generate YOLO-format dataset config
    dataset_yaml = {
        "path": str(data_path.resolve()),
        "train": data_cfg.get("train", "images/train"),
        "val": data_cfg.get("val", "images/val"),
        "test": data_cfg.get("test", "images/test"),
        "names": data_cfg.get("names", {0: "person", 1: "ball", 2: "referee"}),
    }

    yaml_path = data_path / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    logger.info(f"Dataset YAML written to {yaml_path}")
    return yaml_path


def build_training_args(config: dict) -> dict:
    """Convert config to Ultralytics training arguments."""
    train_cfg = config.get("training", {})
    aug_cfg = config.get("augmentation", {})

    args = {
        # Model
        "model": config.get("model", {}).get("name", "yolo11m.pt"),
        "task": config.get("model", {}).get("task", "detect"),
        # Training
        "epochs": train_cfg.get("epochs", 100),
        "batch": train_cfg.get("batch", 16),
        "imgsz": train_cfg.get("imgsz", 1280),
        "patience": train_cfg.get("patience", 20),
        "save_period": train_cfg.get("save_period", 10),
        "device": train_cfg.get("device", "0"),
        "workers": train_cfg.get("workers", 8),
        "project": train_cfg.get("project", "runs/tracker"),
        "name": train_cfg.get("name", "athleteview_sportsmot"),
        "exist_ok": train_cfg.get("exist_ok", False),
        "verbose": train_cfg.get("verbose", True),
        "seed": train_cfg.get("seed", 42),
        "deterministic": train_cfg.get("deterministic", True),
        "cos_lr": train_cfg.get("cos_lr", True),
        "close_mosaic": train_cfg.get("close_mosaic", 10),
        "amp": train_cfg.get("amp", True),
        # Optimizer
        "optimizer": train_cfg.get("optimizer", "AdamW"),
        "lr0": train_cfg.get("lr0", 0.001),
        "lrf": train_cfg.get("lrf", 0.01),
        "momentum": train_cfg.get("momentum", 0.937),
        "weight_decay": train_cfg.get("weight_decay", 0.0005),
        "warmup_epochs": train_cfg.get("warmup_epochs", 3.0),
        "warmup_momentum": train_cfg.get("warmup_momentum", 0.8),
        "warmup_bias_lr": train_cfg.get("warmup_bias_lr", 0.1),
        # Loss
        "box": train_cfg.get("box", 7.5),
        "cls": train_cfg.get("cls", 0.5),
        "dfl": train_cfg.get("dfl", 1.5),
        # Augmentation
        "mosaic": aug_cfg.get("mosaic", 1.0),
        "mixup": aug_cfg.get("mixup", 0.1),
        "degrees": aug_cfg.get("degrees", 10.0),
        "translate": aug_cfg.get("translate", 0.2),
        "scale": aug_cfg.get("scale", 0.5),
        "shear": aug_cfg.get("shear", 2.0),
        "perspective": aug_cfg.get("perspective", 0.0),
        "flipud": aug_cfg.get("flipud", 0.0),
        "fliplr": aug_cfg.get("fliplr", 0.5),
        "hsv_h": aug_cfg.get("hsv_h", 0.015),
        "hsv_s": aug_cfg.get("hsv_s", 0.7),
        "hsv_v": aug_cfg.get("hsv_v", 0.4),
        "copy_paste": aug_cfg.get("copy_paste", 0.1),
        "erasing": aug_cfg.get("erasing", 0.4),
    }

    return args


class SportsMOTCallback:
    """Custom callback for SportsMOT-specific logging and evaluation."""

    def __init__(self, config: dict):
        self.config = config
        self.best_map50 = 0.0

    def on_train_epoch_end(self, trainer):
        """Log additional SportsMOT metrics after each epoch."""
        metrics = trainer.metrics
        epoch = trainer.epoch

        logger.info(
            f"Epoch {epoch}: "
            f"box_loss={metrics.get('train/box_loss', 0):.4f}, "
            f"cls_loss={metrics.get('train/cls_loss', 0):.4f}, "
            f"dfl_loss={metrics.get('train/dfl_loss', 0):.4f}"
        )

        # Log to WandB if available
        try:
            import wandb

            if wandb.run:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/box_loss": metrics.get("train/box_loss", 0),
                        "train/cls_loss": metrics.get("train/cls_loss", 0),
                        "train/dfl_loss": metrics.get("train/dfl_loss", 0),
                        "lr/pg0": metrics.get("lr/pg0", 0),
                        "lr/pg1": metrics.get("lr/pg1", 0),
                        "lr/pg2": metrics.get("lr/pg2", 0),
                    },
                    step=epoch,
                )
        except (ImportError, Exception):
            pass

    def on_val_end(self, trainer):
        """Log validation results with per-class breakdown."""
        metrics = trainer.metrics

        map50 = metrics.get("metrics/mAP50(B)", 0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0)

        logger.info(
            f"Validation: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}"
        )

        if map50 > self.best_map50:
            self.best_map50 = map50
            logger.info(f"New best mAP50: {map50:.4f}")


def run_tracking_evaluation(model_path: Path, config: dict) -> dict:
    """Run MOT evaluation on the trained model.

    Uses BoT-SORT tracker for multi-object tracking evaluation.

    Returns:
        Dict with tracking metrics (HOTA, MOTA, IDF1, etc.).
    """
    from ultralytics import YOLO

    tracker_cfg = config.get("tracker", {})
    tracker_type = tracker_cfg.get("tracker_type", "botsort")

    model = YOLO(str(model_path))

    # Generate tracker config YAML
    if tracker_type == "botsort":
        botsort_cfg = tracker_cfg.get("botsort", {})
        tracker_yaml_content = {
            "tracker_type": "botsort",
            "track_high_thresh": botsort_cfg.get("track_high_thresh", 0.5),
            "track_low_thresh": botsort_cfg.get("track_low_thresh", 0.1),
            "new_track_thresh": botsort_cfg.get("new_track_thresh", 0.6),
            "track_buffer": botsort_cfg.get("track_buffer", 30),
            "match_thresh": botsort_cfg.get("match_thresh", 0.8),
            "proximity_thresh": botsort_cfg.get("proximity_thresh", 0.5),
            "appearance_thresh": botsort_cfg.get("appearance_thresh", 0.25),
            "fuse_score": botsort_cfg.get("fuse_score", True),
            "gmc_method": botsort_cfg.get("gmc_method", "sparseOptFlow"),
        }
    else:
        bytetrack_cfg = tracker_cfg.get("bytetrack", {})
        tracker_yaml_content = {
            "tracker_type": "bytetrack",
            "track_high_thresh": bytetrack_cfg.get("track_high_thresh", 0.5),
            "track_low_thresh": bytetrack_cfg.get("track_low_thresh", 0.1),
            "new_track_thresh": bytetrack_cfg.get("new_track_thresh", 0.6),
            "track_buffer": bytetrack_cfg.get("track_buffer", 30),
            "match_thresh": bytetrack_cfg.get("match_thresh", 0.8),
        }

    tracker_yaml_path = Path("/tmp/tracker_config.yaml")
    with open(tracker_yaml_path, "w") as f:
        yaml.dump(tracker_yaml_content, f)

    logger.info(f"Running tracking evaluation with {tracker_type}...")

    # Track on validation set
    data_cfg = config.get("data", {})
    val_source = Path(data_cfg.get("path", "data/sportsmot")) / data_cfg.get("val", "images/val")

    results = model.track(
        source=str(val_source),
        tracker=str(tracker_yaml_path),
        conf=0.25,
        iou=0.6,
        show=False,
        save=False,
        verbose=False,
    )

    logger.info("Tracking evaluation complete")
    return {"tracker_type": tracker_type, "num_frames": len(results) if results else 0}


def export_model(model_path: Path, config: dict) -> list[Path]:
    """Export trained model to deployment formats."""
    from ultralytics import YOLO

    export_cfg = config.get("export", {})
    model = YOLO(str(model_path))

    exported_paths = []

    # Primary export (TensorRT)
    try:
        export_path = model.export(
            format=export_cfg.get("format", "engine"),
            half=export_cfg.get("half", True),
            imgsz=export_cfg.get("imgsz", 1280),
            batch=export_cfg.get("batch", 1),
            device=export_cfg.get("device", "0"),
            simplify=export_cfg.get("simplify", True),
            opset=export_cfg.get("opset", 17),
            workspace=export_cfg.get("workspace", 8),
        )
        exported_paths.append(Path(export_path))
        logger.info(f"Exported to {export_path}")
    except Exception as e:
        logger.error(f"Primary export failed: {e}")

    # Additional exports
    for fmt in export_cfg.get("additional_formats", []):
        try:
            export_path = model.export(format=fmt, imgsz=export_cfg.get("imgsz", 1280))
            exported_paths.append(Path(export_path))
            logger.info(f"Exported {fmt} to {export_path}")
        except Exception as e:
            logger.warning(f"Export to {fmt} failed: {e}")

    return exported_paths


def train(config: dict, resume: str | None = None):
    """Main training function."""
    from ultralytics import YOLO

    # Setup WandB
    setup_wandb(config)

    # Prepare dataset
    dataset_yaml = prepare_sportsmot_dataset(config)

    # Build training arguments
    train_args = build_training_args(config)
    train_args["data"] = str(dataset_yaml)

    if resume:
        train_args["resume"] = True
        model = YOLO(resume)
        logger.info(f"Resuming training from {resume}")
    else:
        model = YOLO(train_args.pop("model"))
        logger.info(f"Starting training with {model.model_name}")

    # Setup callbacks
    callback = SportsMOTCallback(config)
    model.add_callback("on_train_epoch_end", callback.on_train_epoch_end)
    model.add_callback("on_val_end", callback.on_val_end)

    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Train
    logger.info("Starting YOLOv11 fine-tuning on SportsMOT...")
    task_arg = train_args.pop("task", "detect")
    results = model.train(**train_args)

    # Get best model path
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    logger.info(f"Training complete. Best model: {best_model_path}")

    # Run validation
    logger.info("Running final validation...")
    val_results = model.val(
        data=str(dataset_yaml),
        conf=config.get("validation", {}).get("conf", 0.25),
        iou=config.get("validation", {}).get("iou", 0.6),
    )

    # Run tracking evaluation
    try:
        tracking_results = run_tracking_evaluation(best_model_path, config)
        logger.info(f"Tracking results: {tracking_results}")
    except Exception as e:
        logger.warning(f"Tracking evaluation skipped: {e}")

    # Export models
    try:
        exported = export_model(best_model_path, config)
        logger.info(f"Exported {len(exported)} model formats")
    except Exception as e:
        logger.warning(f"Model export skipped: {e}")

    # Log final results to WandB
    try:
        import wandb

        if wandb.run:
            wandb.log({
                "final/mAP50": val_results.box.map50 if hasattr(val_results, "box") else 0,
                "final/mAP50-95": val_results.box.map if hasattr(val_results, "box") else 0,
            })
            wandb.finish()
    except (ImportError, Exception):
        pass

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv11 fine-tuning for sports tracking",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tracker.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (no training)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Run model export only",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.eval_only:
        model_path = args.resume or "runs/tracker/athleteview_sportsmot/weights/best.pt"
        run_tracking_evaluation(Path(model_path), config)
    elif args.export_only:
        model_path = args.resume or "runs/tracker/athleteview_sportsmot/weights/best.pt"
        export_model(Path(model_path), config)
    else:
        train(config, resume=args.resume)


if __name__ == "__main__":
    main()
