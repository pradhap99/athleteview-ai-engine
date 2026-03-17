#!/usr/bin/env python3
"""Fine-tune ViTPose++ for sports pose estimation.

Loads a ViTPose++-Large model from HuggingFace and fine-tunes it on
COCO-format keypoint annotations with mixed-precision training,
cosine LR schedule, and TensorBoard / W&B logging.

Usage:
    python scripts/train_pose.py --config configs/pose.yaml
    python scripts/train_pose.py --config configs/pose.yaml --resume checkpoints/pose/best.pt
    python scripts/train_pose.py --config configs/pose.yaml --wandb --epochs 80
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate training configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PoseDataset(Dataset):
    """COCO-format keypoint dataset for pose estimation.

    Expects a COCO-style annotation JSON with ``images``, ``annotations``
    (each containing ``keypoints`` and ``bbox``), and ``categories``.
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        input_size: tuple[int, int] = (256, 192),
        heatmap_size: tuple[int, int] = (64, 48),
        num_joints: int = 17,
        is_train: bool = True,
        sigma: float = 2.0,
    ):
        self.img_dir = Path(img_dir)
        self.input_size = input_size  # (height, width)
        self.heatmap_size = heatmap_size  # (height, width)
        self.num_joints = num_joints
        self.is_train = is_train
        self.sigma = sigma

        with open(ann_file) as f:
            coco = json.load(f)

        # Build image lookup
        self.images = {img["id"]: img for img in coco["images"]}

        # Filter annotations that have keypoints
        self.annotations = [
            ann for ann in coco["annotations"]
            if ann.get("num_keypoints", sum(
                1 for i in range(2, len(ann.get("keypoints", [])), 3)
                if ann["keypoints"][i] > 0
            )) > 0
        ]
        logger.info(
            f"Loaded {len(self.annotations)} annotations from {ann_file} "
            f"({len(self.images)} images)"
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def _generate_heatmaps(
        self, keypoints: np.ndarray, visibility: np.ndarray,
    ) -> np.ndarray:
        """Generate Gaussian heatmaps for each visible joint."""
        heatmaps = np.zeros(
            (self.num_joints, self.heatmap_size[0], self.heatmap_size[1]),
            dtype=np.float32,
        )
        for j in range(self.num_joints):
            if visibility[j] < 1:
                continue
            mu_x = keypoints[j, 0] * self.heatmap_size[1]
            mu_y = keypoints[j, 1] * self.heatmap_size[0]
            # Generate 2-D Gaussian
            x = np.arange(self.heatmap_size[1], dtype=np.float32)
            y = np.arange(self.heatmap_size[0], dtype=np.float32)
            xx, yy = np.meshgrid(x, y)
            heatmaps[j] = np.exp(
                -((xx - mu_x) ** 2 + (yy - mu_y) ** 2) / (2 * self.sigma ** 2)
            )
        return heatmaps

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        img_info = self.images[ann["image_id"]]

        # Load image
        img_path = self.img_dir / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            # Return a zeroed sample on read failure
            return {
                "image": torch.zeros(3, self.input_size[0], self.input_size[1]),
                "heatmaps": torch.zeros(
                    self.num_joints, self.heatmap_size[0], self.heatmap_size[1]
                ),
                "target_weight": torch.zeros(self.num_joints),
            }
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop person bounding box
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        img_h, img_w = img.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            crop = img

        # Resize to input size
        crop = cv2.resize(crop, (self.input_size[1], self.input_size[0]))

        # Parse keypoints: [x1, y1, v1, x2, y2, v2, ...]
        kps = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        coords = kps[:, :2].copy()
        visibility = kps[:, 2].copy()

        # Normalise coordinates relative to bounding box
        bbox_w = max(x2 - x1, 1)
        bbox_h = max(y2 - y1, 1)
        coords[:, 0] = (coords[:, 0] - x1) / bbox_w
        coords[:, 1] = (coords[:, 1] - y1) / bbox_h
        coords = np.clip(coords, 0.0, 1.0)

        # Data augmentation (training only)
        if self.is_train:
            # Random horizontal flip
            if np.random.random() < 0.5:
                crop = crop[:, ::-1, :].copy()
                coords[:, 0] = 1.0 - coords[:, 0]

        # Normalise image
        image = crop.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)  # HWC -> CHW

        # Generate heatmaps
        heatmaps = self._generate_heatmaps(coords, visibility)

        target_weight = (visibility > 0).astype(np.float32)

        return {
            "image": torch.from_numpy(image),
            "heatmaps": torch.from_numpy(heatmaps),
            "target_weight": torch.from_numpy(target_weight),
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_vitpose_model(config: dict, device: torch.device) -> nn.Module:
    """Load ViTPose++ model from HuggingFace."""
    from transformers import AutoModel

    model_cfg = config.get("model", {})
    hub_id = model_cfg.get("hub_id", "usyd-community/vitpose-plus-large")
    num_joints = config.get("keypoints", {}).get("num_joints", 17)

    logger.info(f"Loading ViTPose model from {hub_id} ...")
    model = AutoModel.from_pretrained(hub_id, trust_remote_code=True)

    # Ensure the final head outputs the correct number of joints
    if hasattr(model, "head") and hasattr(model.head, "final_layer"):
        out_channels = model.head.final_layer.out_channels
        if out_channels != num_joints:
            logger.info(
                f"Replacing final layer: {out_channels} -> {num_joints} joints"
            )
            in_channels = model.head.final_layer.in_channels
            model.head.final_layer = nn.Conv2d(
                in_channels, num_joints, kernel_size=1,
            )

    model = model.to(device)
    logger.info(
        f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)"
    )
    return model


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def cosine_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
):
    """Update learning rate with linear warmup + cosine annealing."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_pck(
    pred_heatmaps: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    target_weight: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute PCK (Percentage of Correct Keypoints) at a given threshold."""
    batch, num_joints, h, w = pred_heatmaps.shape

    # Decode predicted coordinates from heatmaps
    pred_flat = pred_heatmaps.view(batch, num_joints, -1)
    gt_flat = gt_heatmaps.view(batch, num_joints, -1)

    pred_idx = pred_flat.argmax(dim=2)
    gt_idx = gt_flat.argmax(dim=2)

    pred_x = (pred_idx % w).float()
    pred_y = (pred_idx // w).float()
    gt_x = (gt_idx % w).float()
    gt_y = (gt_idx // w).float()

    # Normalise distance by head size (approximate as heatmap height)
    dist = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) / h
    mask = target_weight > 0

    correct = ((dist < threshold) & mask).float().sum()
    total = mask.float().sum()
    return (correct / total).item() if total > 0 else 0.0


# ---------------------------------------------------------------------------
# W&B helper
# ---------------------------------------------------------------------------

def setup_wandb(config: dict, enabled: bool) -> None:
    """Initialize Weights & Biases logging."""
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if not enabled and not wandb_cfg.get("enabled", False):
        logger.info("WandB logging disabled")
        return
    try:
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "athleteview-pose"),
            entity=wandb_cfg.get("entity"),
            name=config.get("name", "vitpose_plus_large"),
            tags=wandb_cfg.get("tags", []),
            config=config,
        )
        logger.info(f"WandB initialized: {wandb.run.url}")
    except ImportError:
        logger.warning("wandb not installed, skipping")
    except Exception as e:
        logger.warning(f"WandB init failed: {e}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict, args: argparse.Namespace):
    """Main training entry point."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Hyperparameters from config + CLI overrides
    train_cfg = config.get("training", {})
    epochs = args.epochs or train_cfg.get("epochs", 50)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    base_lr = train_cfg.get("optimizer", {}).get("lr", 5e-4)
    weight_decay = train_cfg.get("optimizer", {}).get("weight_decay", 0.1)
    warmup_epochs = train_cfg.get("lr_scheduler", {}).get("warmup_epochs", 5)
    min_lr = train_cfg.get("lr_scheduler", {}).get("min_lr", 1e-6)
    use_amp = train_cfg.get("use_amp", True)
    grad_clip_norm = train_cfg.get("grad_clip", {}).get("max_norm", 1.0)

    data_cfg = config.get("data", {})
    input_size = tuple(data_cfg.get("input_size", [256, 192]))
    heatmap_size = tuple(data_cfg.get("heatmap_size", [64, 48]))
    num_joints = config.get("keypoints", {}).get("num_joints", 17)

    # Setup W&B
    setup_wandb(config, enabled=args.wandb)

    # Datasets
    train_ds = PoseDataset(
        ann_file=data_cfg["train"]["ann_file"],
        img_dir=data_cfg["train"]["img_dir"],
        input_size=input_size,
        heatmap_size=heatmap_size,
        num_joints=num_joints,
        is_train=True,
    )
    val_ds = PoseDataset(
        ann_file=data_cfg["val"]["ann_file"],
        img_dir=data_cfg["val"]["img_dir"],
        input_size=input_size,
        heatmap_size=heatmap_size,
        num_joints=num_joints,
        is_train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=True,
    )

    # Model
    model = load_vitpose_model(config, device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=tuple(train_cfg.get("optimizer", {}).get("betas", [0.9, 0.999])),
        weight_decay=weight_decay,
    )

    # Loss
    criterion = nn.MSELoss()

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    # TensorBoard
    log_cfg = config.get("logging", {})
    tb_dir = log_cfg.get("tensorboard_dir", "logs/pose/tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    # Checkpointing
    ckpt_cfg = config.get("checkpoint", {})
    save_dir = Path(ckpt_cfg.get("save_dir", "checkpoints/pose"))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_metric = 0.0
    start_epoch = 0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    # Log system info
    logger.info(f"PyTorch {torch.__version__} | Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    logger.info(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {base_lr}")

    # ---- Training loop ----
    for epoch in range(start_epoch, epochs):
        model.train()
        lr = cosine_lr_with_warmup(
            optimizer, epoch, epochs, warmup_epochs, base_lr, min_lr,
        )
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            gt_heatmaps = batch["heatmaps"].to(device)
            target_weight = batch["target_weight"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                outputs = model(images)
                pred_heatmaps = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                # Weighted MSE loss
                weight_mask = target_weight.unsqueeze(-1).unsqueeze(-1)
                loss = criterion(pred_heatmaps * weight_mask, gt_heatmaps * weight_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        pck_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                gt_heatmaps = batch["heatmaps"].to(device)
                target_weight = batch["target_weight"].to(device)

                outputs = model(images)
                pred_heatmaps = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                weight_mask = target_weight.unsqueeze(-1).unsqueeze(-1)
                loss = criterion(pred_heatmaps * weight_mask, gt_heatmaps * weight_mask)

                val_loss += loss.item()
                pck_total += compute_pck(pred_heatmaps, gt_heatmaps, target_weight, threshold=0.5)
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        avg_pck = pck_total / max(val_batches, 1)

        logger.info(
            f"Epoch {epoch}/{epochs-1} | "
            f"train_loss={avg_loss:.4f} | val_loss={avg_val_loss:.4f} | "
            f"PCK@0.5={avg_pck:.4f} | lr={lr:.2e} | {elapsed:.1f}s"
        )

        # TensorBoard logging
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        writer.add_scalar("val/PCK@0.5", avg_pck, epoch)
        writer.add_scalar("train/lr", lr, epoch)

        # W&B logging
        try:
            import wandb

            if wandb.run:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": avg_loss,
                    "val/loss": avg_val_loss,
                    "val/PCK@0.5": avg_pck,
                    "train/lr": lr,
                }, step=epoch)
        except (ImportError, Exception):
            pass

        # Checkpointing
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": config,
        }

        # Save latest
        torch.save(ckpt_data, save_dir / "latest.pt")

        # Save best
        if avg_pck > best_metric:
            best_metric = avg_pck
            ckpt_data["best_metric"] = best_metric
            torch.save(ckpt_data, save_dir / "best.pt")
            logger.info(f"New best PCK@0.5: {best_metric:.4f}")

        # Periodic saving
        max_keep = ckpt_cfg.get("max_keep", 5)
        if (epoch + 1) % max(1, epochs // max_keep) == 0:
            torch.save(ckpt_data, save_dir / f"epoch_{epoch}.pt")

    writer.close()
    logger.info(f"Training complete. Best PCK@0.5: {best_metric:.4f}")
    logger.info(f"Checkpoints saved to {save_dir}")

    try:
        import wandb

        if wandb.run:
            wandb.log({"final/best_PCK@0.5": best_metric})
            wandb.finish()
    except (ImportError, Exception):
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ViTPose++ fine-tuning for sports pose estimation",
    )
    parser.add_argument(
        "--config", type=str, default="configs/pose.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device string (e.g. cuda:0, cpu)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI data-dir override
    if args.data_dir:
        data_root = Path(args.data_dir)
        for split in ("train", "val", "test"):
            if split in config.get("data", {}):
                cfg = config["data"][split]
                cfg["img_dir"] = str(data_root / Path(cfg["img_dir"]).name)

    train(config, args)


if __name__ == "__main__":
    main()
