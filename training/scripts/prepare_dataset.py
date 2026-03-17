#!/usr/bin/env python3
"""Dataset preparation pipeline for AthleteView AI Engine.

Handles:
- Video to frame extraction
- Annotation format conversion (COCO <-> YOLO)
- Train/val/test splits with stratification
- Augmentation preview generation

Usage:
    python scripts/prepare_dataset.py --task extract_frames --video-dir /data/videos --output /data/frames
    python scripts/prepare_dataset.py --task convert_annotations --format coco-to-yolo --input /data/coco --output /data/yolo
    python scripts/prepare_dataset.py --task split --input /data/frames --ratios 0.8 0.1 0.1
    python scripts/prepare_dataset.py --task augmentation_preview --input /data/frames --num-samples 10
"""

import argparse
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import albumentations as A
except ImportError:
    A = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Video to Frame Extraction
# ============================================================================


def extract_frames(
    video_dir: Path,
    output_dir: Path,
    fps: int | None = None,
    frame_interval: int = 1,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
    quality: int = 95,
) -> dict:
    """Extract frames from videos in a directory.

    Args:
        video_dir: Directory containing video files.
        output_dir: Output directory for extracted frames.
        fps: Target FPS (None to use video's native FPS).
        frame_interval: Extract every Nth frame.
        max_frames: Maximum frames per video (None for all).
        resize: Target (width, height) or None.
        quality: JPEG save quality (1-100).

    Returns:
        Statistics dict with counts per video.
    """
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts"}
    video_files = sorted(
        f for f in video_dir.rglob("*") if f.suffix.lower() in video_extensions
    )

    if not video_files:
        logger.error(f"No video files found in {video_dir}")
        return {}

    logger.info(f"Found {len(video_files)} videos in {video_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for video_path in tqdm(video_files, desc="Extracting frames"):
        video_name = video_path.stem
        frame_dir = output_dir / video_name
        frame_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            continue

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Compute frame skip interval based on target FPS
        if fps and native_fps > 0:
            skip = max(1, int(round(native_fps / fps)))
        else:
            skip = frame_interval

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip == 0:
                if max_frames and saved_count >= max_frames:
                    break

                if resize:
                    frame = cv2.resize(
                        frame, resize, interpolation=cv2.INTER_LANCZOS4
                    )

                frame_path = frame_dir / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality],
                )
                saved_count += 1

            frame_count += 1

        cap.release()

        stats[video_name] = {
            "total_frames": total_frames,
            "extracted_frames": saved_count,
            "native_fps": native_fps,
            "skip_interval": skip,
        }
        logger.info(
            f"  {video_name}: {saved_count}/{total_frames} frames "
            f"(FPS: {native_fps:.1f}, skip: {skip})"
        )

    # Save extraction metadata
    meta_path = output_dir / "extraction_meta.json"
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    total_extracted = sum(s["extracted_frames"] for s in stats.values())
    logger.info(
        f"Extraction complete: {total_extracted} frames from {len(stats)} videos"
    )
    return stats


# ============================================================================
# Annotation Format Conversion
# ============================================================================


def coco_to_yolo(coco_dir: Path, output_dir: Path, classes: list[str] | None = None):
    """Convert COCO format annotations to YOLO format.

    COCO format: JSON with bounding boxes as [x, y, width, height] (absolute).
    YOLO format: .txt files with [class_id, x_center, y_center, width, height] (normalized).

    Args:
        coco_dir: Directory with COCO annotation JSON and images.
        output_dir: Output directory for YOLO format data.
        classes: Class name list for remapping (None to use COCO category order).
    """
    # Find annotation file
    ann_candidates = list(coco_dir.glob("annotations/*.json"))
    if not ann_candidates:
        ann_candidates = list(coco_dir.glob("*.json"))
    if not ann_candidates:
        logger.error(f"No COCO annotation JSON found in {coco_dir}")
        return

    for ann_file in ann_candidates:
        logger.info(f"Converting {ann_file.name}...")

        with open(ann_file) as f:
            coco_data = json.load(f)

        # Build category mapping
        categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
        if classes:
            cat_to_yolo = {}
            for cat_id, cat_name in categories.items():
                if cat_name in classes:
                    cat_to_yolo[cat_id] = classes.index(cat_name)
        else:
            cat_ids = sorted(categories.keys())
            cat_to_yolo = {cid: idx for idx, cid in enumerate(cat_ids)}

        # Build image dimension lookup
        images = {
            img["id"]: {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }
            for img in coco_data.get("images", [])
        }

        # Group annotations by image
        img_annotations = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            if cat_id not in cat_to_yolo:
                continue

            img_info = images.get(img_id)
            if not img_info:
                continue

            # Convert COCO bbox [x, y, w, h] to YOLO [x_center, y_center, w, h] normalized
            x, y, w, h = ann["bbox"]
            img_w, img_h = img_info["width"], img_info["height"]

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            yolo_class = cat_to_yolo[cat_id]
            img_annotations[img_id].append(
                f"{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        # Write YOLO labels
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        for img_id, img_info in images.items():
            label_name = Path(img_info["file_name"]).stem + ".txt"
            label_path = labels_dir / label_name

            annotations = img_annotations.get(img_id, [])
            with open(label_path, "w") as f:
                f.write("\n".join(annotations))

        # Write classes file
        if classes:
            class_names = classes
        else:
            class_names = [categories[cid] for cid in sorted(categories.keys())]

        classes_path = output_dir / "classes.txt"
        with open(classes_path, "w") as f:
            f.write("\n".join(class_names))

        logger.info(
            f"Converted {len(images)} images, "
            f"{sum(len(v) for v in img_annotations.values())} annotations"
        )


def yolo_to_coco(
    yolo_dir: Path,
    output_dir: Path,
    img_dir: Path | None = None,
) -> dict:
    """Convert YOLO format annotations to COCO format.

    Args:
        yolo_dir: Directory with YOLO .txt label files.
        output_dir: Output directory for COCO JSON.
        img_dir: Image directory (to read dimensions). Defaults to yolo_dir/../images.

    Returns:
        COCO format annotation dict.
    """
    labels_dir = yolo_dir / "labels" if (yolo_dir / "labels").is_dir() else yolo_dir
    if img_dir is None:
        img_dir = yolo_dir / "images" if (yolo_dir / "images").is_dir() else yolo_dir

    # Load class names
    classes_file = yolo_dir / "classes.txt"
    if classes_file.exists():
        class_names = classes_file.read_text().strip().split("\n")
    else:
        logger.warning("No classes.txt found; using numeric IDs as class names")
        class_names = None

    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    ann_id = 1
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    label_files = sorted(labels_dir.glob("*.txt"))
    logger.info(f"Converting {len(label_files)} YOLO label files to COCO...")

    category_ids_seen = set()

    for img_id, label_file in enumerate(tqdm(label_files, desc="Converting"), start=1):
        stem = label_file.stem

        # Find corresponding image
        img_path = None
        for ext in img_extensions:
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            logger.warning(f"No image found for {label_file.name}")
            continue

        # Get image dimensions
        img = Image.open(img_path)
        img_w, img_h = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        # Parse YOLO annotations
        lines = label_file.read_text().strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h

            x = x_center - w / 2
            y = y_center - h / 2

            category_ids_seen.add(cls_id)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,  # COCO categories are 1-indexed
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    # Build categories
    for cls_id in sorted(category_ids_seen):
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        coco["categories"].append({
            "id": cls_id + 1,
            "name": name,
            "supercategory": "object",
        })

    # Save COCO JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "annotations.json"
    with open(output_file, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        f"Saved COCO annotations: {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations, "
        f"{len(coco['categories'])} categories"
    )
    return coco


# ============================================================================
# Train/Val/Test Splitting
# ============================================================================


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    by_video: bool = True,
    copy_files: bool = True,
):
    """Split dataset into train/val/test sets.

    Args:
        input_dir: Input directory containing images (and optionally labels).
        output_dir: Output directory with train/val/test subdirectories.
        ratios: (train, val, test) ratios. Must sum to ~1.0.
        seed: Random seed for reproducibility.
        by_video: If True, split by video directory (keeps frames from same video together).
        copy_files: If True, copy files; otherwise create symlinks.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"

    random.seed(seed)
    np.random.seed(seed)

    images_dir = input_dir / "images" if (input_dir / "images").is_dir() else input_dir
    labels_dir = input_dir / "labels" if (input_dir / "labels").is_dir() else None

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if by_video:
        # Group by parent directory (video name)
        video_dirs = sorted(
            {f.parent for f in images_dir.rglob("*") if f.suffix.lower() in img_extensions}
        )
        if len(video_dirs) <= 1:
            by_video = False
            logger.info("Single directory detected, splitting by individual files")

    if by_video:
        items = list(video_dirs)
    else:
        items = sorted(f for f in images_dir.rglob("*") if f.suffix.lower() in img_extensions)

    random.shuffle(items)

    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val :]

    splits = {"train": train_items, "val": val_items, "test": test_items}

    for split_name, split_items in splits.items():
        split_img_dir = output_dir / "images" / split_name
        split_lbl_dir = output_dir / "labels" / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        if labels_dir:
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for item in tqdm(split_items, desc=f"Creating {split_name} split"):
            if by_video:
                files = sorted(f for f in item.rglob("*") if f.suffix.lower() in img_extensions)
            else:
                files = [item]

            for img_file in files:
                dest_img = split_img_dir / img_file.name
                if copy_files:
                    shutil.copy2(img_file, dest_img)
                else:
                    dest_img.symlink_to(img_file.resolve())

                # Copy corresponding label if it exists
                if labels_dir:
                    label_file = labels_dir / (img_file.stem + ".txt")
                    if label_file.exists():
                        dest_label = split_lbl_dir / label_file.name
                        if copy_files:
                            shutil.copy2(label_file, dest_label)
                        else:
                            dest_label.symlink_to(label_file.resolve())

                count += 1

        logger.info(f"  {split_name}: {count} images ({len(split_items)} groups)")

    # Write split metadata
    meta = {
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "counts": {
            name: len(items) for name, items in splits.items()
        },
        "split_by": "video" if by_video else "file",
    }
    meta_path = output_dir / "split_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Split metadata saved to {meta_path}")


# ============================================================================
# Augmentation Preview
# ============================================================================


def get_sports_augmentation_pipeline() -> "A.Compose":
    """Build the sports-specific augmentation pipeline."""
    if A is None:
        raise ImportError("albumentations is required for augmentation preview")

    return A.Compose(
        [
            A.RandomResizedCrop(height=640, width=640, scale=(0.5, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=4.0, p=1.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10, 50), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ],
                p=0.2,
            ),
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                min_holes=1, min_height=8, min_width=8,
                p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3
        ),
    )


def augmentation_preview(
    input_dir: Path,
    output_dir: Path,
    num_samples: int = 10,
    num_augmentations: int = 5,
):
    """Generate augmentation preview images.

    For each sample image, generates multiple augmented versions side by side.

    Args:
        input_dir: Directory containing sample images.
        output_dir: Output directory for preview grids.
        num_samples: Number of sample images to augment.
        num_augmentations: Number of augmented versions per image.
    """
    if A is None:
        logger.error("albumentations is required. Install with: pip install albumentations")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png"}
    images = sorted(f for f in input_dir.rglob("*") if f.suffix.lower() in img_extensions)

    if not images:
        logger.error(f"No images found in {input_dir}")
        return

    # Sample images
    if len(images) > num_samples:
        images = random.sample(images, num_samples)

    transform = get_sports_augmentation_pipeline()

    for img_path in tqdm(images, desc="Generating augmentation previews"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Generate augmented versions
        augmented_images = [img_rgb]  # original first
        for _ in range(num_augmentations):
            try:
                result = transform(image=img_rgb, bboxes=[], class_labels=[])
                augmented_images.append(result["image"])
            except Exception as e:
                logger.warning(f"Augmentation failed for {img_path.name}: {e}")
                continue

        # Create grid image
        target_h = 320
        resized = []
        for aug_img in augmented_images:
            h, w = aug_img.shape[:2]
            scale = target_h / h
            new_w = int(w * scale)
            resized.append(cv2.resize(aug_img, (new_w, target_h)))

        # Pad to same width
        max_w = max(r.shape[1] for r in resized)
        padded = []
        for r in resized:
            if r.shape[1] < max_w:
                pad = np.zeros((target_h, max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)

        grid = np.vstack(padded)
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(grid_bgr, "Original", (10, 25), font, 0.7, (0, 255, 0), 2)
        for i in range(num_augmentations):
            y = target_h * (i + 1) + 25
            cv2.putText(grid_bgr, f"Aug #{i + 1}", (10, y), font, 0.7, (0, 255, 255), 2)

        output_path = output_dir / f"preview_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), grid_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

    logger.info(f"Augmentation previews saved to {output_dir}")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AthleteView AI Engine - Dataset Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="task", help="Task to perform")

    # Extract frames
    extract_parser = subparsers.add_parser("extract_frames", help="Extract frames from videos")
    extract_parser.add_argument("--video-dir", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--fps", type=int, default=None)
    extract_parser.add_argument("--frame-interval", type=int, default=1)
    extract_parser.add_argument("--max-frames", type=int, default=None)
    extract_parser.add_argument("--resize", type=int, nargs=2, default=None)
    extract_parser.add_argument("--quality", type=int, default=95)

    # Convert annotations
    convert_parser = subparsers.add_parser("convert", help="Convert annotation formats")
    convert_parser.add_argument(
        "--format",
        choices=["coco-to-yolo", "yolo-to-coco"],
        required=True,
    )
    convert_parser.add_argument("--input", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--img-dir", type=Path, default=None)
    convert_parser.add_argument("--classes", nargs="+", default=None)

    # Split dataset
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--input", type=Path, required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--by-video", action="store_true", default=True)
    split_parser.add_argument("--symlink", action="store_true")

    # Augmentation preview
    aug_parser = subparsers.add_parser("augmentation_preview", help="Generate augmentation previews")
    aug_parser.add_argument("--input", type=Path, required=True)
    aug_parser.add_argument("--output", type=Path, required=True)
    aug_parser.add_argument("--num-samples", type=int, default=10)
    aug_parser.add_argument("--num-augmentations", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == "extract_frames":
        extract_frames(
            video_dir=args.video_dir,
            output_dir=args.output,
            fps=args.fps,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            resize=tuple(args.resize) if args.resize else None,
            quality=args.quality,
        )
    elif args.task == "convert":
        if args.format == "coco-to-yolo":
            coco_to_yolo(
                coco_dir=args.input,
                output_dir=args.output,
                classes=args.classes,
            )
        elif args.format == "yolo-to-coco":
            yolo_to_coco(
                yolo_dir=args.input,
                output_dir=args.output,
                img_dir=args.img_dir,
            )
    elif args.task == "split":
        split_dataset(
            input_dir=args.input,
            output_dir=args.output,
            ratios=tuple(args.ratios),
            seed=args.seed,
            by_video=args.by_video,
            copy_files=not args.symlink,
        )
    elif args.task == "augmentation_preview":
        augmentation_preview(
            input_dir=args.input,
            output_dir=args.output,
            num_samples=args.num_samples,
            num_augmentations=args.num_augmentations,
        )
    else:
        logger.error("No task specified. Use --help for usage.")


if __name__ == "__main__":
    main()
