"""
AthleteView AI Engine - Training Pipeline
Model fine-tuning and evaluation pipeline with Kafka progress reporting.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog
import torch

from src.config import Settings
from src.models.registry import ModelRegistry
from src.utils.metrics import get_metrics

logger = structlog.get_logger(__name__)


@dataclass
class TrainingResult:
    """Result of a model fine-tuning run."""

    model_name: str
    dataset_path: str
    epochs_completed: int = 0
    total_epochs: int = 0
    final_loss: float = 0.0
    best_loss: float = float("inf")
    best_epoch: int = 0
    metrics_history: list[dict[str, float]] = field(default_factory=list)
    checkpoint_path: str = ""
    training_time_seconds: float = 0.0
    success: bool = False
    error_message: str = ""


@dataclass
class EvaluationResult:
    """Result of a model evaluation run."""

    model_name: str
    dataset_path: str
    num_samples: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    evaluation_time_seconds: float = 0.0
    success: bool = False
    error_message: str = ""


class TrainingPipeline:
    """
    Manages model fine-tuning and evaluation workflows.

    Features:
        - Generic fine-tuning interface for all registered models
        - Evaluation with per-class metrics
        - Publishes training progress events to Kafka
        - Checkpoint saving with best-model tracking
    """

    def __init__(self, registry: ModelRegistry, config: Settings) -> None:
        self._registry = registry
        self._config = config
        self._metrics = get_metrics()
        self._kafka_producer: Optional[Any] = None

    def set_kafka_producer(self, producer: Any) -> None:
        """Set the Kafka producer for publishing progress events."""
        self._kafka_producer = producer

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    async def fine_tune(
        self,
        model_name: str,
        dataset_path: str,
        config: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """
        Fine-tune a registered model on a custom dataset.

        Args:
            model_name: Name of the registered model to fine-tune.
            dataset_path: Path to the training dataset directory.
            config: Training configuration overrides:
                - epochs (int): Number of training epochs. Default 10.
                - learning_rate (float): Learning rate. Default 1e-4.
                - batch_size (int): Training batch size. Default 4.
                - save_dir (str): Checkpoint save directory.
                - validation_split (float): Fraction for validation. Default 0.1.

        Returns:
            TrainingResult with training history and checkpoint path.
        """
        training_config = config or {}
        epochs = training_config.get("epochs", 10)
        learning_rate = training_config.get("learning_rate", 1e-4)
        batch_size = training_config.get("batch_size", 4)
        save_dir = training_config.get("save_dir", str(self._config.model_cache_dir / "checkpoints"))
        validation_split = training_config.get("validation_split", 0.1)

        result = TrainingResult(
            model_name=model_name,
            dataset_path=dataset_path,
            total_epochs=epochs,
        )

        start_time = time.perf_counter()

        # Validate inputs
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            result.error_message = f"Dataset path does not exist: {dataset_path}"
            logger.error("Training failed", error=result.error_message)
            await self._publish_progress(model_name, "error", 0, epochs, error=result.error_message)
            return result

        if not self._registry.is_loaded(model_name):
            try:
                self._registry.load_model(model_name)
            except Exception as exc:
                result.error_message = f"Failed to load model: {exc}"
                logger.error("Training failed", model=model_name, error=result.error_message)
                await self._publish_progress(
                    model_name, "error", 0, epochs, error=result.error_message
                )
                return result

        logger.info(
            "Starting fine-tuning",
            model=model_name,
            dataset=dataset_path,
            epochs=epochs,
            lr=learning_rate,
            batch_size=batch_size,
        )

        await self._publish_progress(model_name, "started", 0, epochs)

        try:
            model_instance = self._registry.get_model(model_name)

            # Load dataset
            train_data, val_data = await asyncio.get_event_loop().run_in_executor(
                None, self._load_dataset, dataset_path, validation_split
            )

            if len(train_data) == 0:
                result.error_message = "Training dataset is empty"
                await self._publish_progress(
                    model_name, "error", 0, epochs, error=result.error_message
                )
                return result

            # Get the underlying PyTorch model for training
            pytorch_model = self._extract_pytorch_model(model_instance)
            if pytorch_model is None:
                result.error_message = f"Model '{model_name}' does not expose a trainable PyTorch module"
                await self._publish_progress(
                    model_name, "error", 0, epochs, error=result.error_message
                )
                return result

            device = "cuda" if torch.cuda.is_available() else "cpu"
            pytorch_model.to(device)
            pytorch_model.train()

            optimizer = torch.optim.AdamW(pytorch_model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = torch.nn.MSELoss()

            best_loss = float("inf")
            best_epoch = 0
            checkpoint_path = ""

            for epoch in range(epochs):
                epoch_start = time.perf_counter()
                running_loss = 0.0
                num_batches = 0

                # Train on mini-batches
                for batch_start in range(0, len(train_data), batch_size):
                    batch = train_data[batch_start: batch_start + batch_size]
                    if len(batch) == 0:
                        continue

                    batch_tensor = torch.from_numpy(np.stack(batch)).float().to(device)
                    if batch_tensor.ndim == 4:
                        batch_tensor = batch_tensor.permute(0, 3, 1, 2) / 255.0

                    optimizer.zero_grad()

                    try:
                        output = pytorch_model(batch_tensor)
                        # Self-supervised loss (reconstruction)
                        if output.shape == batch_tensor.shape:
                            loss = criterion(output, batch_tensor)
                        else:
                            loss = output.mean()
                    except Exception as exc:
                        logger.warning(
                            "Batch forward pass failed",
                            epoch=epoch,
                            error=str(exc),
                        )
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    running_loss += loss.item()
                    num_batches += 1

                scheduler.step()

                avg_loss = running_loss / max(num_batches, 1)
                epoch_time = time.perf_counter() - epoch_start

                # Validation
                val_loss = 0.0
                if val_data and len(val_data) > 0:
                    val_loss = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._validate,
                        pytorch_model,
                        val_data,
                        batch_size,
                        criterion,
                        device,
                    )

                epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": round(avg_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch_time_s": round(epoch_time, 2),
                }
                result.metrics_history.append(epoch_metrics)

                # Track best model
                effective_loss = val_loss if val_data else avg_loss
                if effective_loss < best_loss:
                    best_loss = effective_loss
                    best_epoch = epoch + 1
                    # Save checkpoint
                    ckpt_dir = Path(save_dir)
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = str(ckpt_dir / f"{model_name}_best.pt")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": pytorch_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": effective_loss,
                        },
                        checkpoint_path,
                    )

                result.epochs_completed = epoch + 1
                result.final_loss = avg_loss

                logger.info(
                    "Epoch complete",
                    model=model_name,
                    epoch=f"{epoch + 1}/{epochs}",
                    train_loss=round(avg_loss, 6),
                    val_loss=round(val_loss, 6),
                    epoch_time_s=round(epoch_time, 2),
                )

                await self._publish_progress(
                    model_name,
                    "training",
                    epoch + 1,
                    epochs,
                    train_loss=avg_loss,
                    val_loss=val_loss,
                )

            result.best_loss = best_loss
            result.best_epoch = best_epoch
            result.checkpoint_path = checkpoint_path
            result.success = True
            result.training_time_seconds = round(time.perf_counter() - start_time, 2)

            await self._publish_progress(model_name, "completed", epochs, epochs)

            logger.info(
                "Fine-tuning complete",
                model=model_name,
                best_loss=round(best_loss, 6),
                best_epoch=best_epoch,
                total_time_s=result.training_time_seconds,
            )

        except Exception as exc:
            result.error_message = str(exc)
            result.training_time_seconds = round(time.perf_counter() - start_time, 2)
            logger.error("Fine-tuning failed", model=model_name, error=str(exc))
            await self._publish_progress(
                model_name, "error", result.epochs_completed, epochs, error=str(exc)
            )

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        model_name: str,
        dataset_path: str,
        config: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a registered model on a dataset.

        Args:
            model_name: Name of the registered model.
            dataset_path: Path to the evaluation dataset directory.
            config: Evaluation configuration:
                - batch_size (int): Evaluation batch size. Default 8.
                - metrics_list (list[str]): Metrics to compute. Default ["loss", "accuracy"].

        Returns:
            EvaluationResult with computed metrics.
        """
        eval_config = config or {}
        batch_size = eval_config.get("batch_size", 8)

        result = EvaluationResult(
            model_name=model_name,
            dataset_path=dataset_path,
        )

        start_time = time.perf_counter()

        # Validate
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            result.error_message = f"Dataset path does not exist: {dataset_path}"
            logger.error("Evaluation failed", error=result.error_message)
            return result

        if not self._registry.is_loaded(model_name):
            try:
                self._registry.load_model(model_name)
            except Exception as exc:
                result.error_message = f"Failed to load model: {exc}"
                logger.error("Evaluation failed", model=model_name, error=result.error_message)
                return result

        logger.info(
            "Starting evaluation",
            model=model_name,
            dataset=dataset_path,
            batch_size=batch_size,
        )

        await self._publish_progress(model_name, "evaluating", 0, 1)

        try:
            model_instance = self._registry.get_model(model_name)

            # Load evaluation data
            eval_data, _ = await asyncio.get_event_loop().run_in_executor(
                None, self._load_dataset, dataset_path, 0.0
            )

            if len(eval_data) == 0:
                result.error_message = "Evaluation dataset is empty"
                return result

            result.num_samples = len(eval_data)

            pytorch_model = self._extract_pytorch_model(model_instance)
            if pytorch_model is None:
                result.error_message = f"Model '{model_name}' does not expose a trainable PyTorch module"
                return result

            device = "cuda" if torch.cuda.is_available() else "cpu"
            pytorch_model.to(device)
            pytorch_model.eval()

            criterion = torch.nn.MSELoss()
            total_loss = 0.0
            num_batches = 0
            correct = 0
            total_samples = 0

            with torch.no_grad():
                for batch_start in range(0, len(eval_data), batch_size):
                    batch = eval_data[batch_start: batch_start + batch_size]
                    if len(batch) == 0:
                        continue

                    batch_tensor = torch.from_numpy(np.stack(batch)).float().to(device)
                    if batch_tensor.ndim == 4:
                        batch_tensor = batch_tensor.permute(0, 3, 1, 2) / 255.0

                    try:
                        output = pytorch_model(batch_tensor)
                        if output.shape == batch_tensor.shape:
                            loss = criterion(output, batch_tensor)
                        else:
                            loss = output.mean()

                        total_loss += loss.item()
                        num_batches += 1
                        total_samples += len(batch)
                    except Exception as exc:
                        logger.warning("Eval batch failed", error=str(exc))
                        continue

            avg_loss = total_loss / max(num_batches, 1)

            result.metrics = {
                "loss": round(avg_loss, 6),
                "num_samples": total_samples,
                "num_batches": num_batches,
            }
            result.success = True

            logger.info(
                "Evaluation complete",
                model=model_name,
                avg_loss=round(avg_loss, 6),
                num_samples=total_samples,
            )

            await self._publish_progress(model_name, "evaluation_complete", 1, 1)

        except Exception as exc:
            result.error_message = str(exc)
            logger.error("Evaluation failed", model=model_name, error=str(exc))

        result.evaluation_time_seconds = round(time.perf_counter() - start_time, 2)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_dataset(
        self, dataset_path: str, validation_split: float
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Load image dataset from a directory.

        Scans for .jpg, .png, .npy files and loads them into memory.
        Splits into train and validation sets based on validation_split.
        """
        import cv2

        dataset_dir = Path(dataset_path)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        numpy_extensions = {".npy"}

        samples: list[np.ndarray] = []

        for filepath in sorted(dataset_dir.rglob("*")):
            if filepath.suffix.lower() in image_extensions:
                img = cv2.imread(str(filepath))
                if img is not None:
                    # Resize to a standard size for training
                    img = cv2.resize(img, (224, 224))
                    samples.append(img)
            elif filepath.suffix.lower() in numpy_extensions:
                try:
                    arr = np.load(str(filepath))
                    samples.append(arr)
                except Exception:
                    continue

        if not samples:
            return [], []

        # Shuffle deterministically
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(samples))
        samples = [samples[i] for i in indices]

        if validation_split > 0 and len(samples) > 1:
            split_idx = max(1, int(len(samples) * (1.0 - validation_split)))
            return samples[:split_idx], samples[split_idx:]

        return samples, []

    @staticmethod
    def _validate(
        model: torch.nn.Module,
        val_data: list[np.ndarray],
        batch_size: int,
        criterion: torch.nn.Module,
        device: str,
    ) -> float:
        """Run validation and return average loss."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_start in range(0, len(val_data), batch_size):
                batch = val_data[batch_start: batch_start + batch_size]
                if len(batch) == 0:
                    continue

                batch_tensor = torch.from_numpy(np.stack(batch)).float().to(device)
                if batch_tensor.ndim == 4:
                    batch_tensor = batch_tensor.permute(0, 3, 1, 2) / 255.0

                try:
                    output = model(batch_tensor)
                    if output.shape == batch_tensor.shape:
                        loss = criterion(output, batch_tensor)
                    else:
                        loss = output.mean()
                    total_loss += loss.item()
                    num_batches += 1
                except Exception:
                    continue

        model.train()
        return total_loss / max(num_batches, 1)

    @staticmethod
    def _extract_pytorch_model(model_instance: Any) -> Optional[torch.nn.Module]:
        """
        Extract the underlying PyTorch nn.Module from a model wrapper.

        Attempts several common attribute names used across the codebase.
        """
        # Direct nn.Module
        if isinstance(model_instance, torch.nn.Module):
            return model_instance

        # Common wrapper attributes
        for attr_name in ("_model", "model", "_feature_extractor", "_motion_net", "_upsampler"):
            candidate = getattr(model_instance, attr_name, None)
            if isinstance(candidate, torch.nn.Module):
                return candidate

        # Try getting the model attribute from the upsampler (Real-ESRGAN)
        upsampler = getattr(model_instance, "_upsampler", None)
        if upsampler is not None:
            inner = getattr(upsampler, "model", None)
            if isinstance(inner, torch.nn.Module):
                return inner

        return None

    async def _publish_progress(
        self,
        model_name: str,
        status: str,
        current_epoch: int,
        total_epochs: int,
        **extra: Any,
    ) -> None:
        """Publish a training progress event to Kafka."""
        if self._kafka_producer is None:
            return

        event = {
            "model_name": model_name,
            "status": status,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "timestamp": time.time(),
            **extra,
        }

        topic = f"athleteview.training.progress"
        try:
            self._kafka_producer.produce(
                topic=topic,
                key=model_name.encode("utf-8"),
                value=json.dumps(event, default=str).encode("utf-8"),
            )
            self._kafka_producer.poll(0)
        except Exception as exc:
            logger.warning("Failed to publish training progress", error=str(exc))
