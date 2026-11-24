from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from src.training.schedulers import SchedulerHandle
from src.utils.metrics import BinarySegmentationMetrics


class SegmentationTrainer:
    def __init__(
        self,
        max_epochs: int,
        device: str,
        amp: bool,
        grad_clip_norm: float | None,
        early_stopping_patience: int,
        monitor: str,
        monitor_mode: str,
        checkpoint_dir: str,
        log_every_n_steps: int,
        image_log_every_n_epochs: int,
        logger: Any,
    ) -> None:
        if monitor not in {"dice", "iou"}:
            raise ValueError("monitor must be one of: dice, iou")
        if monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be one of: min, max")
        self.max_epochs = max_epochs
        self.device = self._resolve_device(device)
        self.use_amp = amp and self.device.type == "cuda"
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_every_n_steps = log_every_n_steps
        self.image_log_every_n_epochs = image_log_every_n_epochs
        self.logger = logger

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Set trainer.device=cpu or auto.")
        return torch.device(requested)

    def fit(
        self,
        model: nn.Module,
        datamodule: Any,
        criterion: nn.Module,
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        scheduler_factory: Callable[..., SchedulerHandle],
        config: Mapping[str, Any],
    ) -> dict[str, float | int | str]:
        datamodule.setup()
        model.to(self.device)
        optimizer = optimizer_factory(params=model.parameters())
        scheduler_handle = scheduler_factory(optimizer=optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        best_score = float("inf") if self.monitor_mode == "min" else float("-inf")
        best_epoch, stale_epochs = 0, 0
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / "best.ckpt"
        latest_checkpoint_path = self.checkpoint_dir.parent / "latest.ckpt"

        for epoch in range(1, self.max_epochs + 1):
            train_loader = datamodule.train_dataloader()
            train_metrics, _ = self._run_epoch(model, train_loader, criterion, optimizer, scaler, epoch, "train")
            val_metrics, examples = self._run_epoch(model, datamodule.val_dataloader(), criterion, None, scaler, epoch, "val")
            epoch_metrics = {f"train/{key}": value for key, value in train_metrics.items()}
            epoch_metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
            epoch_metrics["lr"] = optimizer.param_groups[0]["lr"]
            epoch_metrics["epoch"] = epoch
            self.logger.log_metrics(epoch_metrics, step=epoch * len(train_loader))

            self._step_scheduler(scheduler_handle, val_metrics[self.monitor])
            score = val_metrics[self.monitor]
            if self._is_improvement(score, best_score):
                best_score, best_epoch, stale_epochs = score, epoch, 0
                self._save_checkpoint(checkpoint_path, model, optimizer, epoch, best_score, config)
                shutil.copy2(checkpoint_path, latest_checkpoint_path)
                self.logger.log_artifact(checkpoint_path, name="best-model", artifact_type="model")
            else:
                stale_epochs += 1

            print(
                f"Epoch {epoch:03d}/{self.max_epochs} | "
                f"train loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} iou={train_metrics['iou']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"best val_{self.monitor}={best_score:.4f} (epoch {best_epoch}) | "
                f"early-stop={stale_epochs}/{self.early_stopping_patience}"
            )

            if examples is not None and epoch % self.image_log_every_n_epochs == 0:
                self._log_examples(*examples, epoch=epoch)
            if stale_epochs >= self.early_stopping_patience:
                break

        return {"best_epoch": best_epoch, f"best_val_{self.monitor}": best_score, "checkpoint": str(checkpoint_path)}

    def evaluate(
        self,
        model: nn.Module,
        dataloader: Any,
        criterion: nn.Module,
        prediction_dir: str | None = None,
    ) -> dict[str, float]:
        model.to(self.device)
        metrics, _ = self._run_epoch(model, dataloader, criterion, None, None, epoch=0, stage="test", prediction_dir=prediction_dir)
        return metrics

    def _run_epoch(
        self,
        model: nn.Module,
        dataloader: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scaler: torch.cuda.amp.GradScaler | None,
        epoch: int,
        stage: str,
        prediction_dir: str | None = None,
    ) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None]:
        training = optimizer is not None
        model.train(training)
        metrics = BinarySegmentationMetrics()
        total_loss, sample_count = 0.0, 0
        examples = None
        output_dir = Path(prediction_dir) if prediction_dir else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        with torch.set_grad_enabled(training):
            progress = tqdm(dataloader, desc=f"{stage} {epoch}" if epoch else stage, leave=False)
            for batch_index, (images, masks, filenames) in enumerate(progress, start=1):
                images, masks = images.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = model(images)
                    loss = criterion(logits, masks)
                if training:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    if self.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()

                batch_size = images.shape[0]
                total_loss += loss.detach().item() * batch_size
                sample_count += batch_size
                metrics.update(logits.detach(), masks)
                progress.set_postfix(loss=f"{loss.detach().item():.4f}")
                if training and batch_index % self.log_every_n_steps == 0:
                    self.logger.log_metrics({"train/step_loss": loss.detach().item()}, step=(epoch - 1) * len(dataloader) + batch_index)
                if examples is None and not training:
                    examples = self._to_loggable_examples(images, masks, logits, list(filenames))
                if output_dir is not None:
                    self._save_predictions(logits, filenames, output_dir)

        result = {"loss": total_loss / max(sample_count, 1), **metrics.compute()}
        return result, examples

    def _is_improvement(self, score: float, best_score: float) -> bool:
        return score < best_score if self.monitor_mode == "min" else score > best_score

    @staticmethod
    def _step_scheduler(handle: SchedulerHandle, metric: float) -> None:
        if handle.scheduler is None:
            return
        if handle.step_on == "metric":
            handle.scheduler.step(metric)
        else:
            handle.scheduler.step()

    @staticmethod
    def _save_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
        config: Mapping[str, Any],
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "score": score,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": dict(config),
            },
            path,
        )

    @staticmethod
    def _save_predictions(logits: torch.Tensor, filenames: list[str] | tuple[str, ...], output_dir: Path) -> None:
        predictions = (torch.sigmoid(logits) >= 0.5).squeeze(1).byte().cpu().numpy() * 255
        for prediction, filename in zip(predictions, filenames):
            destination = output_dir / f"{Path(filename).stem}.png"
            cv2.imwrite(str(destination), prediction)

    @staticmethod
    def _to_loggable_examples(
        images: torch.Tensor, masks: torch.Tensor, logits: torch.Tensor, filenames: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        count = min(4, images.shape[0])
        image_array = images[:count].detach().cpu().permute(0, 2, 3, 1).numpy()
        image_array = np.clip(image_array * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        mask_array = masks[:count].detach().cpu().squeeze(1).numpy().astype(np.uint8)
        prediction_array = (torch.sigmoid(logits[:count]) >= 0.5).detach().cpu().squeeze(1).numpy().astype(np.uint8)
        return image_array, mask_array, prediction_array, filenames[:count]

    def _log_examples(
        self, images: np.ndarray, masks: np.ndarray, predictions: np.ndarray, captions: list[str], epoch: int
    ) -> None:
        self.logger.log_predictions(images, masks, predictions, captions, step=epoch)
