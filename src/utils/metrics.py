from __future__ import annotations

import torch


class BinarySegmentationMetrics:
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-7) -> None:
        self.threshold = threshold
        self.smooth = smooth
        self.reset()

    def reset(self) -> None:
        self.intersection = 0.0
        self.prediction_sum = 0.0
        self.target_sum = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = (torch.sigmoid(logits) >= self.threshold).float()
        targets = targets.float()
        self.intersection += float((predictions * targets).sum().item())
        self.prediction_sum += float(predictions.sum().item())
        self.target_sum += float(targets.sum().item())

    def compute(self) -> dict[str, float]:
        union = self.prediction_sum + self.target_sum - self.intersection
        return {
            "dice": (2.0 * self.intersection + self.smooth) / (self.prediction_sum + self.target_sum + self.smooth),
            "iou": (self.intersection + self.smooth) / (union + self.smooth),
        }
