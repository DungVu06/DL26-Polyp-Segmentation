from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import nn


def _flatten_binary(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have equal shape, got {logits.shape} and {targets.shape}")
    probabilities = torch.sigmoid(logits).flatten(1)
    return probabilities, targets.float().flatten(1)


class BCEWithLogitsSegLoss(nn.Module):
    def __init__(self, pos_weight: float | None = None, name: str | None = None) -> None:
        super().__init__()
        del name
        self.register_buffer("pos_weight", None if pos_weight is None else torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return functional.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.pos_weight)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, name: str | None = None) -> None:
        super().__init__()
        del name
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities, targets = _flatten_binary(logits, targets)
        intersection = (probabilities * targets).sum(dim=1)
        score = (2.0 * intersection + self.smooth) / (probabilities.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1.0 - score.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, name: str | None = None) -> None:
        super().__init__()
        del name
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0, name: str | None = None) -> None:
        super().__init__()
        del name
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities, targets = _flatten_binary(logits, targets)
        true_positive = (probabilities * targets).sum(dim=1)
        false_positive = (probabilities * (1.0 - targets)).sum(dim=1)
        false_negative = ((1.0 - probabilities) * targets).sum(dim=1)
        score = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + self.smooth
        )
        return 1.0 - score.mean()


class FocalTverskyLoss(TverskyLoss):
    def __init__(self, gamma: float = 0.75, **kwargs: float) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return super().forward(logits, targets).pow(self.gamma)


class WeightedSumLoss(nn.Module):
    def __init__(self, losses: Mapping[str, nn.Module], weights: Mapping[str, float], name: str | None = None) -> None:
        super().__init__()
        del name
        if set(losses) != set(weights):
            raise ValueError("Each loss must have exactly one corresponding weight.")
        self.losses = nn.ModuleDict(dict(losses))
        self.weights = {key: float(value) for key, value in weights.items()}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sum(self.weights[name] * loss(logits, targets) for name, loss in self.losses.items())
