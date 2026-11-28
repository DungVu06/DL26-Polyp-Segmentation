from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SchedulerHandle:
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None
    step_on: str


def create_reduce_on_plateau(
    optimizer: torch.optim.Optimizer,
    mode: str = "max",
    factor: float = 0.5,
    patience: int = 8,
    threshold: float = 1e-4,
    min_lr: float = 1e-6,
    total_steps: int | None = None,
) -> SchedulerHandle:
    del total_steps
    return SchedulerHandle(
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, min_lr=min_lr
        ),
        step_on="metric",
    )


def create_cosine_annealing(
    optimizer: torch.optim.Optimizer, total_steps: int, eta_min: float = 1e-6
) -> SchedulerHandle:
    return SchedulerHandle(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min), "step")


def create_no_scheduler(optimizer: torch.optim.Optimizer, total_steps: int | None = None) -> SchedulerHandle:
    del optimizer, total_steps
    return SchedulerHandle(None, "none")
