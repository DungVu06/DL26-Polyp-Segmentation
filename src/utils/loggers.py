from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class NoOpLogger:
    def __init__(self, run_config: Mapping[str, Any] | None = None) -> None:
        del run_config

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        del metrics, step

    def log_predictions(self, images: Any, masks: Any, predictions: Any, captions: Sequence[str], step: int) -> None:
        del images, masks, predictions, captions, step

    def log_artifact(self, path: str | Path, name: str, artifact_type: str) -> None:
        del path, name, artifact_type

    def finish(self) -> None:
        return None


class WandbLogger:
    def __init__(
        self,
        project: str,
        entity: str | None,
        run_name: str,
        tags: Sequence[str],
        mode: str,
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as error:
            raise ImportError("W&B is configured but not installed. Run `pip install wandb`.") from error
        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=list(tags),
            config=dict(run_config or {}),
            mode=mode,
        )

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self.run.log(dict(metrics), step=step)

    def log_predictions(self, images: Any, masks: Any, predictions: Any, captions: Sequence[str], step: int) -> None:
        examples = []
        for image, mask, prediction, caption in zip(images, masks, predictions, captions):
            examples.append(
                self.wandb.Image(
                    image,
                    caption=caption,
                    masks={
                        "ground_truth": {"mask_data": mask, "class_labels": {0: "background", 1: "polyp"}},
                        "prediction": {"mask_data": prediction, "class_labels": {0: "background", 1: "polyp"}},
                    },
                )
            )
        self.run.log({"validation/predictions": examples}, step=step)

    def log_artifact(self, path: str | Path, name: str, artifact_type: str) -> None:
        artifact = self.wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        self.run.finish()
