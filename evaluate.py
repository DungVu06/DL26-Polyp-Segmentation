from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.utils.seed import seed_everything


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.project.seed)
    resolved_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    datamodule = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    criterion = instantiate(cfg.loss)
    logger = instantiate(cfg.logger, run_config=resolved_config, _recursive_=False)
    trainer = instantiate(cfg.trainer, logger=logger)

    checkpoint_path = Path(cfg.evaluation.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    datamodule.setup()
    prediction_dir = cfg.evaluation.prediction_dir if cfg.evaluation.save_predictions else None
    try:
        threshold = 0.5
        if cfg.evaluation.tune_threshold:
            threshold, val_metrics = trainer.tune_threshold(model, datamodule.val_dataloader())
            print(f"Validation threshold: {threshold:.3f} | metrics: {val_metrics}")
            logger.log_metrics({"validation/best_threshold": threshold, **{f"validation/tuned_{key}": value for key, value in val_metrics.items()}})
        result = trainer.evaluate(model, datamodule.test_dataloader(), criterion, prediction_dir, threshold=threshold)
        logger.log_metrics({"test/threshold": threshold, **{f"test/{key}": value for key, value in result.items()}})
        print(f"Test metrics: {result}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
