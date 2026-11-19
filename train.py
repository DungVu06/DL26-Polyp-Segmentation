from __future__ import annotations

import hydra
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
    optimizer_factory = instantiate(cfg.optimizer)
    scheduler_factory = instantiate(cfg.scheduler)
    logger = instantiate(cfg.logger, config=resolved_config)
    trainer = instantiate(cfg.trainer, logger=logger)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Run: {cfg.project.run_name} | device: {trainer.device} | parameters: {parameter_count:,}")
    try:
        result = trainer.fit(model, datamodule, criterion, optimizer_factory, scheduler_factory, resolved_config)
        print(f"Training complete: {result}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
