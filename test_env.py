from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.dataset)
    datamodule.setup()
    images, masks, filenames = next(iter(datamodule.train_dataloader()))
    model = instantiate(cfg.model)
    logits = model(images)
    criterion = instantiate(cfg.loss)
    loss = criterion(logits, masks)

    print(f"Dataset root: {datamodule.dataset_root}")
    print(f"Fixed split: train={len(datamodule.train_dataset)}, val={len(datamodule.val_dataset)}, test={len(datamodule.test_dataset)}")
    print(f"Batch: images={tuple(images.shape)}, masks={tuple(masks.shape)}, first={filenames[0]}")
    print(f"Model output: {tuple(logits.shape)} | {cfg.loss.name} loss={loss.item():.4f}")
    assert logits.shape == masks.shape, "Binary model output must match mask shape."


if __name__ == "__main__":
    main()
