from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        filenames: Sequence[str],
        transform: Callable | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.filenames = list(filenames)
        self.transform = transform
        self._validate_files()

    def _validate_files(self) -> None:
        missing = [
            name
            for name in self.filenames
            if not (self.image_dir / name).is_file() or not (self.mask_dir / name).is_file()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing {len(missing)} Kvasir image/mask pair(s), e.g. {preview}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.filenames[index]
        image = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.mask_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read image/mask pair: {filename}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.float32)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return image.float(), mask.float(), filename


class KvasirSegDataModule:
    required_splits = ("train", "val", "test")

    def __init__(
        self,
        root_dir: str | None,
        split_file: str,
        kaggle_dataset: str,
        download: bool,
        image_size: Sequence[int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        train_transform: Callable,
        eval_transform: Callable,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser() if root_dir else None
        self.split_file = Path(split_file).expanduser()
        self.kaggle_dataset = kaggle_dataset
        self.download = download
        self.image_size = tuple(int(v) for v in image_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_transform_factory = train_transform
        self.eval_transform_factory = eval_transform
        self.dataset_root: Path | None = None
        self.splits: dict[str, list[str]] | None = None

    def prepare_data(self) -> None:
        if self.dataset_root is not None:
            return
        candidate = self.root_dir
        if candidate is None:
            if not self.download:
                raise ValueError("dataset.root_dir is null and dataset.download is false")
            try:
                import kagglehub
            except ImportError as error:
                raise ImportError("Install kagglehub or pass dataset.root_dir=<local path>.") from error
            candidate = Path(kagglehub.dataset_download(self.kaggle_dataset))

        self.dataset_root = self._find_kvasir_root(candidate)
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

    @staticmethod
    def _find_kvasir_root(candidate: Path) -> Path:
        candidates = (candidate, candidate / "Kvasir-SEG")
        for path in candidates:
            if (path / "images").is_dir() and (path / "masks").is_dir():
                return path
        raise FileNotFoundError(
            f"Could not find images/ and masks/ under {candidate}. "
            "Pass dataset.root_dir to the directory containing Kvasir-SEG."
        )

    def setup(self) -> None:
        self.prepare_data()
        if self.splits is not None:
            return
        with self.split_file.open(encoding="utf-8") as handle:
            raw_splits = json.load(handle)
        if set(raw_splits) != set(self.required_splits):
            raise ValueError(f"Split JSON must contain exactly {self.required_splits}, got {tuple(raw_splits)}")
        self.splits = {name: list(raw_splits[name]) for name in self.required_splits}
        if len(set().union(*map(set, self.splits.values()))) != sum(map(len, self.splits.values())):
            raise ValueError("Train/val/test splits must not overlap.")

        assert self.dataset_root is not None
        image_dir, mask_dir = self.dataset_root / "images", self.dataset_root / "masks"
        self.train_dataset = KvasirSegDataset(
            image_dir, mask_dir, self.splits["train"], self.train_transform_factory(self.image_size)
        )
        self.val_dataset = KvasirSegDataset(
            image_dir, mask_dir, self.splits["val"], self.eval_transform_factory(self.image_size)
        )
        self.test_dataset = KvasirSegDataset(
            image_dir, mask_dir, self.splits["test"], self.eval_transform_factory(self.image_size)
        )

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        self.setup()
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        self.setup()
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        self.setup()
        return self._loader(self.test_dataset, shuffle=False)
