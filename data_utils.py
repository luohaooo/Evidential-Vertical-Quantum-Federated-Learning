"""Datasets and vertical partitions used by the paper experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from paper_config import PAPER_TASKS


def split_image_grid(image: torch.Tensor) -> torch.Tensor:
    """Split a 28 x 28 image into four non-overlapping 14 x 14 parties."""
    _, height, width = image.shape
    half_height, half_width = height // 2, width // 2
    return torch.stack(
        [
            image[:, :half_height, :half_width].squeeze(0),
            image[:, :half_height, half_width:].squeeze(0),
            image[:, half_height:, :half_width].squeeze(0),
            image[:, half_height:, half_width:].squeeze(0),
        ]
    )


class VerticalImageDataset(Dataset):
    """Filtered MNIST/Fashion-MNIST with one image quadrant per party."""

    def __init__(
        self,
        dataset_name: str,
        root: str | Path,
        classes: tuple[int, ...],
        train: bool = True,
    ):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        dataset_cls = {
            "mnist": datasets.MNIST,
            "fashion-mnist": datasets.FashionMNIST,
        }.get(dataset_name)
        if dataset_cls is None:
            raise ValueError(f"Unsupported image dataset: {dataset_name}")
        self.dataset = dataset_cls(
            root=str(root), train=train, download=True, transform=transform
        )
        targets = self.dataset.targets
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for label in classes:
            mask |= targets == label
        self.indices = torch.where(mask)[0].tolist()
        self.label_map = {label: index for index, label in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[self.indices[index]]
        return split_image_grid(image), self.label_map[int(label)]


class BreastCancerDataset(Dataset):
    """Wisconsin Diagnostic Breast Cancer data partitioned into three parties."""

    def __init__(self, csv_path: str | Path):
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Breast-cancer CSV not found: {path}. See README.md for setup."
            )
        frame = pd.read_csv(path).drop(columns=["id"])
        labels = frame["diagnosis"].map({"B": 0, "M": 1})
        if labels.isna().any():
            raise ValueError("diagnosis must contain only B and M")
        feature_columns = [
            column
            for column in frame.columns
            if column not in ("diagnosis", "label")
        ]
        values = frame[feature_columns].to_numpy(dtype=np.float32)
        if values.shape[1] != 30:
            raise ValueError(f"Expected 30 breast-cancer features, got {values.shape[1]}")
        values = _minmax_to_minus_one_one(values)
        self.features = torch.from_numpy(values.reshape(-1, 3, 10))
        self.labels = torch.from_numpy(labels.to_numpy(dtype=np.int64))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class CreditFraudDataset(Dataset):
    """Balanced credit-card subset partitioned into four seven-feature parties."""

    def __init__(self, csv_path: str | Path, samples_per_class: int = 400):
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Credit-card CSV not found: {path}. See README.md for setup."
            )
        frame = pd.read_csv(path)
        negative = frame[frame["Class"] == 0].iloc[:samples_per_class]
        positive = frame[frame["Class"] == 1].iloc[:samples_per_class]
        if len(negative) < samples_per_class or len(positive) < samples_per_class:
            raise ValueError("The CSV does not contain enough samples from both classes")
        frame = pd.concat([negative, positive], ignore_index=True)
        columns = [f"V{index}" for index in range(1, 29)]
        values = frame[columns].to_numpy(dtype=np.float32)
        values = _minmax_to_minus_one_one(values)
        self.features = torch.from_numpy(values.reshape(-1, 4, 7))
        self.labels = torch.from_numpy(frame["Class"].to_numpy(dtype=np.int64))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def _minmax_to_minus_one_one(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=0, keepdims=True)
    maximum = values.max(axis=0, keepdims=True)
    denominator = maximum - minimum
    denominator[denominator == 0] = 1.0
    return 2.0 * (values - minimum) / denominator - 1.0


def build_dataset(
    task_name: str,
    data_root: str | Path = "data",
    breast_csv: str | Path | None = None,
    credit_csv: str | Path | None = None,
) -> Dataset:
    config = PAPER_TASKS[task_name]
    data_root = Path(data_root)
    if config.dataset in ("mnist", "fashion-mnist"):
        return VerticalImageDataset(
            config.dataset, data_root, config.classes, train=True
        )
    if config.dataset == "breast":
        return BreastCancerDataset(breast_csv or data_root / "breast.csv")
    if config.dataset == "credit":
        return CreditFraudDataset(credit_csv or data_root / "creditcard.csv")
    raise ValueError(f"Unsupported task: {task_name}")


def make_loaders(
    dataset: Dataset,
    batch_size: int,
    split_seed: int = 42,
    loader_seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation, and test loaders using the paper's 7:1:2 split."""
    total = len(dataset)
    train_size = int(0.7 * total)
    test_size = int(0.2 * total)
    validation_size = total - train_size - test_size
    train_set, test_set, validation_set = random_split(
        dataset,
        [train_size, test_size, validation_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    generator = torch.Generator().manual_seed(loader_seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader, test_loader

