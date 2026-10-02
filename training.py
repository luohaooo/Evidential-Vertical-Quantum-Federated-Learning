"""Shared deterministic training utilities."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


METRIC_KEYS = (
    "train_losses",
    "train_acc",
    "val_losses",
    "val_acc",
    "test_losses",
    "test_acc",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _epoch(
    loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    description: str,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=description, leave=False)
        for features, labels in progress:
            features = features.to(device)
            labels = labels.to(device)
            predictions = model(features)
            loss = loss_fn(predictions, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (predictions.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
            progress.set_postfix(loss=total_loss / total_samples)

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> dict[str, list[float]]:
    loss_fn = nn.CrossEntropyLoss()
    history = {key: [] for key in METRIC_KEYS}
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _epoch(
            train_loader,
            model,
            loss_fn,
            device,
            optimizer,
            f"epoch {epoch}/{epochs} train",
        )
        validation_loss, validation_accuracy = _epoch(
            validation_loader,
            model,
            loss_fn,
            device,
            None,
            f"epoch {epoch}/{epochs} validation",
        )
        test_loss, test_accuracy = _epoch(
            test_loader,
            model,
            loss_fn,
            device,
            None,
            f"epoch {epoch}/{epochs} test",
        )
        history["train_losses"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_losses"].append(validation_loss)
        history["val_acc"].append(validation_accuracy)
        history["test_losses"].append(test_loss)
        history["test_acc"].append(test_accuracy)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_acc={validation_accuracy:.4f} test_acc={test_accuracy:.4f}"
        )
    return history


def average_histories(
    histories: Iterable[dict[str, list[float]]],
) -> dict[str, list[float]]:
    histories = list(histories)
    if not histories:
        raise ValueError("At least one history is required")
    lengths = {len(history["train_losses"]) for history in histories}
    if len(lengths) != 1:
        raise ValueError("All histories must contain the same number of epochs")
    return {
        key: np.asarray([history[key] for history in histories], dtype=float)
        .mean(axis=0)
        .tolist()
        for key in METRIC_KEYS
    }

