#!/usr/bin/env python3
"""Train a model/task pair that appears in the paper figures."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import Subset

from data_utils import build_dataset, make_loaders
from models import build_model
from paper_config import PAPER_TASKS, RESULT_FILENAMES, models_for_task
from training import average_histories, set_seed, train_model


def atomic_pickle(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream)
    temporary.replace(destination)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def run_single(
    task_name: str,
    model_name: str,
    run_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    data_root: Path,
    breast_csv: Path | None,
    credit_csv: Path | None,
    num_workers: int,
    max_samples: int | None,
    teleport_error_rate: float = 0.0,
) -> dict[str, list[float]]:
    set_seed(run_seed)
    dataset = build_dataset(task_name, data_root, breast_csv, credit_csv)
    if max_samples is not None and max_samples < len(dataset):
        indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(42)
        )[:max_samples]
        dataset = Subset(dataset, indices.tolist())
    train_loader, validation_loader, test_loader = make_loaders(
        dataset,
        batch_size,
        split_seed=42,
        loader_seed=run_seed,
        num_workers=num_workers,
    )
    model = build_model(
        task_name,
        model_name,
        teleport_error_rate=teleport_error_rate,
        random_seed=run_seed,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return train_model(
        model,
        train_loader,
        validation_loader,
        test_loader,
        optimizer,
        epochs,
        device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=PAPER_TASKS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--breast-csv", type=Path)
    parser.add_argument("--credit-csv", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--teleport-error-rate", type=float, default=0.0)
    args = parser.parse_args()
    if args.model not in models_for_task(args.task):
        parser.error(
            f"--model for {args.task} must be one of: "
            + ", ".join(models_for_task(args.task))
        )
    if args.runs < 1 or args.num_workers < 0:
        parser.error("--runs must be positive and --num-workers cannot be negative")
    if args.max_samples is not None and args.max_samples < 10:
        parser.error("--max-samples must be at least 10")
    if not 0.0 <= args.teleport_error_rate <= 1.0:
        parser.error("--teleport-error-rate must be in [0, 1]")
    if args.teleport_error_rate and args.model != "eviqvfl":
        parser.error("teleportation noise is implemented only for eviqvfl")
    return args


def main() -> None:
    args = parse_args()
    config = PAPER_TASKS[args.task]
    epochs = args.epochs or config.epochs
    batch_size = args.batch_size or config.batch_size
    learning_rate = args.learning_rate or config.learning_rates[args.model]
    device = resolve_device(args.device)
    destination = args.output_root / args.task
    run_root = destination / "runs" / args.model
    histories = []
    for run_index in range(args.runs):
        seed = args.seed + run_index
        print(
            f"task={args.task} model={args.model} run={run_index + 1}/{args.runs} "
            f"seed={seed} device={device}"
        )
        history = run_single(
            args.task,
            args.model,
            seed,
            epochs,
            batch_size,
            learning_rate,
            device,
            args.data_root,
            args.breast_csv,
            args.credit_csv,
            args.num_workers,
            args.max_samples,
            args.teleport_error_rate,
        )
        histories.append(history)
        atomic_pickle(history, run_root / f"run_{run_index + 1:03d}.pkl")

    aggregate = average_histories(histories)
    output_path = destination / RESULT_FILENAMES[args.model]
    atomic_pickle(aggregate, output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()

