#!/usr/bin/env python3
"""Run the MNIST-36 teleportation-noise experiment reported in Section 4.6."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np

from paper_config import PAPER_TASKS
from run_experiment import atomic_pickle, resolve_device, run_single


DEFAULT_RATES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75)


def rate_tag(rate: float) -> str:
    return format(rate, ".6g").replace(".", "p")


def atomic_json(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    temporary.replace(destination)


def confidence_interval(values: np.ndarray) -> dict[str, float | list[float]]:
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half_width = 1.96 * standard_deviation / math.sqrt(len(values))
    return {
        "mean": float(values.mean()),
        "std": standard_deviation,
        "ci95_half_width": float(half_width),
        "values": values.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/noise_sweep"))
    args = parser.parse_args()
    if args.runs < 1 or args.epochs < 1:
        parser.error("--runs and --epochs must be positive")
    if any(rate < 0.0 or rate > 1.0 for rate in args.rates):
        parser.error("all noise rates must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    config = PAPER_TASKS["mnist36"]
    for rate in args.rates:
        rate_root = args.output_root / f"epsilon_{rate_tag(rate)}"
        histories = []
        for run_index in range(args.runs):
            run_path = rate_root / "runs" / f"run_{run_index + 1:03d}.pkl"
            if run_path.is_file():
                with run_path.open("rb") as stream:
                    history = pickle.load(stream)
                print(f"reuse={run_path}")
            else:
                seed = args.seed + run_index
                print(
                    f"epsilon={rate:g} run={run_index + 1}/{args.runs} "
                    f"seed={seed} device={device}"
                )
                history = run_single(
                    "mnist36",
                    "eviqvfl",
                    seed,
                    args.epochs,
                    args.batch_size,
                    args.learning_rate,
                    device,
                    args.data_root,
                    None,
                    None,
                    args.num_workers,
                    args.max_samples,
                    teleport_error_rate=rate,
                )
                atomic_pickle(history, run_path)
            histories.append(history)

        final_accuracy = np.asarray(
            [history["test_acc"][-1] for history in histories], dtype=float
        )
        final_loss = np.asarray(
            [history["test_losses"][-1] for history in histories], dtype=float
        )
        summary = {
            "task": "mnist36",
            "model": "eviqvfl",
            "classes": list(config.classes),
            "error_rate": rate,
            "effective_bit_flip_rate": 2.0 * rate / 3.0,
            "runs": args.runs,
            "epochs": args.epochs,
            "final_test_accuracy": confidence_interval(final_accuracy),
            "final_test_loss": confidence_interval(final_loss),
        }
        atomic_json(summary, rate_root / "summary.json")
        atomic_pickle(summary, rate_root / "summary.pkl")
        print(f"saved={rate_root / 'summary.json'}")


if __name__ == "__main__":
    main()

