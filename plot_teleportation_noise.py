#!/usr/bin/env python3
"""Recreate the MNIST-36 teleportation-noise figure included in the paper."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from run_noise_sweep import DEFAULT_RATES, rate_tag


def configure_fonts() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Times New Roman" if "Times New Roman" in installed else "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(input_root: Path, rates: list[float]) -> list[dict]:
    rows = []
    for rate in rates:
        path = input_root / f"epsilon_{rate_tag(rate)}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing noise summary: {path}. Run run_noise_sweep.py first."
            )
        row = json.loads(path.read_text(encoding="utf-8"))
        if "final_test_loss" not in row:
            legacy_path = path.with_suffix(".pkl")
            if not legacy_path.is_file():
                raise KeyError(
                    f"{path} has no final_test_loss and no legacy summary.pkl"
                )
            with legacy_path.open("rb") as stream:
                legacy = pickle.load(stream)
            runs = int(legacy["completed_iters"])
            loss_std = float(legacy["test_losses_std"][-1])
            row["final_test_loss"] = {
                "mean": float(legacy["test_losses"][-1]),
                "std": loss_std,
                "ci95_half_width": 1.96 * loss_std / math.sqrt(runs),
            }
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/noise_sweep"))
    parser.add_argument(
        "--output", type=Path, default=Path("figures/teleportation_noise_mnist36.pdf")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_fonts()
    rates = list(args.rates)
    rows = load_rows(args.input_root, rates)
    accuracy = 100.0 * np.asarray(
        [row["final_test_accuracy"]["mean"] for row in rows]
    )
    accuracy_error = 100.0 * np.asarray(
        [row["final_test_accuracy"]["ci95_half_width"] for row in rows]
    )
    loss = np.asarray([row["final_test_loss"]["mean"] for row in rows])
    loss_error = np.asarray(
        [row["final_test_loss"]["ci95_half_width"] for row in rows]
    )
    positions = np.arange(len(rates))
    labels = [f"{rate:g}" for rate in rates]

    figure, (full_axis, detail_axis) = plt.subplots(1, 2, figsize=(8.4, 3.35))
    full_axis.errorbar(
        positions,
        accuracy,
        yerr=accuracy_error,
        color="#2878b5",
        marker="o",
        markersize=3.5,
        linewidth=1.3,
        capsize=2,
        label="eviQVFL",
    )
    full_axis.axhline(50.0, color="black", linestyle="--", linewidth=0.9, label="Random guess")
    full_axis.set_title("(a) Full range")
    full_axis.set_ylabel("Final test accuracy (%)")
    full_axis.set_xlabel(r"Pauli noise strength $\epsilon$")
    full_axis.set_xticks(positions, labels, rotation=35, ha="right")
    full_axis.set_ylim(48.0, 101.0)
    full_axis.legend(loc="lower left", frameon=True)

    detail_indices = [index for index, rate in enumerate(rates) if rate <= 0.7]
    detail_positions = np.arange(len(detail_indices))
    detail_labels = [labels[index] for index in detail_indices]
    detail_axis.errorbar(
        detail_positions,
        accuracy[detail_indices],
        yerr=accuracy_error[detail_indices],
        color="#2878b5",
        marker="o",
        markersize=3.5,
        linewidth=1.3,
        capsize=2,
        label="Accuracy",
    )
    detail_axis.set_title(r"(b) $\epsilon \leq 0.7$")
    detail_axis.set_xlabel(r"Pauli noise strength $\epsilon$")
    detail_axis.set_ylabel("Final test accuracy (%)")
    detail_axis.set_xticks(detail_positions, detail_labels, rotation=35, ha="right")

    loss_axis = detail_axis.twinx()
    loss_axis.errorbar(
        detail_positions,
        loss[detail_indices],
        yerr=loss_error[detail_indices],
        color="#d9534f",
        marker="s",
        markersize=3.2,
        linestyle="--",
        linewidth=1.2,
        capsize=2,
        label="Loss",
    )
    loss_axis.set_ylabel("Final test loss")
    accuracy_handles, accuracy_labels = detail_axis.get_legend_handles_labels()
    loss_handles, loss_labels = loss_axis.get_legend_handles_labels()
    detail_axis.legend(
        accuracy_handles + loss_handles,
        accuracy_labels + loss_labels,
        loc="center left",
        frameon=True,
    )

    for axis in (full_axis, detail_axis, loss_axis):
        axis.tick_params(colors="black")
        axis.yaxis.label.set_color("black")
    figure.tight_layout(w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
