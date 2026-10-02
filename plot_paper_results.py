#!/usr/bin/env python3
"""Recreate the accuracy and loss plots included in the paper."""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

from paper_config import PAPER_TASKS, RESULT_FILENAMES, models_for_task


@dataclass(frozen=True)
class CurveStyle:
    label: str
    color: str
    linestyle: str


STYLES = {
    "classical-average": CurveStyle("classical-average", "tab:blue", "-."),
    "classical-fuse": CurveStyle("classical-fuse", "tab:blue", "-"),
    "measure-average": CurveStyle("measure-average", "tab:red", "-."),
    "measure-vqc": CurveStyle("measure-VQC", "tab:red", "-"),
    "teleported-vqc": CurveStyle("teleported-VQC", "tab:green", ":"),
    "teleported-random-fixed": CurveStyle(
        "teleported-random-fixed", "tab:brown", "--"
    ),
    "eviqvfl": CurveStyle("eviQVFL", "tab:green", "-"),
}


def configure_fonts() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Times New Roman" if "Times New Roman" in installed else "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_histories(task_name: str, input_root: Path) -> dict[str, dict]:
    histories = {}
    for model_name in models_for_task(task_name):
        path = input_root / task_name / RESULT_FILENAMES[model_name]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing result for a paper curve: {path}. Run run_experiment.py first."
            )
        with path.open("rb") as stream:
            histories[model_name] = pickle.load(stream)
    return histories


def plot_metric(
    task_name: str,
    histories: dict[str, dict],
    metric: str,
    output_dir: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(5, 6))
    for model_name in models_for_task(task_name):
        style = STYLES[model_name]
        values = histories[model_name][metric]
        axis.plot(
            range(1, len(values) + 1),
            values,
            color=style.color,
            linestyle=style.linestyle,
            linewidth=2,
            label=style.label,
        )

    if metric == "test_acc":
        all_values = [value for history in histories.values() for value in history[metric]]
        if task_name == "fashion19":
            axis.set_ylim(0.92, 1.0)
        else:
            axis.set_ylim(max(0.0, math.floor(min(all_values) * 10) / 10), 1.0)
        prefix, ylabel, location = "acc", "Accuracy", "lower right"
    else:
        num_classes = len(PAPER_TASKS[task_name].classes)
        lower_bound = math.log(math.e + num_classes - 1) - 1
        axis.axhline(
            lower_bound,
            color="tab:purple",
            linestyle="--",
            linewidth=1.5,
            label="quantum lower bound",
        )
        all_values = [value for history in histories.values() for value in history[metric]]
        axis.set_ylim(0.0, max(0.8, math.ceil(max(all_values) * 10) / 10))
        prefix, ylabel, location = "loss", "Loss", "upper right"

    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.xaxis.set_major_locator(MultipleLocator(10))
    axis.legend(loc=location, frameon=True)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{prefix}_{task_name}.pdf"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=PAPER_TASKS, default=list(PAPER_TASKS))
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-root", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_fonts()
    for task_name in args.tasks:
        histories = load_histories(task_name, args.input_root)
        for metric in ("test_acc", "train_losses"):
            output = plot_metric(task_name, histories, metric, args.output_root)
            print(f"saved={output}")


if __name__ == "__main__":
    main()

