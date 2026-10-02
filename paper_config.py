"""Experiment scope and hyperparameters used by the paper figures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskConfig:
    dataset: str
    classes: tuple[int, ...]
    num_clients: int
    qubits_per_client: int
    local_vqc_layers: int
    epochs: int
    batch_size: int
    learning_rates: dict[str, float] = field(default_factory=dict)


PAPER_MODEL_NAMES = (
    "classical-average",
    "classical-fuse",
    "measure-average",
    "measure-vqc",
    "teleported-vqc",
    "teleported-random-fixed",
    "eviqvfl",
)

COMMON_IMAGE_LR = {
    "classical-average": 5e-3,
    "classical-fuse": 5e-3,
    "measure-average": 5e-3,
    "measure-vqc": 1e-2,
    "teleported-vqc": 5e-3,
    "teleported-random-fixed": 5e-3,
    "eviqvfl": 5e-3,
}

PAPER_TASKS = {
    "mnist36": TaskConfig("mnist", (3, 6), 4, 4, 2, 20, 64, COMMON_IMAGE_LR),
    "mnist2356": TaskConfig("mnist", (2, 3, 5, 6), 4, 4, 2, 20, 64, COMMON_IMAGE_LR),
    "fashion19": TaskConfig("fashion-mnist", (1, 9), 4, 4, 2, 20, 64, COMMON_IMAGE_LR),
    "fashion1349": TaskConfig("fashion-mnist", (1, 3, 4, 9), 4, 4, 2, 20, 64, COMMON_IMAGE_LR),
    "breast": TaskConfig(
        "breast", (0, 1), 3, 4, 1, 100, 32,
        {
            "classical-average": 2e-2,
            "classical-fuse": 5e-3,
            "measure-average": 1e-2,
            "measure-vqc": 1e-2,
            "eviqvfl": 1e-2,
        },
    ),
    "credit": TaskConfig(
        "credit", (0, 1), 4, 3, 1, 100, 64,
        {
            "classical-average": 5e-3,
            "classical-fuse": 5e-3,
            "measure-average": 1e-2,
            "measure-vqc": 5e-3,
            "eviqvfl": 5e-3,
        },
    ),
}


def models_for_task(task_name: str) -> tuple[str, ...]:
    """Return exactly the models displayed for a paper task."""
    if task_name not in PAPER_TASKS:
        raise KeyError(f"Unknown paper task: {task_name}")
    if task_name == "mnist36":
        return PAPER_MODEL_NAMES
    return (
        "classical-average",
        "classical-fuse",
        "measure-average",
        "measure-vqc",
        "eviqvfl",
    )


RESULT_FILENAMES = {
    "classical-average": "VFL_classical_aggregate.pkl",
    "classical-fuse": "VFL_classical_fuse.pkl",
    "measure-average": "VFL_avg.pkl",
    "measure-vqc": "Measure_then_encode.pkl",
    "teleported-vqc": "QVFL_overall.pkl",
    "teleported-random-fixed": "random_fixed.pkl",
    "eviqvfl": "QVFL.pkl",
}

