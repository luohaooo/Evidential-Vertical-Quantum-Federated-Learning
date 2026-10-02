#!/usr/bin/env python3
"""Run all task/model combinations displayed in the paper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from paper_config import PAPER_TASKS, models_for_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=PAPER_TASKS, default=list(PAPER_TASKS))
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--breast-csv", type=Path)
    parser.add_argument("--credit-csv", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("run_experiment.py")
    for task_name in args.tasks:
        for model_name in models_for_task(task_name):
            command = [
                sys.executable,
                str(script),
                "--task",
                task_name,
                "--model",
                model_name,
                "--runs",
                str(args.runs),
                "--device",
                args.device,
                "--data-root",
                str(args.data_root),
                "--output-root",
                str(args.output_root),
                "--num-workers",
                str(args.num_workers),
            ]
            if args.breast_csv:
                command.extend(["--breast-csv", str(args.breast_csv)])
            if args.credit_csv:
                command.extend(["--credit-csv", str(args.credit_csv)])
            print("running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

