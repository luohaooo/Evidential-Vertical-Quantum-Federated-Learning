# Evidential Quantum Vertical Federated Learning

This repository is the cleaned reference implementation for the manuscript
**Evidential Quantum Vertical Federated Learning**. It contains the model code,
the experiment entry points, and the plotting scripts needed for the results
and experimental figures reported in the paper.

Raw datasets, training logs, checkpoints, intermediate simulation results, and
unused ablations are intentionally excluded from the repository.

## Included paper scope

The release exposes only the six tasks reported in the manuscript:

| Task name | Dataset and classes | Parties |
| --- | --- | ---: |
| `mnist36` | MNIST `{3, 6}` | 4 |
| `mnist2356` | MNIST `{2, 3, 5, 6}` | 4 |
| `fashion19` | Fashion-MNIST `{1, 9}` | 4 |
| `fashion1349` | Fashion-MNIST `{1, 3, 4, 9}` | 4 |
| `breast` | Wisconsin Diagnostic Breast Cancer | 3 |
| `credit` | Credit Card Fraud Detection | 4 |

The retained models match the curves shown in the paper:

- `classical-average`
- `classical-fuse`
- `measure-average`
- `measure-vqc`
- `eviqvfl`
- `teleported-vqc` (MNIST `{3,6}` only)
- `teleported-random-fixed` (MNIST `{3,6}` only)

Unreported tasks such as MNIST `{2,3,6}` and Fashion-MNIST `{1,3,9}`, together
with the exploratory classical-attention baseline and unused gradient-sweep
figures, are not part of this release.

## Repository layout

```text
.
|-- evidence.py                    # evidential fusion and teleportation noise
|-- models.py                      # eviQVFL and paper baselines
|-- vqc.py                         # variational and fixed quantum circuits
|-- data_utils.py                  # vertical datasets and data loaders
|-- training.py                    # shared training/evaluation utilities
|-- paper_config.py                # paper task and model scope
|-- run_experiment.py              # one task/model experiment
|-- run_paper_experiments.py       # all displayed task/model combinations
|-- run_noise_sweep.py             # Section 4.6 noise experiment
|-- plot_paper_results.py          # accuracy/loss figures
|-- plot_teleportation_noise.py    # teleportation-noise figure
|-- tc/                            # tensor-train implementation
|-- tests/                         # lightweight consistency tests
`-- figures/                       # only figures currently used in the paper
```

The `data/`, `outputs/`, `simulation/`, `checkpoints/`, and `logs/` directories
are ignored by Git.

## Environment

The experiments were organized and tested with Python 3.9 and the dependency
versions listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For CUDA execution, install a PyTorch build compatible with the local CUDA
toolkit before installing the remaining requirements if the pinned wheel is
not suitable for the machine.

## Data preparation

No raw data are committed.

- MNIST and Fashion-MNIST are downloaded automatically by `torchvision` into
  the directory passed through `--data-root`.
- Put the Wisconsin breast-cancer CSV at `data/breast.csv`, or pass its path
  with `--breast-csv`. The file must contain `id`, `diagnosis`, and the standard
  30 diagnostic feature columns.
- Put the credit-card fraud CSV at `data/creditcard.csv`, or pass its path with
  `--credit-csv`. The file must contain `Class` and `V1` through `V28`.

## Run one experiment

For example, train eviQVFL on MNIST `{3,6}`:

```bash
python run_experiment.py \
  --task mnist36 \
  --model eviqvfl \
  --runs 50 \
  --device cuda
```

Run a baseline displayed in Fig. 9:

```bash
python run_experiment.py \
  --task mnist36 \
  --model teleported-random-fixed \
  --runs 50 \
  --device cuda
```

For a quick functional check, reduce the workload:

```bash
python run_experiment.py \
  --task mnist36 \
  --model eviqvfl \
  --epochs 1 \
  --runs 1 \
  --max-samples 128 \
  --device cpu
```

The aggregate history is written to `outputs/<task>/` using the filenames used
by the plotting scripts. Per-run histories are stored under
`outputs/<task>/runs/`.

## Run all performance experiments

The following command covers exactly the task/model combinations displayed in
the manuscript:

```bash
python run_paper_experiments.py \
  --runs 50 \
  --device cuda \
  --breast-csv data/breast.csv \
  --credit-csv data/creditcard.csv
```

This is computationally expensive because the quantum baselines simulate up to
16 qubits and the paper averages independent runs.

## Recreate the accuracy and loss figures

After the corresponding result histories have been generated:

```bash
python plot_paper_results.py --input-root outputs --output-root figures
```

The script generates only the twelve performance PDFs used in the paper:
`acc_<task>.pdf` and `loss_<task>.pdf` for the six retained tasks.

## Recreate the teleportation-noise figure

Run the 20-run MNIST `{3,6}` noise sweep:

```bash
python run_noise_sweep.py --runs 20 --epochs 20 --device cuda
```

Then generate the two-panel figure:

```bash
python plot_teleportation_noise.py
```

The modeled symmetric Pauli channel has total error probability `epsilon`. For
the computational-basis evidence masses used by the fusion circuit, it is
equivalent to an independent bit-flip channel with probability `2*epsilon/3`.

## Tests

```bash
python -m unittest discover -s tests
```

The tests check the evidential transform, the teleportation-noise mapping, the
paper task/model scope, and the absence of trainable parameters in the
random-fixed server circuit.

## Reproducibility notes

- The train/validation/test split is fixed at 70%/10%/20% with split seed 42.
- Model initialization and data-loader shuffling use consecutive seeds starting
  from the value supplied through `--seed`.
- The random-fixed server circuit is newly sampled for each independent run and
  remains unchanged during that run.
- Generated data and intermediate results are deliberately excluded from
  version control. The `figures/` directory contains only the framework,
  circuit, and final experimental figures currently used by the manuscript.
