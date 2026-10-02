"""Models and baselines that appear in the paper figures."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
import torchquantum as tq

from evidence import CCRServer, evidence2prob
from paper_config import PAPER_TASKS, models_for_task
from tc.tc_fc import TTLinear
from vqc import Overall_VQC, RandomFixedOverall_VQC, VQC


class ImageFeatureReducer(nn.Module):
    def __init__(self, num_qubits: int):
        super().__init__()
        factor = 2 if num_qubits % 2 == 0 else 1
        output_modes = [1, factor, num_qubits // factor, 1]
        self.flatten = nn.Flatten()
        self.reducer = TTLinear(
            [2, 7, 7, 2], output_modes, tt_rank=[1, 2, 2, 2, 1]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.reducer(self.flatten(inputs))


class BreastFeatureReducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.reducer = TTLinear([2, 5], [2, 2], tt_rank=[1, 2, 1])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.reducer(self.flatten(inputs))


class CreditFeatureReducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.reducer = nn.Linear(7, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.reducer(self.flatten(inputs))


class QuantumParty(nn.Module):
    def __init__(
        self,
        feature_reducer: nn.Module,
        num_qubits: int,
        num_layers: int,
        num_classes: int,
    ):
        super().__init__()
        self.feature_reducer = feature_reducer
        self.num_qubits = num_qubits
        self.vqc = VQC(
            n_wires=num_qubits, n_qlayers=num_layers, n_types=num_classes
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.sigmoid(self.feature_reducer(inputs))
        device = tq.QuantumDevice(
            n_wires=self.num_qubits,
            bsz=features.shape[0],
            device=features.device,
        )
        return self.vqc(features, device)


class ImageClassicalParty(nn.Module):
    def __init__(self, num_qubits: int, num_classes: int):
        super().__init__()
        self.feature_reducer = ImageFeatureReducer(num_qubits)
        self.hidden = nn.Linear(num_qubits, num_qubits)
        self.output = nn.Linear(num_qubits, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.sigmoid(self.feature_reducer(inputs))
        return self.output(torch.relu(self.hidden(features)))


class BreastClassicalParty(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.feature_reducer = BreastFeatureReducer()
        self.hidden = nn.Linear(4, 2)
        self.output = nn.Linear(2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.sigmoid(self.feature_reducer(inputs))
        return self.output(torch.relu(self.hidden(features)))


class CreditClassicalParty(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.reducer = nn.Linear(7, 3)
        self.hidden = nn.Linear(3, 2)
        self.output = nn.Linear(2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.reducer(torch.sigmoid(self.flatten(inputs)))
        return self.output(torch.relu(self.hidden(features)))


class ClassicalAverage(nn.Module):
    def __init__(self, parties: list[nn.Module]):
        super().__init__()
        self.parties = nn.ModuleList(parties)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = [party(inputs[:, index]) for index, party in enumerate(self.parties)]
        return torch.stack(logits, dim=1).mean(dim=1)


class ClassicalFuse(nn.Module):
    def __init__(self, parties: list[nn.Module], num_classes: int):
        super().__init__()
        self.parties = nn.ModuleList(parties)
        self.fusion = nn.Linear(len(parties) * num_classes, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = [party(inputs[:, index]) for index, party in enumerate(self.parties)]
        return self.fusion(torch.cat(logits, dim=1))


def masses_to_probabilities(masses: torch.Tensor, num_classes: int) -> torch.Tensor:
    transform = evidence2prob(num_classes).to(masses.device)
    probabilities = transform.matmul(masses.t()).t()
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)


class EviQVFL(nn.Module):
    def __init__(
        self,
        parties: list[nn.Module],
        teleport_error_rate: float = 0.0,
    ):
        super().__init__()
        self.parties = nn.ModuleList(parties)
        self.server = CCRServer(len(parties), teleport_error_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        masses = [party(inputs[:, index]) for index, party in enumerate(self.parties)]
        return self.server(torch.stack(masses, dim=1))


class MeasureAverage(nn.Module):
    def __init__(self, parties: list[nn.Module], num_classes: int):
        super().__init__()
        self.parties = nn.ModuleList(parties)
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = [
            masses_to_probabilities(party(inputs[:, index]), self.num_classes)
            for index, party in enumerate(self.parties)
        ]
        return torch.stack(probabilities, dim=1).mean(dim=1)


class MeasureVQC(nn.Module):
    def __init__(
        self,
        parties: list[nn.Module],
        num_classes: int,
        server_layers: int = 2,
    ):
        super().__init__()
        self.parties = nn.ModuleList(parties)
        self.num_classes = num_classes
        self.server_wires = len(parties) * num_classes
        self.server_vqc = VQC(
            n_wires=self.server_wires,
            n_qlayers=server_layers,
            n_types=num_classes,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = [
            masses_to_probabilities(party(inputs[:, index]), self.num_classes)
            for index, party in enumerate(self.parties)
        ]
        encoded = torch.sigmoid(torch.cat(probabilities, dim=1))
        device = tq.QuantumDevice(
            n_wires=self.server_wires,
            bsz=encoded.shape[0],
            device=encoded.device,
        )
        masses = self.server_vqc(encoded, device)
        return masses_to_probabilities(masses, self.num_classes)


class TeleportedVQC(nn.Module):
    """Trainable server-side VQC baseline used only in the MNIST-36 figure."""

    def __init__(
        self,
        feature_reducers: list[nn.Module],
        qubits_per_client: int,
        local_layers: int,
        num_classes: int,
        server_layers: int = 1,
    ):
        super().__init__()
        self.feature_reducers = nn.ModuleList(feature_reducers)
        self.qubits_per_client = qubits_per_client
        self.num_classes = num_classes
        self.total_qubits = len(feature_reducers) * qubits_per_client
        self.vqc = Overall_VQC(
            n_clients=len(feature_reducers),
            n_qubit_client=qubits_per_client,
            n_layers=local_layers,
            n_types=num_classes,
            n_global_layers=server_layers,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = [
            reducer(inputs[:, index])
            for index, reducer in enumerate(self.feature_reducers)
        ]
        encoded = torch.sigmoid(torch.cat(features, dim=1))
        device = tq.QuantumDevice(
            n_wires=self.total_qubits,
            bsz=encoded.shape[0],
            device=encoded.device,
        )
        masses = self.vqc(encoded, device)
        return masses_to_probabilities(masses, self.num_classes)


class TeleportedRandomFixed(nn.Module):
    """Random, zero-trainable-parameter server circuit used in Fig. 9."""

    def __init__(
        self,
        feature_reducers: list[nn.Module],
        qubits_per_client: int,
        local_layers: int,
        num_classes: int,
        random_seed: int,
        fusion_layers: int = 1,
    ):
        super().__init__()
        self.feature_reducers = nn.ModuleList(feature_reducers)
        self.qubits_per_client = qubits_per_client
        self.num_classes = num_classes
        self.total_qubits = len(feature_reducers) * qubits_per_client
        self.vqc = RandomFixedOverall_VQC(
            n_clients=len(feature_reducers),
            n_qubit_client=qubits_per_client,
            n_local_layers=local_layers,
            n_fusion_layers=fusion_layers,
            n_types=num_classes,
            random_seed=random_seed,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = [
            reducer(inputs[:, index])
            for index, reducer in enumerate(self.feature_reducers)
        ]
        encoded = torch.sigmoid(torch.cat(features, dim=1))
        device = tq.QuantumDevice(
            n_wires=self.total_qubits,
            bsz=encoded.shape[0],
            device=encoded.device,
        )
        masses = self.vqc(encoded, device)
        return masses_to_probabilities(masses, self.num_classes)


def _feature_reducer_factory(task_name: str) -> Callable[[], nn.Module]:
    dataset = PAPER_TASKS[task_name].dataset
    if dataset in ("mnist", "fashion-mnist"):
        qubits = PAPER_TASKS[task_name].qubits_per_client
        return lambda: ImageFeatureReducer(qubits)
    if dataset == "breast":
        return BreastFeatureReducer
    if dataset == "credit":
        return CreditFeatureReducer
    raise ValueError(f"Unsupported task: {task_name}")


def _quantum_parties(task_name: str) -> list[nn.Module]:
    config = PAPER_TASKS[task_name]
    factory = _feature_reducer_factory(task_name)
    return [
        QuantumParty(
            factory(),
            config.qubits_per_client,
            config.local_vqc_layers,
            len(config.classes),
        )
        for _ in range(config.num_clients)
    ]


def _classical_parties(task_name: str) -> list[nn.Module]:
    config = PAPER_TASKS[task_name]
    num_classes = len(config.classes)
    if config.dataset in ("mnist", "fashion-mnist"):
        return [
            ImageClassicalParty(config.qubits_per_client, num_classes)
            for _ in range(config.num_clients)
        ]
    if config.dataset == "breast":
        return [BreastClassicalParty(num_classes) for _ in range(config.num_clients)]
    if config.dataset == "credit":
        return [CreditClassicalParty(num_classes) for _ in range(config.num_clients)]
    raise ValueError(f"Unsupported task: {task_name}")


def build_model(
    task_name: str,
    model_name: str,
    teleport_error_rate: float = 0.0,
    random_seed: int = 42,
) -> nn.Module:
    """Build one of the models actually shown for ``task_name`` in the paper."""
    if model_name not in models_for_task(task_name):
        allowed = ", ".join(models_for_task(task_name))
        raise ValueError(f"{model_name} is not shown for {task_name}; choose: {allowed}")
    config = PAPER_TASKS[task_name]
    num_classes = len(config.classes)

    if model_name == "classical-average":
        return ClassicalAverage(_classical_parties(task_name))
    if model_name == "classical-fuse":
        return ClassicalFuse(_classical_parties(task_name), num_classes)
    if model_name == "eviqvfl":
        return EviQVFL(_quantum_parties(task_name), teleport_error_rate)
    if model_name == "measure-average":
        return MeasureAverage(_quantum_parties(task_name), num_classes)
    if model_name == "measure-vqc":
        return MeasureVQC(_quantum_parties(task_name), num_classes)

    reducer_factory = _feature_reducer_factory(task_name)
    reducers = [reducer_factory() for _ in range(config.num_clients)]
    if model_name == "teleported-vqc":
        return TeleportedVQC(
            reducers,
            config.qubits_per_client,
            config.local_vqc_layers,
            num_classes,
        )
    if model_name == "teleported-random-fixed":
        return TeleportedRandomFixed(
            reducers,
            config.qubits_per_client,
            config.local_vqc_layers,
            num_classes,
            random_seed=random_seed,
        )
    raise AssertionError(f"Unhandled model: {model_name}")

