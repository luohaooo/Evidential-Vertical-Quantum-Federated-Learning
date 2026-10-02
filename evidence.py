"""Evidence fusion and the modeled teleportation Pauli channel."""

from __future__ import annotations

import math

import torch
from torch import nn


CCR_KERNEL = torch.tensor(
    [[[1, 1], [1, 0]], [[0, 0], [0, 1]]], dtype=torch.float32
)


def evidence2prob(num_classes: int) -> torch.Tensor:
    """Create the plausibility transform for a binary power-set encoding."""
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if num_classes == 1:
        return torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    previous = evidence2prob(num_classes - 1)
    zeros = torch.zeros(1, previous.shape[1], dtype=previous.dtype)
    ones = torch.ones(1, previous.shape[1], dtype=previous.dtype)
    return torch.cat(
        [torch.cat([previous, previous], dim=1), torch.cat([zeros, ones], dim=1)],
        dim=0,
    )


class TeleportationPauliNoise(nn.Module):
    """Mass-space equivalent of the symmetric single-qubit Pauli channel.

    For computational-basis evidence masses, X and Y flip the evidence bit,
    whereas Z does not. The effective independent bit-flip probability is
    therefore ``2 * epsilon / 3`` for each transmitted evidence qubit.
    """

    def __init__(self, error_rate: float = 0.0):
        super().__init__()
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be in [0, 1]")
        self.error_rate = float(error_rate)

    def forward(self, masses: torch.Tensor) -> torch.Tensor:
        if self.error_rate == 0.0:
            return masses
        if masses.ndim < 2:
            raise ValueError("masses must have at least two dimensions")
        mass_dim = masses.shape[-1]
        num_qubits = int(math.log2(mass_dim))
        if 2**num_qubits != mass_dim:
            raise ValueError("the final mass dimension must be a power of two")

        flip_rate = 2.0 * self.error_rate / 3.0
        noisy = masses.reshape(*masses.shape[:-1], *([2] * num_qubits))
        first_axis = noisy.ndim - num_qubits
        for axis in range(first_axis, noisy.ndim):
            noisy = (1.0 - flip_rate) * noisy + flip_rate * torch.flip(
                noisy, dims=(axis,)
            )
        return noisy.reshape_as(masses)

    def extra_repr(self) -> str:
        return f"error_rate={self.error_rate}"


class CCRServer(nn.Module):
    """Non-parametric evidential fusion server used by eviQVFL."""

    def __init__(self, num_clients: int, teleport_error_rate: float = 0.0):
        super().__init__()
        if num_clients < 1:
            raise ValueError("num_clients must be positive")
        self.num_clients = num_clients
        self.register_buffer("kernel", CCR_KERNEL)
        self.teleportation_noise = TeleportationPauliNoise(teleport_error_rate)

    def forward(self, masses: torch.Tensor) -> torch.Tensor:
        if masses.ndim != 3 or masses.shape[1] != self.num_clients:
            raise ValueError(
                "masses must have shape [batch, num_clients, mass_dimension]"
            )
        masses = self.teleportation_noise(masses)
        fused = masses[:, 0]
        num_classes = int(math.log2(fused.shape[1]))
        if 2**num_classes != fused.shape[1]:
            raise ValueError("mass dimension must be a power of two")
        for client in range(1, self.num_clients):
            fused = self._combine(fused, masses[:, client], num_classes)

        transform = evidence2prob(num_classes).to(fused.device)
        plausibility = transform.matmul(fused.t()).t()
        return plausibility / plausibility.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def _combine(
        self, mass_a: torch.Tensor, mass_b: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        tensor = self.kernel
        for _ in range(num_classes - 1):
            tensor = torch.kron(tensor, self.kernel)
        return torch.einsum("ijk,bj,bk->bi", tensor, mass_a, mass_b)

