from typing import Tuple, Callable

# PyTorch
import torch

# Torch Quantum
import torchquantum as tq
import torchquantum.functional as tqf


class VQC(tq.QuantumModule):

    def __init__(self,
                 n_wires: int = 8,
                 n_qlayers: int = 1,
                 n_types: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.n_types = n_types

        # Setting up tensor product encoder
        enc_cnt = list()
        for i in range(self.n_wires):
            cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            enc_cnt.append(cnt)
        self.encoder = tq.GeneralEncoder(enc_cnt)

        # We create trainable model parameters, which are stored in dict
        self.params_rx_dct = tq.QuantumModuleDict()
        self.params_ry_dct = tq.QuantumModuleDict()
        self.params_rz_dct = tq.QuantumModuleDict()

        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[str(i + k*self.n_wires)] = tq.RX(has_params=True, trainable=True)
                self.params_ry_dct[str(i + k*self.n_wires)] = tq.RY(has_params=True, trainable=True)
                self.params_rz_dct[str(i + k*self.n_wires)] = tq.RZ(has_params=True, trainable=True)
        # The observables are Hermitian operator based on Pauli-Z
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self,
                x: torch.Tensor,
                q_device: tq.QuantumDevice):

        self.q_device = q_device

        # encode classical inputs
        self.encoder(self.q_device, x)

        # vqc part
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[str(i + k*self.n_wires)](self.q_device, wires=i)
                self.params_ry_dct[str(i + k*self.n_wires)](self.q_device, wires=i)
                self.params_rz_dct[str(i + k*self.n_wires)](self.q_device, wires=i)

            for i in range(self.n_wires):
                if i == self.n_wires-1:
                    tqf.cnot(self.q_device, wires=[i, 0], static=self.static_mode,
                             parent_graph=self.graph)
                else:
                    tqf.cnot(self.q_device, wires=[i, i+1], static=self.static_mode,
                             parent_graph=self.graph)

        state = self.q_device.states
        probs = torch.abs(state) ** 2
        if self.n_types < self.n_wires:
            probs_k = probs.sum(dim=tuple(range(self.n_types+1, self.n_wires+1)))
            B = probs_k.shape[0]
            probs_k = probs_k.reshape(B,-1)
        elif self.n_types == self.n_wires:
            probs_k = probs
            B = probs_k.shape[0]
            probs_k = probs_k.reshape(B,-1)
        else:
            print("The qubits num cannot be less than the classes to be classified.")

        return probs_k


class Overall_VQC(tq.QuantumModule):
    """
    Overall VQC with three-phase quantum circuit:
    1. Local VQC: Each client group processes independently (n_layers times)
    2. Global VQC: All qubits interact with cross-group entanglement (n_global_layers times)
    3. Measurement: Only measure the last n_types qubits

    Args:
        n_clients: Number of client groups
        n_qubit_client: Number of qubits per client group
        n_layers: Number of local VQC layers (default: 1)
        n_types: Number of classification types (default: 2)
        n_global_layers: Number of server VQC layers (default: 1)
    """

    def __init__(self,
                 n_clients: int = 4,
                 n_qubit_client: int = 4,
                 n_layers: int = 1,
                 n_types: int = 2,
                 n_global_layers: int = 1):
        super().__init__()

        self.n_clients = n_clients
        self.n_qubit_client = n_qubit_client
        self.n_layers = n_layers
        self.n_types = n_types
        self.n_total = n_clients * n_qubit_client
        if not isinstance(n_global_layers, int) or isinstance(n_global_layers, bool) or n_global_layers < 1:
            raise ValueError("n_global_layers must be a positive integer")
        self.n_global_layers = n_global_layers

        # Encoder: RY gate for each qubit
        enc_cnt = []
        for i in range(self.n_total):
            cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            enc_cnt.append(cnt)
        self.encoder = tq.GeneralEncoder(enc_cnt)

        # Local VQC parameters (within each group)
        self.local_params_rx = tq.QuantumModuleDict()
        self.local_params_ry = tq.QuantumModuleDict()
        self.local_params_rz = tq.QuantumModuleDict()

        for layer in range(self.n_layers):
            for group in range(self.n_clients):
                for offset in range(self.n_qubit_client):
                    qubit_idx = group * self.n_qubit_client + offset
                    key = f"local_L{layer}_G{group}_Q{offset}"
                    self.local_params_rx[key] = tq.RX(has_params=True, trainable=True)
                    self.local_params_ry[key] = tq.RY(has_params=True, trainable=True)
                    self.local_params_rz[key] = tq.RZ(has_params=True, trainable=True)

        # Global VQC parameters (all qubits)
        self.global_params_rx = tq.QuantumModuleDict()
        self.global_params_ry = tq.QuantumModuleDict()
        self.global_params_rz = tq.QuantumModuleDict()

        for layer in range(self.n_global_layers):
            for i in range(self.n_total):
                # Retain the original first-layer keys for existing checkpoints.
                key = f"global_Q{i}" if layer == 0 else f"global_L{layer}_Q{i}"
                self.global_params_rx[key] = tq.RX(has_params=True, trainable=True)
                self.global_params_ry[key] = tq.RY(has_params=True, trainable=True)
                self.global_params_rz[key] = tq.RZ(has_params=True, trainable=True)

    @tq.static_support
    def forward(self,
                x: torch.Tensor,
                q_device: tq.QuantumDevice):

        self.q_device = q_device

        # Phase 0: Encode classical inputs to all qubits
        self.encoder(self.q_device, x)

        # Phase 1: Local VQC - Process each group independently (n_layers times)
        for layer in range(self.n_layers):
            # Apply single-qubit gates and entanglement within each group
            for group in range(self.n_clients):
                base_qubit = group * self.n_qubit_client

                # Single-qubit rotations within group
                for offset in range(self.n_qubit_client):
                    qubit_idx = base_qubit + offset
                    key = f"local_L{layer}_G{group}_Q{offset}"
                    self.local_params_rx[key](self.q_device, wires=qubit_idx)
                    self.local_params_ry[key](self.q_device, wires=qubit_idx)
                    self.local_params_rz[key](self.q_device, wires=qubit_idx)

                # CNOT entanglement within group (ring topology)
                for offset in range(self.n_qubit_client):
                    ctrl_qubit = base_qubit + offset
                    targ_qubit = base_qubit + ((offset + 1) % self.n_qubit_client)
                    tqf.cnot(self.q_device, wires=[ctrl_qubit, targ_qubit],
                            static=self.static_mode, parent_graph=self.graph)

        # Phase 2: Server layers, each with rotations and a global CNOT ring.
        for layer in range(self.n_global_layers):
            for i in range(self.n_total):
                key = f"global_Q{i}" if layer == 0 else f"global_L{layer}_Q{i}"
                self.global_params_rx[key](self.q_device, wires=i)
                self.global_params_ry[key](self.q_device, wires=i)
                self.global_params_rz[key](self.q_device, wires=i)

            for i in range(self.n_total):
                ctrl_qubit = i
                targ_qubit = (i + 1) % self.n_total
                tqf.cnot(self.q_device, wires=[ctrl_qubit, targ_qubit],
                        static=self.static_mode, parent_graph=self.graph)

        # Phase 3: Measurement - Only measure last n_types qubits
        state = self.q_device.states
        probs = torch.abs(state) ** 2

        # Marginalize (sum over) the first (n_total - n_types) qubits
        # state shape: [batch, 2, 2, ..., 2] with (n_total + 1) dimensions
        if self.n_types < self.n_total:
            # Dimensions to sum over: 1 to (n_total - n_types + 1)
            dims_to_sum = tuple(range(1, self.n_total - self.n_types + 1))
            probs_reduced = probs.sum(dim=dims_to_sum)
            # Result shape: [batch, 2, 2, ..., 2] with (n_types + 1) dimensions
            B = probs_reduced.shape[0]
            probs_final = probs_reduced.reshape(B, -1)  # [batch, 2^n_types]
        elif self.n_types == self.n_total:
            B = probs.shape[0]
            probs_final = probs.reshape(B, -1)
        else:
            raise ValueError(f"n_types ({self.n_types}) cannot exceed n_total ({self.n_total})")
        
        # print(probs_final.shape)

        return probs_final


class RandomFixedOverall_VQC(tq.QuantumModule):
    """Trainable party circuits followed by a frozen random fusion circuit.

    The first phase applies an independent trainable VQC to every party's
    qubit group.  Because there are no cross-party gates in this phase, the
    resulting joint state is the tensor product of the party states.  Under
    ideal, noiseless teleportation, constructing that tensor product directly
    is equivalent to teleporting every party state to the server.

    The server phase applies fixed random rotations and a random CNOT ring.
    Its angles and topology are buffers rather than parameters, so the server
    contributes no trainable parameters.
    """

    def __init__(self,
                 n_clients: int = 4,
                 n_qubit_client: int = 4,
                 n_local_layers: int = 2,
                 n_fusion_layers: int = 1,
                 n_types: int = 2,
                 random_seed: int = 42):
        super().__init__()

        self.n_clients = n_clients
        self.n_qubit_client = n_qubit_client
        self.n_local_layers = n_local_layers
        self.n_fusion_layers = n_fusion_layers
        self.n_types = n_types
        self.n_total = n_clients * n_qubit_client
        self.random_seed = random_seed

        if n_clients < 1 or n_qubit_client < 1:
            raise ValueError("n_clients and n_qubit_client must be positive")
        if n_local_layers < 1 or n_fusion_layers < 1:
            raise ValueError("local and fusion layer counts must be positive")
        if not 1 <= n_types <= self.n_total:
            raise ValueError("n_types must be between 1 and the total qubit count")

        enc_cnt = [
            {'input_idx': [wire], 'func': 'ry', 'wires': [wire]}
            for wire in range(self.n_total)
        ]
        self.encoder = tq.GeneralEncoder(enc_cnt)

        self.local_params_rx = tq.QuantumModuleDict()
        self.local_params_ry = tq.QuantumModuleDict()
        self.local_params_rz = tq.QuantumModuleDict()
        for layer in range(self.n_local_layers):
            for client in range(self.n_clients):
                for offset in range(self.n_qubit_client):
                    key = f"local_L{layer}_C{client}_Q{offset}"
                    self.local_params_rx[key] = tq.RX(has_params=True, trainable=True)
                    self.local_params_ry[key] = tq.RY(has_params=True, trainable=True)
                    self.local_params_rz[key] = tq.RZ(has_params=True, trainable=True)

        generator = torch.Generator(device='cpu').manual_seed(random_seed)
        fixed_angles = 2 * torch.pi * torch.rand(
            n_fusion_layers, self.n_total, 3, generator=generator
        ) - torch.pi
        fixed_permutations = torch.stack([
            torch.randperm(self.n_total, generator=generator)
            for _ in range(n_fusion_layers)
        ])
        self.register_buffer('fixed_angles', fixed_angles)
        self.register_buffer('fixed_permutations', fixed_permutations)

    @tq.static_support
    def forward(self,
                x: torch.Tensor,
                q_device: tq.QuantumDevice):
        self.q_device = q_device
        self.encoder(self.q_device, x)

        # Independent trainable party-side VQCs.
        for layer in range(self.n_local_layers):
            for client in range(self.n_clients):
                base = client * self.n_qubit_client
                for offset in range(self.n_qubit_client):
                    wire = base + offset
                    key = f"local_L{layer}_C{client}_Q{offset}"
                    self.local_params_rx[key](self.q_device, wires=wire)
                    self.local_params_ry[key](self.q_device, wires=wire)
                    self.local_params_rz[key](self.q_device, wires=wire)
                for offset in range(self.n_qubit_client):
                    control = base + offset
                    target = base + ((offset + 1) % self.n_qubit_client)
                    tqf.cnot(self.q_device, wires=[control, target],
                             static=self.static_mode, parent_graph=self.graph)

        # Server-side frozen random circuit.
        for layer in range(self.n_fusion_layers):
            for wire in range(self.n_total):
                angles = self.fixed_angles[layer, wire]
                tqf.rx(self.q_device, wires=wire, params=angles[0].reshape(1, 1),
                       static=self.static_mode, parent_graph=self.graph)
                tqf.ry(self.q_device, wires=wire, params=angles[1].reshape(1, 1),
                       static=self.static_mode, parent_graph=self.graph)
                tqf.rz(self.q_device, wires=wire, params=angles[2].reshape(1, 1),
                       static=self.static_mode, parent_graph=self.graph)

            permutation = self.fixed_permutations[layer].tolist()
            for index, control in enumerate(permutation):
                target = permutation[(index + 1) % self.n_total]
                tqf.cnot(self.q_device, wires=[control, target],
                         static=self.static_mode, parent_graph=self.graph)

        state = self.q_device.states
        probabilities = torch.abs(state) ** 2
        if self.n_types < self.n_total:
            summed_dims = tuple(range(1, self.n_total - self.n_types + 1))
            probabilities = probabilities.sum(dim=summed_dims)
        return probabilities.reshape(probabilities.shape[0], -1)


