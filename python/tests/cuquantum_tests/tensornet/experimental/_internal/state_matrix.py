# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "CircuitStateMatrix",
    "GenericStateMatrix",
    "SimulationConfigMatrix",
    "ExpectationGradientConfig",
    "NetworkOperatorFactory",
]

import importlib
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from nvmath.internal.utils import infer_object_package
try:
    import cirq
    from cuquantum.tensornet._internal import circuit_parser_utils_cirq
except ImportError:
    cirq = circuit_parser_utils_cirq = None
try:
    import qiskit
except ImportError:
    qiskit = None

from cuquantum.tensornet import NetworkOptions
from cuquantum.tensornet.experimental import TNConfig, MPSConfig, NetworkOperator
from cuquantum.tensornet.experimental._internal.network_state_utils import get_pauli_map

from ...utils.circuit_matrix import CirqCircuitMatrix, QiskitCircuitMatrix, CircuitMatrixABC, get_qiskit_unitary_gate
from ...utils.helpers import get_rng_iterator, get_array_framework_iterator, TensorBackend
from .state_factory import StateFactory, _random_unitary, _random_hermitian


def get_cirq_random_2q_gate(rng):
    random_state = int(rng.integers(0, high=2023))
    random_unitary = cirq.testing.random_unitary(4, random_state=random_state) # random 2-qubit gate
    random_gate = cirq.MatrixGate(random_unitary)
    return random_gate

def gen_random_layered_cirq_circuit(qubits, num_random_layers, rng):
    n_qubits = len(qubits)
    operations = []
    for n in range(num_random_layers):
        for i in range(n%2, n_qubits-1, 2):
            operations.append(get_cirq_random_2q_gate(rng).on(qubits[i], qubits[i+1]))
    return cirq.Circuit(operations)

def cirq_insert_random_layers(circuit, num_random_layers, rng):
    if num_random_layers == 0:
        return circuit
    qubits = sorted(circuit.all_qubits())
    circuit = circuit_parser_utils_cirq.remove_measurements(circuit)
    pre_circuit = gen_random_layered_cirq_circuit(qubits, num_random_layers, rng)
    return pre_circuit.concat_ragged(circuit)

def gen_random_layered_qiskit_circuit(qubits, num_random_layers, rng):
    n_qubits = len(qubits)
    circuit = qiskit.QuantumCircuit(qubits)
    for n in range(num_random_layers):
        for i in range(n%2, n_qubits-1, 2):
            circuit.append(get_qiskit_unitary_gate(rng), qubits[i:i+2])
    return circuit

def qiskit_insert_random_layers(circuit, num_random_layers, rng):
    if num_random_layers == 0:
        return circuit
    qubits = circuit.qubits
    circuit.remove_final_measurements()
    pre_circuit = gen_random_layered_qiskit_circuit(qubits, num_random_layers, rng)
    circuit.data = pre_circuit.data + circuit.data
    return circuit

class StateMatrixABC(CircuitMatrixABC):
    pass

qiskit_L0_circuits = [
    qiskit_insert_random_layers(circuit, 1, np.random.default_rng(i))
    for i, circuit in enumerate(QiskitCircuitMatrix.L0())
]
qiskit_L1_circuits = [
    qiskit_insert_random_layers(circuit, 2, np.random.default_rng(i))
    for i, circuit in enumerate(QiskitCircuitMatrix.L1())
]
qiskit_L2_circuits = [
    qiskit_insert_random_layers(circuit, 2, np.random.default_rng(i))
    for i, circuit in enumerate(QiskitCircuitMatrix.L2())
]

class QiskitCircuitStateMatrix(StateMatrixABC):
    
    @staticmethod
    def L0():
        return qiskit_L0_circuits

    @staticmethod
    def L1():
        return qiskit_L1_circuits

    @staticmethod
    def L2():
        return qiskit_L2_circuits

cirq_L0_circuits = [
    cirq_insert_random_layers(circuit, 1, np.random.default_rng(i))
    for i, circuit in enumerate(CirqCircuitMatrix.L0())
]
cirq_L1_circuits = [
    cirq_insert_random_layers(circuit, 2, np.random.default_rng(i))
    for i, circuit in enumerate(CirqCircuitMatrix.L1())
]
cirq_L2_circuits = [
    cirq_insert_random_layers(circuit, 2, np.random.default_rng(i))
    for i, circuit in enumerate(CirqCircuitMatrix.L2())
]

class CirqCircuitStateMatrix(StateMatrixABC):

    @staticmethod
    def L0():
        return cirq_L0_circuits

    @staticmethod
    def L1():
        return cirq_L1_circuits

    @staticmethod
    def L2():
        return cirq_L2_circuits

class CircuitStateMatrix(StateMatrixABC):

    @staticmethod
    def L0():
        return CirqCircuitStateMatrix.L0() + QiskitCircuitStateMatrix.L0()
    
    @staticmethod
    def L1():
        return CirqCircuitStateMatrix.L1() + QiskitCircuitStateMatrix.L1()
    
    @staticmethod
    def L2():
        return CirqCircuitStateMatrix.L2() + QiskitCircuitStateMatrix.L2()
    

rng_iterator = get_rng_iterator()
array_framework_iterator = get_array_framework_iterator()

def create_state_factory(*args, **kwargs):
    backend = kwargs.pop("backend", None)
    if backend is None:
        backend = next(array_framework_iterator)
    return StateFactory(*args, **kwargs, backend=backend)


generic_states_L0 = [
    create_state_factory(4, "float32", "SDDS", next(rng_iterator)),
    create_state_factory(5, "float64", "SDMDS", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory((2, 3, 4, 3, 2), "complex64", "SDM", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory(4, "complex128", "SDCS", next(rng_iterator), ct_target_place="first"),
    create_state_factory((2, 5, 3, 2), "complex64", "SDDS", next(rng_iterator), initial_mps_dim=2),
    create_state_factory((2, 5, 2, 2), "float64", "SDDD", next(rng_iterator), adjacent_double_layer=False),
    # Diagonal gate test cases
    create_state_factory(4, "complex128", "SADCAS", next(rng_iterator), ct_target_place="first"),
]

generic_states_L1 = [
    create_state_factory(5, "float32", "SDDS", next(rng_iterator)),
    create_state_factory((2, 3, 4, 3, 2), "complex64", "SDDS", next(rng_iterator)),
    create_state_factory(5, "complex64", "SDMS", next(rng_iterator), adjacent_double_layer=False, mpo_bond_dim=3),
    create_state_factory(4, "float64", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=3, mpo_geometry="random-ordered"),
    create_state_factory((2, 5, 3, 2), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_geometry="random"),
    create_state_factory(6, "float64", "SDCS", next(rng_iterator), ct_target_place="first"),
    create_state_factory(4, "float64", "SDCS", next(rng_iterator), ct_target_place="middle"),
    create_state_factory(5, "float64", "SDCS", next(rng_iterator), ct_target_place="last", initial_mps_dim=2),
    create_state_factory((2, 3, 3, 2, 2, 2), "float64", "SMDD", next(rng_iterator), mpo_bond_dim=2, adjacent_double_layer=False),
    # Diagonal gate test cases
    # {S, D, A}: all dtypes, exact simulation
    create_state_factory(4, "float32", "SADAAA", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory((2, 5, 3, 2), "complex64", "SADA", next(rng_iterator), adjacent_double_layer=True),
    create_state_factory((2, 3, 2, 3, 4, 2), "complex128", "ASDA", next(rng_iterator), adjacent_double_layer=True),
]

generic_states_L2 = [
    create_state_factory(4, "float32", "SDDS", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory(6, "float64", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=4, mpo_geometry="adjacent-ordered"),
    create_state_factory(5, "float64", "SDCDS", next(rng_iterator), ct_target_place="first", initial_mps_dim=2),
    create_state_factory(6, "float64", "SDCSD", next(rng_iterator), ct_target_place="middle", adjacent_double_layer=False),
    create_state_factory(5, "float64", "SDCDS", next(rng_iterator), ct_target_place="last", initial_mps_dim=2),
    create_state_factory(8, "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=5, mpo_geometry="random-ordered", initial_mps_dim=2),
    create_state_factory((3, 3, 3, 3, 3), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=3, mpo_num_sites=4, mpo_geometry="random"),
    create_state_factory((2, 3, 2, 3, 4, 2), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_geometry="adjacent-ordered"),
    create_state_factory((2, 5, 3, 2), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_geometry="random", initial_mps_dim=2),
    # Diagonal gate test cases
    # {S, D, M, A}: double precision, approximate simulation
    create_state_factory((2, 5, 3, 2), "float64", "SADMA", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory((2, 3, 2, 3, 4, 2), "complex128", "SADA", next(rng_iterator), initial_mps_dim=2),
]

class GenericStateMatrix(StateMatrixABC):

    @staticmethod
    def L0():
        return generic_states_L0
    
    @staticmethod
    def L1():
        return generic_states_L1

    @staticmethod
    def L2():
        return generic_states_L2


exact_mps_configs = [
    MPSConfig(mpo_application='exact', gauge_option='free'),
    {'mpo_application': 'exact', 'gauge_option': 'simple'},
]

approx_mps_configs_L0 = [
    {'gauge_option': 'free', 'max_extent': 2, 'algorithm': 'gesvdj', 'abs_cutoff': 0.1},
    {'gauge_option': 'simple', 'rel_cutoff': 0.1},
]

approx_mps_configs_L1 = [
    {'gauge_option': 'free', 'max_extent': 2, 'algorithm': 'gesvdj', 'abs_cutoff': 0.1, 'gesvdj_max_sweeps': 100},
    {'gauge_option': 'free', 'abs_cutoff': 0.1, 'normalization': 'L1', 'canonical_center': 1},
    {'gauge_option': 'simple', 'max_extent': 4, 'rel_cutoff': 0.1, 'discarded_weight_cutoff': 0.1, 'normalization': 'LInf'},
    {'gauge_option': 'simple', 'max_extent': 3, 'canonical_center': 2, 'rel_cutoff': 0.1, 'normalization': 'L2'},
    {'gauge_option': 'simple', 'max_extent': 3, 'rel_cutoff': 0.1, 'normalization': 'L2', 'abs_cutoff': 0.1}, 
]

approx_mps_configs_L2 = [
    {'gauge_option': 'free', 'max_extent': 2, 'canonical_center': 0, 'algorithm': 'gesvdj', 'gesvdj_max_sweeps': 100, 'normalization': 'LInf'}, # fixed extent truncation
    {'gauge_option': 'free', 'abs_cutoff': 0.1, 'rel_cutoff': 0.2, 'discarded_weight_cutoff': 0.1}, # value based truncation
    {'gauge_option': 'free', 'max_extent': 4, 'normalization': 'L1', 'abs_cutoff': 0.1},
    {'gauge_option': 'free', 'max_extent': 3, 'canonical_center': 1, 'rel_cutoff': 0.1, 'normalization': 'L2'},
    {'gauge_option': 'simple', 'max_extent': 2, 'canonical_center': 2, 'normalization': 'LInf'}, # SU with fixed extent truncation
    {'gauge_option': 'simple', 'abs_cutoff': 0.1, 'canonical_center': 1, 'rel_cutoff': 0.2, 'discarded_weight_cutoff': 0.1}, # SU with value based truncation
    {'gauge_option': 'simple', 'algorithm': 'gesvdj', 'gesvdj_max_sweeps': 100, 'max_extent': 3, 'rel_cutoff': 0.1, 'normalization': 'L2'}
]

class MPSConfigMatrix:

    @staticmethod
    def exactConfigs():
        return exact_mps_configs

    @staticmethod
    def approxConfigsL0():
        return approx_mps_configs_L0

    @staticmethod
    def approxConfigsL1():
        return approx_mps_configs_L1
    
    @staticmethod
    def approxConfigsL2():
        return approx_mps_configs_L2
    
class SimulationConfigMatrix:

    @staticmethod
    def exactConfigs():
        return [TNConfig()] + MPSConfigMatrix.exactConfigs()


noisy_state_tests_L0 = [
    create_state_factory(4, "float32", "SDUDS", next(rng_iterator)),
    create_state_factory(4, "float64", "SDGDS", next(rng_iterator)),
    create_state_factory(5, "complex64", "SDuMS", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory(5, "complex128", "SDgDS", next(rng_iterator), initial_mps_dim=2),
    # Diagonal gate test cases
    create_state_factory(5, "complex128", "SDgADS", next(rng_iterator), initial_mps_dim=2)
]

noisy_state_tests_L1 = [
    create_state_factory(6, "complex128", "SDuDS", next(rng_iterator), initial_mps_dim=3),
    create_state_factory(6, "complex128", "SDgDS", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory(5, "complex128", "SDUDS", next(rng_iterator), adjacent_double_layer=False, initial_mps_dim=2),
    create_state_factory(7, "complex128", "SDGDS", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory(6, "complex128", "SDuMS", next(rng_iterator), mpo_bond_dim=2, initial_mps_dim=2),
    create_state_factory(5, "complex128", "SDgMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=3, mpo_geometry="random"),
    create_state_factory(5, "complex128", "SDuDgDS", next(rng_iterator)),
    # Diagonal gate test cases
    create_state_factory(6, "complex128", "SDAuDS", next(rng_iterator), initial_mps_dim=3),
    create_state_factory(6, "complex128", "SADgDS", next(rng_iterator), adjacent_double_layer=False),
]

class NoisyStateMatrix(StateMatrixABC):

    @staticmethod
    def L0():
        return noisy_state_tests_L0
    
    @staticmethod
    def L1():
        return noisy_state_tests_L1


# --- Expectation gradient test configs ---
# All use StateFactory with S/D layers only (plain gates for TorchRef).
def _exp_grad_config(dtype, backend, factory, hamiltonian=None, **kwargs):
    """Build one expectation-gradient test config dict (shared keys, default adjoint=1.0).
    hamiltonian is either a Pauli string dict or a NetworkOperatorFactory."""
    out = {
        "dtype": dtype,
        "backend": backend,
        "expectation_value_adjoint": 1.0,
        "factory": factory,
    }
    if hamiltonian is not None:
        out["hamiltonian"] = hamiltonian
    out.update(kwargs)
    return out

def make_mpo_tensor_hermitian(t, which):
    """Make MPO tensor Hermitian in physical (ket, bra) indices so the full MPO is Hermitian.
    which: 'first' (ket, n, bra), 'middle' (p, ket, n, bra), or 'last' (p, ket, bra)."""
    pkg = infer_object_package(t)
    if pkg == "torch":
        if which == "first":
            return (t + t.conj().permute(2, 1, 0)) * 0.5
        if which == "last":
            return (t + t.conj().permute(0, 2, 1)) * 0.5
        return (t + t.conj().permute(0, 3, 2, 1)) * 0.5
    xp = importlib.import_module("cupy") if pkg == "cupy" else np
    if which == "first":
        return (t + xp.conj(t).transpose(2, 1, 0)) * 0.5
    if which == "last":
        return (t + xp.conj(t).transpose(0, 2, 1)) * 0.5
    return (t + xp.conj(t).transpose(0, 3, 2, 1)) * 0.5

class NetworkOperatorFactory:
    """
    Deferred NetworkOperator construction.
    
    See also: StateFactory
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def build(self):
        return _build_network_operator(*self._args, **self._kwargs)


def _build_network_operator(state_dims, rng, backend, dtype, options=None, num_repeats=2, real_coefficients=True, use_random_unitary=False, use_random_hermitian=False, add_mpo=False):
    """Build a NetworkOperator by calling append_product for each term; optionally add one append_mpo (MPO) term.
    """
    if isinstance(options, dict):
        device_id = options.get("device_id", None)
    elif isinstance(options, NetworkOptions):
        device_id = options.device_id
    else:
        device_id = None
    if backend == "numpy":
        device_id = None
    backend_obj = TensorBackend(backend=backend, device_id=device_id)
    operator_obj = NetworkOperator(state_dims, dtype=dtype, options=options)
    n_modes = len(state_dims)
    if not use_random_unitary and not use_random_hermitian:
        if any(state_dims[q] != 2 for q in range(n_modes)):
            raise ValueError("Cannot use random Pauli for non-qubit local dimensions")
        use_random_pauli = True
        pauli_map = get_pauli_map(backend, dtype, device_id=device_id)
        pauli_keys = [k for k in ("I", "X", "Y", "Z") if k in pauli_map]
    else:
        use_random_pauli = False

    prod_modes_formatted = [(q,) for q in range(n_modes)]
    for _ in range(num_repeats):
        coefficient = rng.random(1).item()
        if dtype.startswith("complex") and not real_coefficients:
            coefficient += 1j * rng.random(1).item()
        prod_tensors = []
        for q in range(n_modes):
            if use_random_pauli:
                prod_tensors.append(pauli_map[rng.choice(pauli_keys)])
            else:
                shape = (state_dims[q],) * 2
                if use_random_hermitian:
                    t = _random_hermitian(backend_obj, shape, dtype, rng)
                else:
                    t = _random_unitary(backend_obj, shape, dtype, rng)
                prod_tensors.append(t)
        operator_obj.append_product(coefficient, prod_modes_formatted, prod_tensors)

    if add_mpo and n_modes >= 2:
        # Add one MPO term (same pattern as state_factory get_random_network_operator).
        # Hermitian: real coefficient and each tensor Hermitian in physical (ket, bra) indices.
        def get_random_modes():
            num_rand_modes = rng.integers(2, len(state_dims) + 1)  # [2, n_modes] inclusive
            rand_modes = list(range(len(state_dims)))
            rng.shuffle(rand_modes)
            return rand_modes[:num_rand_modes]

        # Real coefficient so the MPO term is Hermitian (tensors are already made Hermitian below).
        coefficient = rng.random(1).item()
        mpo_modes = get_random_modes()
        num_mpo_modes = len(mpo_modes)
        mpo_tensors = []
        bond_prev = None
        for i, m in enumerate(mpo_modes):
            bond_next = rng.integers(2, 5)
            dim = state_dims[m]
            if i == 0:
                shape = (dim, bond_next, dim)
                which = "first"
            elif i == num_mpo_modes - 1:
                shape = (bond_prev, dim, dim)
                which = "last"
            else:
                shape = (bond_prev, dim, bond_next, dim)
                which = "middle"
            t = backend_obj.random(shape, dtype, rng)
            t = make_mpo_tensor_hermitian(t, which)
            mpo_tensors.append(t)
            bond_prev = bond_next
        operator_obj.append_mpo(coefficient, mpo_modes, mpo_tensors)
    return operator_obj


expectation_gradient_L0 = [
    _exp_grad_config("float32", "cupy", create_state_factory(4, "float32", "SDSDS", np.random.default_rng(41), backend="cupy", mark_gradients=True),
        hamiltonian={"ZZXX": 2.0, "XZXZ": 3.0}
    ),
    _exp_grad_config("complex64", "numpy", create_state_factory(6, "complex64", "SSDDSD", np.random.default_rng(42), backend="numpy", mark_gradients=True),
        hamiltonian={"ZZXZYX": 2.0, "IZIXZI": 3.0, "ZYYZZX": 5.0}
    ),
    # Same Pauli strings with identity removal (exercises lightcone simplification)
    _exp_grad_config("complex64", "numpy", create_state_factory(6, "complex64", "SSDDSD", np.random.default_rng(42), backend="numpy", mark_gradients=True),
        hamiltonian={"ZZXZYX": 2.0, "IZIXZI": 3.0, "ZYYZZX": 5.0},
        remove_identity=True,
    ),
    # Non-Hermitian operator (random unitary product terms)
    _exp_grad_config("complex128", "cupy", create_state_factory(4, "complex128", "SDSD", np.random.default_rng(52), backend="cupy", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((2, 2, 2, 2), np.random.default_rng(38), "cupy", dtype="complex128", num_repeats=2, real_coefficients=True, use_random_unitary=True, add_mpo=False),
        non_hermitian=True,
    ),
]

expectation_gradient_L0_torch = [
    _exp_grad_config("complex128", "torch", create_state_factory((2, 3, 2, 4, 2, 5, 2, 3), "complex128", "SDSDSD", np.random.default_rng(43), backend="torch", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((2, 3, 2, 4, 2, 5, 2, 3), np.random.default_rng(31), "torch", dtype="complex128", num_repeats=3, real_coefficients=True, use_random_hermitian=True),
    ),
    # Same as above but with an MPO term
    _exp_grad_config("complex128", "torch", create_state_factory((2, 3, 2, 4, 2, 5, 2, 3), "complex128", "SDSDSD", np.random.default_rng(43), backend="torch", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((2, 3, 2, 4, 2, 5, 2, 3), np.random.default_rng(31), "torch", dtype="complex128", num_repeats=3, real_coefficients=True, use_random_hermitian=True, add_mpo=True),
    ),
] if torch is not None else []

expectation_gradient_L1 = [
    _exp_grad_config("complex64", "cupy", create_state_factory(8, "complex64", "SDSDDSD", np.random.default_rng(45), backend="cupy", mark_gradients=True),
        hamiltonian={"ZYIZXZIZ": 5.0, "XZZYIZXZ": 2.0, "ZZYIXZYY": 3.0}
    ),
    # Same with identity removal (lightcone simplification)
    _exp_grad_config("complex64", "cupy", create_state_factory(8, "complex64", "SDSDDSD", np.random.default_rng(45), backend="cupy", mark_gradients=True),
        hamiltonian={"ZYIZXZIZ": 5.0, "XZZYIZXZ": 2.0, "ZZYIXZYY": 3.0},
        remove_identity=True,
    ),
    _exp_grad_config("complex128", "numpy", create_state_factory((3, 2, 4, 4, 2, 5), "complex128", "SDSDSD", np.random.default_rng(46), backend="numpy", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((3, 2, 4, 4, 2, 5), np.random.default_rng(32), "numpy", dtype="complex128", num_repeats=3, real_coefficients=True, use_random_hermitian=True),
    ),
    # Same as above but with an MPO term
    _exp_grad_config("complex128", "numpy", create_state_factory((3, 2, 4, 4, 2, 5), "complex128", "SDSDSD", np.random.default_rng(46), backend="numpy", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((3, 2, 4, 4, 2, 5), np.random.default_rng(32), "numpy", dtype="complex128", num_repeats=3, real_coefficients=True, use_random_hermitian=True, add_mpo=True),
    ),
]

expectation_gradient_L1_torch = [
    _exp_grad_config("float64", "torch", create_state_factory(6, "float64", "SDSDSDS", np.random.default_rng(44), backend="torch", mark_gradients=True),
        hamiltonian={"ZXIXZI": 4.0, "IXZIZX": 3.0}
    ),
    # Same with identity removal (lightcone simplification)
    _exp_grad_config("float64", "torch", create_state_factory(6, "float64", "SDSDSDS", np.random.default_rng(44), backend="torch", mark_gradients=True),
        hamiltonian={"ZXIXZI": 4.0, "IXZIZX": 3.0},
        remove_identity=True,
    ),
    # Non-Hermitian operator (random unitary product terms)
    _exp_grad_config("complex64", "torch", create_state_factory((2, 3, 2, 3), "complex64", "SDSDS", np.random.default_rng(53), backend="torch", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((2, 3, 2, 3), np.random.default_rng(39), "torch", dtype="complex64", num_repeats=3, real_coefficients=True, use_random_unitary=True),
        non_hermitian=True,
    ),
    # Non-Hermitian operator with MPO term
    _exp_grad_config("complex64", "torch", create_state_factory((2, 3, 2, 3), "complex64", "SDSDS", np.random.default_rng(53), backend="torch", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((2, 3, 2, 3), np.random.default_rng(39), "torch", dtype="complex64", num_repeats=3, real_coefficients=True, use_random_unitary=True, add_mpo=True),
        non_hermitian=True,
    ),
] if torch is not None else []

expectation_gradient_L2 = [
    _exp_grad_config("complex128", "cupy", create_state_factory((3, 3, 3, 3, 3), "complex128", "SSDDS", np.random.default_rng(47), backend="cupy", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((3, 3, 3, 3, 3), np.random.default_rng(33), "cupy", dtype="complex128", num_repeats=4, real_coefficients=True, use_random_hermitian=True),
    ),
    # Same as above but with an MPO term
    _exp_grad_config("complex128", "cupy", create_state_factory((3, 3, 3, 3, 3), "complex128", "SSDDS", np.random.default_rng(47), backend="cupy", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((3, 3, 3, 3, 3), np.random.default_rng(33), "cupy", dtype="complex128", num_repeats=4, real_coefficients=True, use_random_hermitian=True, add_mpo=True),
    ),
    _exp_grad_config("complex64", "numpy", create_state_factory((2, 3, 2, 4, 2, 5, 2, 3), "complex64", "SDSDDSS", np.random.default_rng(49), backend="numpy", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((2, 3, 2, 4, 2, 5, 2, 3), np.random.default_rng(35), "numpy", dtype="complex64", num_repeats=4, real_coefficients=True, use_random_unitary=True),
        non_hermitian=True,
    ),
    # Same as above but with an MPO term
    _exp_grad_config("complex64", "numpy", create_state_factory((2, 3, 2, 4, 2, 5, 2, 3), "complex64", "SDSDDSS", np.random.default_rng(49), backend="numpy", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((2, 3, 2, 4, 2, 5, 2, 3), np.random.default_rng(35), "numpy", dtype="complex64", num_repeats=4, real_coefficients=True, use_random_unitary=True, add_mpo=True),
        non_hermitian=True,
    ),
]

expectation_gradient_L2_torch = [
    # Products + MPO combined (random I, X, Y, Z Paulis + one MPO term)
    _exp_grad_config("float64", "torch", create_state_factory(6, "float64", "SDSDD", np.random.default_rng(48), backend="torch", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((2, 2, 2, 2, 2, 2), np.random.default_rng(34), "torch", dtype="float64", num_repeats=2, real_coefficients=True, add_mpo=True),
    ),
    # Product terms only (same circuit/seed, no MPO)
    _exp_grad_config("float64", "torch", create_state_factory(6, "float64", "SDSDD", np.random.default_rng(48), backend="torch", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((2, 2, 2, 2, 2, 2), np.random.default_rng(34), "torch", dtype="float64", num_repeats=2, real_coefficients=True, add_mpo=False),
    ),
    # MPO term only (different RNG state than combined case)
    _exp_grad_config("float64", "torch", create_state_factory(6, "float64", "SDSDD", np.random.default_rng(48), backend="torch", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory((2, 2, 2, 2, 2, 2), np.random.default_rng(34), "torch", dtype="float64", num_repeats=0, real_coefficients=True, add_mpo=True),
    ),
] if torch is not None else []


class ExpectationGradientConfig:
    """Config lists for expectation gradient tests (compute_expectation_with_gradients vs TorchRef)."""

    @staticmethod
    def L0():
        return expectation_gradient_L0 + expectation_gradient_L0_torch

    @staticmethod
    def L1():
        return expectation_gradient_L1 + expectation_gradient_L1_torch

    @staticmethod
    def L2():
        return expectation_gradient_L2 + expectation_gradient_L2_torch