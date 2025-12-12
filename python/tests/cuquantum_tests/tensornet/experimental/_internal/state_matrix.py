# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = ["CircuitStateMatrix", "GenericStateMatrix", "SimulationConfigMatrix"]

import numpy as np
try:
    import cirq
    from cuquantum.tensornet._internal import circuit_parser_utils_cirq
except ImportError:
    cirq = circuit_parser_utils_cirq = None
try:
    import qiskit
except ImportError:
    qiskit = None

from cuquantum.tensornet.experimental import TNConfig, MPSConfig

from ...utils.circuit_matrix import CirqCircuitMatrix, QiskitCircuitMatrix, CircuitMatrixABC, get_qiskit_unitary_gate
from ...utils.helpers import get_rng_iterator, get_array_framework_iterator
from .state_factory import StateFactory


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
    return StateFactory(*args, **kwargs, backend=next(array_framework_iterator))

generic_states_L0 = [
    create_state_factory(4, "float32", "SDDS", next(rng_iterator)),
    create_state_factory(5, "float64", "SDMDS", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory((2, 3, 4, 3, 2), "complex64", "SDM", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory(4, "complex128", "SDCS", next(rng_iterator), ct_target_place="first"),
    create_state_factory((2, 5, 3, 2), "complex64", "SDDS", next(rng_iterator), initial_mps_dim=2),
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
]

generic_states_L2 = [
    create_state_factory(4, "float32", "SDDS", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory(6, "float64", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=4, mpo_geometry="adjacent-ordered"),
    create_state_factory(5, "float64", "SDCDS", next(rng_iterator), ct_target_place="first", initial_mps_dim=2),
    create_state_factory(6, "float64", "SDCSD", next(rng_iterator), ct_target_place="middle", adjacent_double_layer=False),
    create_state_factory(5, "float64", "SDCDS", next(rng_iterator), ct_target_place="last", initial_mps_dim=2),
    create_state_factory(8, "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=5, mpo_geometry="random-ordered", initial_mps_dim=2),
    # qudits
    create_state_factory((3, 3, 3, 3, 3), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=3, mpo_num_sites=4, mpo_geometry="random"),
    create_state_factory((2, 3, 2, 3, 4, 2), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_geometry="adjacent-ordered"),
    create_state_factory((2, 5, 3, 2), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_geometry="random", initial_mps_dim=2),
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
]

noisy_state_tests_L1 = [
    create_state_factory(6, "complex128", "SDuDS", next(rng_iterator), initial_mps_dim=3),
    create_state_factory(6, "complex128", "SDgDS", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory(5, "complex128", "SDUDS", next(rng_iterator), adjacent_double_layer=False, initial_mps_dim=2),
    create_state_factory(7, "complex128", "SDGDS", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory(6, "complex128", "SDuMS", next(rng_iterator), mpo_bond_dim=2, initial_mps_dim=2),
    create_state_factory(5, "complex128", "SDgMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=3, mpo_geometry="random"),
    create_state_factory(5, "complex128", "SDuDgDS", next(rng_iterator)),
]

class NoisyStateMatrix(StateMatrixABC):

    @staticmethod
    def L0():
        return noisy_state_tests_L0
    
    @staticmethod
    def L1():
        return noisy_state_tests_L1