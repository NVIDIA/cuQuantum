# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import pytest

import cupy as cp
try:
    import cirq
    from cuquantum.tensornet._internal import circuit_parser_utils_cirq
except ImportError:
    cirq = circuit_parser_utils_cirq = None
try:
    import qiskit
except ImportError:
    qiskit = None
try:
    import torch
except ImportError:
    torch = None
from cuquantum.tensornet.experimental import NetworkState, NetworkOperator

from .circuit_data import cirq_circuits, get_qiskit_unitary_gate, qiskit_circuits
from .test_utils import DEFAULT_RNG

DEFAULT_NUM_RANDOM_LAYERS = 2


def get_cirq_random_2q_gate(rng=DEFAULT_RNG):
    random_state = int(rng.integers(0, high=2023))
    random_unitary = cirq.testing.random_unitary(4, random_state=random_state) # random 2-qubit gate
    random_gate = cirq.MatrixGate(random_unitary)
    return random_gate


def gen_random_layered_cirq_circuit(qubits, num_random_layers=DEFAULT_NUM_RANDOM_LAYERS):
    n_qubits = len(qubits)
    operations = []
    for n in range(num_random_layers):
        for i in range(n%2, n_qubits-1, 2):
            operations.append(get_cirq_random_2q_gate().on(qubits[i], qubits[i+1]))
    return cirq.Circuit(operations)


def cirq_insert_random_layers(circuit, num_random_layers=DEFAULT_NUM_RANDOM_LAYERS):
    if num_random_layers == 0:
        return circuit
    qubits = sorted(circuit.all_qubits())
    circuit = circuit_parser_utils_cirq.remove_measurements(circuit)
    pre_circuit = gen_random_layered_cirq_circuit(qubits, num_random_layers=num_random_layers)
    return pre_circuit.concat_ragged(circuit)

cirq_circuits_mps = [cirq_insert_random_layers(circuit) for circuit in cirq_circuits]


def gen_random_layered_qiskit_circuit(qubits, num_random_layers=DEFAULT_NUM_RANDOM_LAYERS):
    n_qubits = len(qubits)
    circuit = qiskit.QuantumCircuit(qubits)
    for n in range(num_random_layers):
        for i in range(n%2, n_qubits-1, 2):
            circuit.append(get_qiskit_unitary_gate(), qubits[i:i+2])
    return circuit


def qiskit_insert_random_layers(circuit, num_random_layers=DEFAULT_NUM_RANDOM_LAYERS):
    if num_random_layers == 0:
        return circuit
    qubits = circuit.qubits
    circuit.remove_final_measurements()
    pre_circuit = gen_random_layered_qiskit_circuit(qubits, num_random_layers=num_random_layers)
    circuit.data = pre_circuit.data + circuit.data
    return circuit

qiskit_circuits_mps = [qiskit_insert_random_layers(circuit) for circuit in qiskit_circuits]

testing_circuits_mps = cirq_circuits_mps + qiskit_circuits_mps

qudits_to_test = (
    4, 
    6, 
    8,
    (3, 3, 3, 3, 3), 
    (2, 3, 2, 3, 4, 2), 
    (2, 5, 3, 2)
)

# NOTE: as of cuTensorNet v2.4.0, exact qudits simulation is not supported when there are non-adjacent two qudit gates in the system
state_settings = (
    # adjacent_double_layer, mpo_bond_dim, mpo_num_sites, mpo_geometry, ct_target_place, initial_mps_dim
    (True, 2, 4, 'adjacent-ordered', "first", None),
    (True, 2, 4, 'random-ordered', "middle", 2),
    (True, 2, 4, 'random', "last", 3),
    (False, 3, None, 'adjacent-ordered', "first", None),
    (False, 3, None, 'random-ordered', "last", 2),
    (False, 3, None, 'random', "middle", None)
)

dtype_generator = itertools.cycle(['float32', 'float64', 'complex64', 'complex128'])
# TODO: The second condition should be removed once PyTorch has full support for Blackwell
if torch is None or int(cp.cuda.Device().compute_capability) >= 100:
    backend_generator = itertools.cycle(['numpy', 'cupy'])
else:
    backend_generator = itertools.cycle(['numpy', 'cupy', 'torch', 'torch-cpu'])

general_factory_states = []

for qudits in qudits_to_test:
    for state_setting in state_settings:
        state_case = {
            'qudits': qudits,
            'dtype': next(dtype_generator),
            'backend': next(backend_generator),
            'state_setting': state_setting
        }
        general_factory_states.append(state_case)

# NOTE: tests based on random operands from StateFactory may be sentitive to numerical noise
approx_mps_options = (
    {'gauge_option': 'free', 'max_extent': 2, 'canonical_center': 0, 'algorithm': 'gesvdj', 'gesvdj_max_sweeps': 100, 'normalization': 'LInf'}, # fixed extent truncation
    {'gauge_option': 'free', 'abs_cutoff': 0.1, 'canonical_center': 3, 'rel_cutoff': 0.2, 'discarded_weight_cutoff': 0.1}, # value based truncation
    {'gauge_option': 'free', 'max_extent': 4, 'normalization': 'L1', 'abs_cutoff': 0.1},
    {'gauge_option': 'free', 'max_extent': 3, 'canonical_center': 1, 'rel_cutoff': 0.1, 'normalization': 'L2'},
    {'gauge_option': 'simple', 'max_extent': 2, 'canonical_center': 2, 'normalization': 'LInf'}, # SU with fixed extent truncation
    {'gauge_option': 'simple', 'abs_cutoff': 0.1, 'canonical_center': 1, 'rel_cutoff': 0.2, 'discarded_weight_cutoff': 0.1}, # SU with value based truncation
    {'gauge_option': 'simple', 'algorithm': 'gesvdj', 'gesvdj_max_sweeps': 100, 'max_extent': 3, 'rel_cutoff': 0.1, 'normalization': 'L2'}
)

# state with unitary channels to test
# (qudits, initial_mps_dim, config, dtype) for each test case
noisy_state_tests = (
    (4, None, {}, 'complex128'), 
    (7, 2, {}, 'complex64'),
    (7, 2, {'gauge_option': 'free'}, 'complex64'), # special test case 
    (4, 2, {'mpo_application': 'exact', 'gauge_option': 'free', }, 'complex128'), 
    (4, 2, {'mpo_application': 'exact', 'gauge_option': 'free', }, 'complex64'), # exact MPS simulation
    (4, 2, {'mpo_application': 'exact', 'gauge_option': 'simple', }, 'complex64'), # exact MPS simulation
    (5, None, {'rel_cutoff': 0.1, 'gauge_option': 'free'}, 'complex128'),
    (5, None, {'rel_cutoff': 0.1, 'gauge_option': 'simple'}, 'complex128'),
    (5, 2,  {'max_extent': 2, 'abs_cutoff': 0.2, 'gauge_option': 'free'}, "complex128"),  # abs cutoff test
    (6, 2, {'max_extent': 4, 'gauge_option': 'free'}, 'complex128'),
    (6, 2, {'max_extent': 4, 'gauge_option': 'simple'}, 'complex128'),
)

@pytest.fixture(scope="session")
def svd_algorithm_cycle():
    # restrict to gesvd/gesvdj algorithm to avoid accuracy fallout
    return itertools.cycle(('gesvd', 'gesvdj'))

@pytest.fixture(scope="function")
def svd_algorithm(svd_algorithm_cycle):
    return next(svd_algorithm_cycle)

STATE_UPDATE_CONFIGS = ({}, {'max_extent': 2}, {'rel_cutoff': 0.12, 'gauge_option': 'simple'})

def create_vqc_states(config, with_control=False):
    # specify the dimensions of the tensor network state
    n_state_modes = 6
    state_mode_extents = (2, ) * n_state_modes
    dtype = 'complex128'

    # create random operators
    cp.random.seed(8) # seed is chosen such that the value based truncation in STATE_UPDATE_CONFIGS will yield different output MPS extents for state_a and state_b
    random_complex = lambda *args, **kwargs: cp.random.random(*args, **kwargs) + 1.j * cp.random.random(*args, **kwargs)
    op_one_body = random_complex((2, 2,))
    if with_control:
        op_two_body_x = random_complex((2, 2,))
        op_two_body_y = random_complex((2, 2,))
    else:
        op_two_body_x = random_complex((2, 2, 2, 2))
        op_two_body_y = random_complex((2, 2, 2, 2))

    state_a = NetworkState(state_mode_extents, dtype=dtype, config=config)
    state_b = NetworkState(state_mode_extents, dtype=dtype, config=config)

    # apply one body tensor operators to the tensor network state
    for i in range(n_state_modes):
        modes_one_body = (i, )
        # op_one_body are fixed, therefore setting immutable to True
        tensor_id_a = state_a.apply_tensor_operator(modes_one_body, op_one_body, immutable=True)
        tensor_id_b = state_b.apply_tensor_operator(modes_one_body, op_one_body, immutable=True)
        assert tensor_id_a == tensor_id_b

    two_body_op_ids = []
    # apply two body tensor operators to the tensor network state
    for i in range(0, n_state_modes, 2):
        if with_control:
            control_mode = (i, )
            target_mode = (i+1, )  
        else:
            control_mode = None
            target_mode = (i, i+1)
            
        tensor_id_a = state_a.apply_tensor_operator(target_mode, op_two_body_x, control_modes=control_mode, control_values=0, immutable=False)
        tensor_id_b = state_b.apply_tensor_operator(target_mode, op_two_body_y, control_modes=control_mode, control_values=0, immutable=False)
        assert tensor_id_a == tensor_id_b
        two_body_op_ids.append(tensor_id_a)
    
    pauli_string = {'IXIXIX': 0.5, 'IYIYIY': 0.2, 'IZIZIZ': 0.3, 'IIIIII': 0.1, 'ZIZIZI': 0.4, 'XIXIXI': 0.2}
    operator = NetworkOperator.from_pauli_strings(pauli_string, dtype='complex128')

    return (state_a, op_two_body_x), (state_b, op_two_body_y), operator, two_body_op_ids
