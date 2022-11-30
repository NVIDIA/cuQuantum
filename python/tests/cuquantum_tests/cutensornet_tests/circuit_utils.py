# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from types import MappingProxyType

try:
    import cirq
except ImportError:
    cirq = None
import cupy as cp
import numpy as np
try:
    import torch
except ImportError:
    torch = None
try:
    import qiskit
except ImportError:
    qiskit = None

from cuquantum import contract, CircuitToEinsum
from cuquantum.cutensornet._internal.circuit_converter_utils import convert_mode_labels_to_expression
from cuquantum.cutensornet._internal.circuit_converter_utils import EINSUM_SYMBOLS_BASE
from cuquantum.cutensornet._internal.circuit_converter_utils import get_pauli_gates
from cuquantum.cutensornet._internal.circuit_converter_utils import parse_gates_to_mode_labels_operands
from .test_utils import atol_mapper, rtol_mapper


# note: this implementation would cause pytorch tests being silently skipped
# if pytorch is not available, which is the desired effect since otherwise
# it'd be too noisy
backends = [np, cp]
if torch:
    backends.append(torch)


cirq_circuits = []
qiskit_circuits = []

EMPTY_DICT = MappingProxyType(dict())


def gen_qubits_map(qubits):
    n_qubits = len(qubits)
    if n_qubits > len(EINSUM_SYMBOLS_BASE):
        raise NotImplementedError(f'test suite only supports up to {len(EINSUM_SYMBOLS_BASE)} qubits')
    qubits_map = dict(zip(qubits, EINSUM_SYMBOLS_BASE[:n_qubits]))
    return qubits_map


def bitstring_generator(n_qubits, nsample=1):
    for _ in range(nsample):
        bitstring = ''.join(np.random.choice(('0', '1'), n_qubits))
        yield bitstring


def where_fixed_generator(qubits, nfix_max, nsite_max=None):
    indices = np.arange(len(qubits))
    for nfix in range(nfix_max):
        np.random.shuffle(indices)
        fixed_sites = [qubits[indices[ix]] for ix in range(nfix)]
        bitstring = ''.join(np.random.choice(('0', '1'), nfix))
        fixed = dict(zip(fixed_sites, bitstring))
        if nsite_max is None:
            yield fixed
        else:
            for nsite in range(1, nsite_max):
                where = [qubits[indices[ix]] for ix in range(nfix, nfix+nsite)]
                yield where, fixed


def random_pauli_string_generator(n_qubits, num_strings=4):
    for _ in range(num_strings):
        yield ''.join(np.random.choice(['I','X', 'Y', 'Z'], n_qubits))


def get_partial_indices(qubits, fixed):
    partial_indices = [slice(None)] * len(qubits)
    index_map = {'0': slice(0, 1),
                 '1': slice(1, 2)}
    for ix, q in enumerate(qubits):
        if q in fixed:
            partial_indices[ix] = index_map[fixed[q]]
    return partial_indices


################################################
# functions to generate cirq.Circuit for testing
################################################

def get_cirq_qft_circuit(n_qubits):
    qubits = cirq.LineQubit.range(n_qubits)
    qreg = list(qubits)[::-1]
    operations = []
    while len(qreg) > 0:
        q_head = qreg.pop(0)
        operations.append(cirq.H(q_head))
        for i, qubit in enumerate(qreg):
            operations.append((cirq.CZ ** (1 / 2 ** (i + 1)))(qubit, q_head))
    circuit = cirq.Circuit(operations)
    return circuit


def get_cirq_random_circuit(n_qubits, n_moments, op_density=0.9, seed=3):
    return cirq.testing.random_circuit(n_qubits, n_moments, op_density, random_state=seed)


N_QUBITS_RANGE = range(7, 9)
N_MOMENTS_RANGE = DEPTH_RANGE = range(5, 7)

if cirq:
    for n_qubits in N_QUBITS_RANGE:
        cirq_circuits.append(get_cirq_qft_circuit(n_qubits))
        for n_moments in N_MOMENTS_RANGE:
            cirq_circuits.append(get_cirq_random_circuit(n_qubits, n_moments))


#########################################################
# functions to generate qiskit.QuantumCircuit for testing
#########################################################

def get_qiskit_qft_circuit(n_qubits):
    return qiskit.circuit.library.QFT(n_qubits, do_swaps=False).decompose()


def get_qiskit_random_circuit(n_qubits, depth):
    from qiskit.circuit.random import random_circuit
    circuit = random_circuit(n_qubits, depth, max_operands=3)
    return circuit


def get_qiskit_composite_circuit():
    sub_q = qiskit.QuantumRegister(2)
    sub_circ = qiskit.QuantumCircuit(sub_q, name='sub_circ')
    sub_circ.h(sub_q[0])
    sub_circ.crz(1, sub_q[0], sub_q[1])
    sub_circ.barrier()
    sub_circ.id(sub_q[1])
    sub_circ.u(1, 2, -2, sub_q[0])

    # Convert to a gate and stick it into an arbitrary place in the bigger circuit
    sub_inst = sub_circ.to_instruction()

    qr = qiskit.QuantumRegister(3, 'q')
    circ = qiskit.QuantumCircuit(qr)
    circ.h(qr[0])
    circ.cx(qr[0], qr[1])
    circ.cx(qr[1], qr[2])
    circ.append(sub_inst, [qr[1], qr[2]])
    circ.append(sub_inst, [qr[0], qr[2]])
    circ.append(sub_inst, [qr[0], qr[1]])
    return circ


def get_qiskit_nested_circuit():
    qr = qiskit.QuantumRegister(6, 'q')
    circ = qiskit.QuantumCircuit(qr)
    sub_ins = get_qiskit_composite_circuit().to_instruction()
    circ.append(sub_ins, [qr[0], qr[2], qr[4]])
    circ.append(sub_ins, [qr[1], qr[3], qr[5]])
    circ.cx(qr[0], qr[3])
    circ.cx(qr[1], qr[4])
    circ.cx(qr[2], qr[5])
    return circ


if qiskit:
    qiskit_circuits.append(get_qiskit_composite_circuit())
    qiskit_circuits.append(get_qiskit_nested_circuit())
    for n_qubits in N_QUBITS_RANGE:
        qiskit_circuits.append(get_qiskit_qft_circuit(n_qubits))
        for depth in DEPTH_RANGE:
            qiskit_circuits.append(get_qiskit_random_circuit(n_qubits, depth))


###################################################################
#
# Simulator APIs inside cirq and qiskit may be subject to change.
# Version tests are needed. In cases where simulator API changes,
# the implementatitons to be modified are: 
# `CirqTest._get_state_vector_from_simulator` and 
# `QiskitTest._get_state_vector_from_simulator`
#
###################################################################

class BaseTester:
    def __init__(self, circuit, dtype, backend, nsample, nsite_max, nfix_max):
        self.circuit = circuit
        self.converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend)
        self.backend = backend
        self.qubits = self.converter.qubits
        self.n_qubits = self.converter.n_qubits
        self.dtype = dtype
        self.sv = None
        self.nsample = nsample
        self.nsite_max = max(1, min(nsite_max, self.n_qubits-1))
        self.nfix_max = max(min(nfix_max, self.n_qubits-nsite_max-1), 0)
        
    def get_state_vector_from_simulator(self):
        if self.sv is None:
            self.sv = self._get_state_vector_from_simulator()
        return self.sv
    
    def get_amplitude_from_simulator(self, bitstring):
        sv = self.get_state_vector_from_simulator()
        index = [int(ibit) for ibit in bitstring]
        return sv[tuple(index)]
    
    def get_batched_amplitudes_from_simulator(self, fixed):
        sv = self.get_state_vector_from_simulator()
        partial_indices = get_partial_indices(self.qubits, fixed)
        batched_amplitudes = sv[tuple(partial_indices)]
        return batched_amplitudes.reshape((2,)*(self.n_qubits-len(fixed)))
    
    def get_reduced_density_matrix_from_simulator(self, where, fixed=EMPTY_DICT):
        r"""
        For where = (a, b), reduced density matrix is formulated as:
        :math: `rho_{a,b,a^{\prime},b^{\prime}}  = \sum_{c,d,e,...} SV^{\star}_{a^{\prime}, b^{\prime}, c, d, e, ...} SV_{a, b, c, d, e, ...}`
        """
        sv = self.get_state_vector_from_simulator()
        partial_indices = get_partial_indices(self.qubits, fixed)
        sv = sv[tuple(partial_indices)]
        
        qubits_map = gen_qubits_map(self.qubits)
        output_inds = ''.join([qubits_map[q] for q in where])
        output_inds += output_inds.upper()
        left_inds = ''.join([qubits_map[q] for q in self.qubits])
        right_inds = ''
        for q in self.qubits:
            if q in where:
                right_inds += qubits_map[q].upper()
            else:
                right_inds += qubits_map[q]
        expression = left_inds + ',' + right_inds + '->' + output_inds
        if self.backend is torch:
            rdm = contract(expression, sv, sv.conj().resolve_conj())
        else:
            rdm = contract(expression, sv, sv.conj())
        return rdm
    
    def get_expectation_from_sv(self, pauli_string):
        
        input_mode_labels = [[*range(self.n_qubits)]]
        qubits_frontier = dict(zip(self.qubits, itertools.count()))
        next_frontier = max(qubits_frontier.values()) + 1

        pauli_map = dict(zip(self.qubits, pauli_string))
        dtype = getattr(self.backend, self.dtype)
        pauli_gates = get_pauli_gates(pauli_map, dtype=dtype, backend=self.backend)
        gate_mode_labels, gate_operands = parse_gates_to_mode_labels_operands(pauli_gates, 
                                                                              qubits_frontier, 
                                                                              next_frontier)

        mode_labels = input_mode_labels + gate_mode_labels + [[qubits_frontier[ix] for ix in self.qubits]]
        output_mode_labels = []
        expression = convert_mode_labels_to_expression(mode_labels, output_mode_labels)

        sv = self.get_state_vector_from_simulator()
        if self.backend is torch:
            operands = [sv] + gate_operands + [sv.conj().resolve_conj()]
        else:
            operands = [sv] + gate_operands + [sv.conj()]
        expec = contract(expression, *operands)
        return expec

    def _get_state_vector_from_simulator(self):
        raise NotImplementedError
                
    def test_state_vector(self):
        expression, operands = self.converter.state_vector()
        sv1 = contract(expression, *operands)
        sv2 = self.get_state_vector_from_simulator()
        self.backend.allclose(
            sv1, sv2, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
    
    def test_amplitude(self):
        for bitstring in bitstring_generator(self.n_qubits, self.nsample):    
            expression, operands = self.converter.amplitude(bitstring)
            amp1 = contract(expression, *operands)
            amp2 = self.get_amplitude_from_simulator(bitstring)
            self.backend.allclose(
                amp1, amp2, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
    
    def test_batched_amplitudes(self):
        for fixed in where_fixed_generator(self.qubits, self.nfix_max):
            expression, operands = self.converter.batched_amplitudes(fixed)
            batched_amps1 = contract(expression, *operands)
            batched_amps2 = self.get_batched_amplitudes_from_simulator(fixed)
            self.backend.allclose(
                batched_amps1, batched_amps2, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
    
    def test_reduced_density_matrix(self):
        for where, fixed in where_fixed_generator(self.qubits, self.nfix_max, nsite_max=self.nsite_max):
            expression1, operands1 = self.converter.reduced_density_matrix(where, fixed=fixed, lightcone=True)
            expression2, operands2 = self.converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)
            assert len(operands1) <= len(operands2)            
            rdm1 = contract(expression1, *operands1)
            rdm2 = contract(expression2, *operands2)
            rdm3 = self.get_reduced_density_matrix_from_simulator(where, fixed=fixed)

            self.backend.allclose(
                rdm1, rdm2, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
            self.backend.allclose(
                rdm1, rdm3, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
    
    def test_expectation(self):
        for pauli_string in random_pauli_string_generator(self.n_qubits, 2):
            expression1, operands1 = self.converter.expectation(pauli_string, lightcone=True)
            expression2, operands2 = self.converter.expectation(pauli_string, lightcone=False)
            assert len(operands1) <= len(operands2)
            expec1 = contract(expression1, *operands1)
            expec2 = contract(expression2, *operands2)
            expec3 = self.get_expectation_from_sv(pauli_string)

            self.backend.allclose(
                expec1, expec2, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
            self.backend.allclose(
                expec1, expec3, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])

    def run_tests(self):
        self.test_state_vector()
        self.test_amplitude()
        self.test_batched_amplitudes()
        self.test_reduced_density_matrix()
        self.test_expectation()


class CirqTester(BaseTester):
    def _get_state_vector_from_simulator(self):
        qubits = self.qubits
        simulator = cirq.Simulator(dtype=self.dtype)
        result = simulator.simulate(self.circuit, qubit_order=qubits)
        statevector = result.state_vector().reshape((2,)*self.n_qubits)
        if self.backend is torch:
            statevector = torch.as_tensor(statevector, dtype=getattr(torch, self.dtype), device='cuda')
        else:
            statevector = self.backend.asarray(statevector, dtype=self.dtype)
        return statevector


class QiskitTester(BaseTester):    
    def _get_state_vector_from_simulator(self):
        # requires qiskit >= 0.24.0
        precision = {'complex64': 'single',
                     'complex128': 'double'}[self.dtype]
        try:
            # for qiskit >= 0.25.0
            simulator = qiskit.Aer.get_backend('aer_simulator_statevector', precision=precision)
            circuit = qiskit.transpile(self.circuit, simulator)
            circuit.save_statevector()
            result = simulator.run(circuit).result()
        except:
            # for qiskit 0.24.*
            circuit = self.circuit
            simulator = qiskit.Aer.get_backend('statevector_simulator', precision=precision)
            result = qiskit.execute(circuit, simulator).result()
        sv = np.asarray(result.get_statevector()).reshape((2,)*circuit.num_qubits)
        # statevector returned by qiskit's simulator is labelled by the inverse of :attr:`qiskit.QuantumCircuit.qubits`
        # this is different from `cirq` and different from the implementation in :class:`CircuitToEinsum`
        sv = sv.transpose(list(range(circuit.num_qubits))[::-1])
        if self.backend is torch:
            sv = torch.as_tensor(sv, dtype=getattr(torch, self.dtype), device='cuda')
        else:
            sv = self.backend.asarray(sv, dtype=self.dtype)
        return sv
