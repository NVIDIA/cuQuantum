# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import Counter
import itertools
from types import MappingProxyType

try:
    import cirq
except ImportError:
    cirq = None
import cupy as cp
import numpy as np
import pytest
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
except ImportError:
    torch = None
try:
    import qiskit
except ImportError:
    qiskit = None
    
from cuquantum import contract, CircuitToEinsum
from cuquantum import cutensornet as cutn
from cuquantum.cutensornet._internal import circuit_parser_utils_cirq, circuit_parser_utils_qiskit
from cuquantum.cutensornet._internal.circuit_converter_utils import convert_mode_labels_to_expression
from cuquantum.cutensornet._internal.circuit_converter_utils import EINSUM_SYMBOLS_BASE
from cuquantum.cutensornet._internal.circuit_converter_utils import get_pauli_gates
from cuquantum.cutensornet._internal.circuit_converter_utils import parse_gates_to_mode_labels_operands
from cuquantum.cutensornet._internal.utils import infer_object_package

from .test_utils import atol_mapper, get_stream_for_backend, rtol_mapper
from .test_cutensornet import manage_resource


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
        for _ in range(2):
            np.random.shuffle(indices)
            fixed_sites = [qubits[indices[ix]] for ix in range(nfix)]
            bitstring = ''.join(np.random.choice(('0', '1'), nfix))
            fixed = dict(zip(fixed_sites, bitstring))
            if nsite_max is None:
                yield fixed
            else:
                for nsite in range(1, nsite_max+1):
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

try:
    from cuquantum_benchmarks.frontends.frontend_cirq import Cirq as cuqnt_cirq
    from cuquantum_benchmarks.benchmarks import qpe, quantum_volume, qaoa
    cirq_generators = [qpe.QPE, quantum_volume.QuantumVolume, qaoa.QAOA]
    config = {'measure': True, 'unfold': True, 'p': 4}
    for generator in cirq_generators:
        for n_qubits in (5, 6):
            seq = generator.generateGatesSequence(n_qubits, config)
            circuit = cuqnt_cirq(n_qubits, config).generateCircuit(seq)
            cirq_circuits.append(circuit)
except:
    pass


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


def get_cc_unitary_gate(seed=None):
    # random unitary two qubit gate
    from qiskit.extensions import UnitaryGate
    if seed is None:
        seed = 1234
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(size=(4, 4)) + 1j*rng.standard_normal(size=(4, 4))
    q, r = np.linalg.qr(m)
    d = np.diag(r)
    q *= d/abs(d)
    gate = UnitaryGate(q).control(2)
    return gate


def get_qiskit_multi_control_circuit():
    qubits = qiskit.QuantumRegister(5)
    circuit = qiskit.QuantumCircuit(qubits)
    for q in qubits:
        circuit.h(q)
    qs = list(qubits)
    # 3 layers of multi-controlled qubits
    np.random.seed(0)
    for i in range(2):
        np.random.shuffle(qs)
        ccu_gate = get_cc_unitary_gate(i)
        circuit.append(ccu_gate, qs[:4])
        for q in qubits:
            if i % 2 == 1:
                circuit.h(q)
            else:
                circuit.x(q)
    circuit.global_phase = 0.5
    circuit.p(0.1, qubits[0])
    return circuit


if qiskit:
    circuit = get_qiskit_composite_circuit()
    qiskit_circuits.append(circuit.copy())
    circuit.global_phase = 0.5
    qiskit_circuits.append(circuit)
    qiskit_circuits.append(get_qiskit_nested_circuit())
    qiskit_circuits.append(get_qiskit_multi_control_circuit())
    for n_qubits in N_QUBITS_RANGE:
        qiskit_circuits.append(get_qiskit_qft_circuit(n_qubits))
        for depth in DEPTH_RANGE:
            qiskit_circuits.append(get_qiskit_random_circuit(n_qubits, depth))

try:
    from cuquantum_benchmarks.frontends.frontend_qiskit import Qiskit as cuqnt_qiskit
    from cuquantum_benchmarks.benchmarks import qpe, quantum_volume, qaoa
    qiskit_generators = [qpe.QPE, quantum_volume.QuantumVolume, qaoa.QAOA]
    config = {'measure': True, 'unfold': True, 'p': 4}
    for generator in qiskit_generators:
        for n_qubits in (5, 6):
            seq = generator.generateGatesSequence(n_qubits, config)
            circuit = cuqnt_qiskit(n_qubits, config).generateCircuit(seq)
            qiskit_circuits.append(circuit)
except:
    pass


def compute_histogram_overlap(hist1, hist2, nshots):
    # assuming hist1 & hist2 have the same sample size (=nshots)
    overlap = 0
    for val, count in hist1.items():
        if val not in hist2:
            continue
        overlap += min(hist1[val], hist2[val])
    overlap /= nshots
    return overlap


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
    def __init__(self, circuit, dtype, backend, nsample, nsite_max, nfix_max, nshots=5000, seed=1024):
        self.circuit = circuit
        self.converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend)
        self.backend = backend
        self.qubits = list(self.converter.qubits)
        self.n_qubits = self.converter.n_qubits
        self.dtype = dtype
        self.sv = None
        self.nsample = nsample
        self.nsite_max = max(1, min(nsite_max, self.n_qubits-1))
        self.nfix_max = max(min(nfix_max, self.n_qubits-nsite_max-1), 0)
        self.nshots = nshots
        self.seed = seed
        self.state_purity = cutn.StatePurity.PURE
        self.state_prepared = False
        
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

    def _get_sampling_from_simulator(self, qubits_to_sample=None, seed=None):
        raise NotImplementedError
    
    def get_sampling_from_sv(self, qubits_to_sample=None, seed=None):
        sv = self.get_state_vector_from_simulator()
        p = abs(sv) ** 2
        # convert p to double type in case probs does not add up to 1
        if self.backend is np:
            p = p.astype('float64')
        elif self.backend is cp:
            p = cp.asnumpy(p).astype('float64')
        elif self.backend is torch:
            if p.device.type == 'cpu':
                p = p.numpy().astype('float64')
            else:
                p = p.cpu().numpy().astype('float64')
        if qubits_to_sample is not None:
            sorted_qubits_to_sample = [q for q in self.qubits if q in qubits_to_sample]
            axis = [i for (i, q) in enumerate(self.qubits) if q not in qubits_to_sample]
            if axis:
                p = p.sum(tuple(axis))
                # potential transpose to match the order of qubits_to_sample
                transpose_order = [sorted_qubits_to_sample.index(q) for q in qubits_to_sample]
                p = p.transpose(*transpose_order)
        # normalize
        p /= p.sum()
        if seed is not None:
            np.random.seed(seed)
        samples = np.random.choice(np.arange(p.size), p=p.flat, size=self.nshots)
        hist_sv = np.unique(samples, return_counts=True)
        return dict(zip(*hist_sv))
    
    def maybe_prepare_state(self):
        if not self.state_prepared:
            if not hasattr(self, 'state'):
                raise RuntimeError("state not initialized")
            if self.backend is not cp:
                raise RuntimeError("This func is only expected to be executed for cupy backend")
            gates = self.converter.gates
            immutable = 0
            adjoint = 0
            unitary = 1 # assuming all gates unitary
            self.operands = []
            for (operand, qubits) in gates:
                n_state_modes = len(qubits)
                state_modes = [self.qubits.index(q) for q in qubits]
                # keep operand alive otherwise cupy will re-use the memory space
                operand = operand.T.astype(operand.dtype, order=np.random.choice(['C', 'F']))
                self.operands.append(operand)
                tensor_mode_strides = [stride_in_bytes//operand.itemsize for stride_in_bytes in operand.strides]
                update_tensor = np.random.choice([True, False], p=[0.1, 0.9])
                if update_tensor:
                    tmp = cp.empty_like(operand)
                    tensor_id = cutn.state_apply_tensor(self.handle, self.state, n_state_modes, 
                        state_modes, tmp.data.ptr, tensor_mode_strides, 
                        immutable, adjoint, unitary)
                    cutn.state_update_tensor(self.handle, self.state, tensor_id, operand.data.ptr, unitary)
                else:
                    cutn.state_apply_tensor(self.handle, self.state, n_state_modes, 
                        state_modes, operand.data.ptr, tensor_mode_strides, 
                        immutable, adjoint, unitary)
            self.state_prepared = True

    def _run_cutensornet_sampling_marginal(self, task, create_args, execute_args, stream):
        self.maybe_prepare_state()
        if task == 'marginal':
            create_func = cutn.create_marginal
            configure_func = cutn.marginal_configure
            hyper_sample_attr = cutn.MarginalAttribute.OPT_NUM_HYPER_SAMPLES
            num_hyper_samples_dtype = cutn.marginal_get_attribute_dtype(hyper_sample_attr)
            prepare_func = cutn.marginal_prepare
            execute_func = cutn.marginal_compute
            destroy_func = cutn.destroy_marginal
        elif task == 'sampler':
            create_func = cutn.create_sampler
            configure_func = cutn.sampler_configure
            hyper_sample_attr = cutn.SamplerAttribute.OPT_NUM_HYPER_SAMPLES
            num_hyper_samples_dtype = cutn.sampler_get_attribute_dtype(hyper_sample_attr)
            prepare_func = cutn.sampler_prepare
            execute_func = cutn.sampler_sample
            destroy_func = cutn.destroy_sampler
        else:
            raise ValueError("only supports marginal and sampler")
        
        dev = cp.cuda.Device()
        free_mem = dev.mem_info[0]
        scratch_size = free_mem // 2 # maximal usage of 50% device memory

        task_obj = create_func(self.handle, self.state, *create_args)
        num_hyper_samples = np.asarray(8, dtype=num_hyper_samples_dtype)
        configure_func(self.handle, task_obj, hyper_sample_attr, 
            num_hyper_samples.ctypes.data, num_hyper_samples.dtype.itemsize)
        prepare_func(self.handle, task_obj, scratch_size, self.workspace, stream.ptr) # similar args for marginal and sampler
        workspace_size_d = cutn.workspace_get_memory_size(self.handle, 
                        self.workspace, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)

        if workspace_size_d >= scratch_size:
            destroy_func(task_obj)
            return None

        scratch_space = cp.cuda.alloc(workspace_size_d)
        cutn.workspace_set_memory(self.handle, 
            self.workspace, cutn.Memspace.DEVICE, 
            cutn.WorkspaceKind.SCRATCH, scratch_space.ptr, workspace_size_d)
        
        execute_func(self.handle, task_obj, *execute_args, stream.ptr)
        stream.synchronize()
        destroy_func(task_obj)
        return True
    
    def get_reduced_density_matrix_from_cutn(self, where, fixed=EMPTY_DICT):
        n_marginal_modes = len(where)
        marginal_modes = [self.qubits.index(q) for q in where]
        if fixed:
            n_projected_modes = len(fixed)
            projected_modes = []
            projected_mode_values = []
            for q, bit in fixed.items():
                projected_modes.append(self.qubits.index(q))
                projected_mode_values.append(int(bit))
        else:
            n_projected_modes = projected_modes = projected_mode_values = 0
        
        rdm = cp.empty((2,2)*n_marginal_modes, dtype=self.dtype, order=np.random.choice(['C', 'F']))
        rdm_strides = [s // rdm.itemsize for s in rdm.strides]
        stream = cp.cuda.get_current_stream()

        create_args = (n_marginal_modes, marginal_modes, n_projected_modes, projected_modes, rdm_strides)
        execute_args = (projected_mode_values, self.workspace, rdm.data.ptr)
        if self._run_cutensornet_sampling_marginal('marginal', create_args, execute_args, stream):
            return rdm
        else:
            return None
 
    def get_sampling_from_cutensornet(self, qubits_to_sample=None, seed=None):
        if qubits_to_sample is None:
            qubits_to_sample = self.qubits
        n_modes_to_sample = len(qubits_to_sample)
        modes_to_sample = [self.qubits.index(q) for q in qubits_to_sample]
        samples = np.empty((self.nshots, n_modes_to_sample), dtype='int64', order='C') # equivalent to (n_modes, nshots) in F order
        stream = cp.cuda.get_current_stream()

        create_args = (n_modes_to_sample, modes_to_sample)
        execute_args = (self.nshots, self.workspace, samples.ctypes.data)
        if self._run_cutensornet_sampling_marginal('sampler', create_args, execute_args, stream):
            sampling = {}
            for bitstring, n_sampling in zip(*np.unique(samples, axis=0, return_counts=True)):
                bitstring = np.array2string(bitstring, separator='')[1:-1]
                sampling[int(bitstring, 2)] = n_sampling
            return sampling
        else:
            return None
    
    def test_qubits(self):
        assert len(self.qubits) == self.n_qubits
    
    def test_gates(self):
        for (gate_operand, qubits) in self.converter.gates:
            assert gate_operand.ndim == len(qubits) * 2
            assert infer_object_package(gate_operand) == self.backend.__name__
    
    def test_state_vector(self):
        expression, operands = self.converter.state_vector()
        sv1 = contract(expression, *operands)
        sv2 = self.get_state_vector_from_simulator()
        assert self.backend.allclose(
            sv1, sv2, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
    
    def test_amplitude(self):
        for bitstring in bitstring_generator(self.n_qubits, self.nsample):    
            expression, operands = self.converter.amplitude(bitstring)
            amp1 = contract(expression, *operands)
            amp2 = self.get_amplitude_from_simulator(bitstring)
            assert self.backend.allclose(
                amp1, amp2, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
    
    def test_batched_amplitudes(self):
        for fixed in where_fixed_generator(self.qubits, self.nfix_max):
            expression, operands = self.converter.batched_amplitudes(fixed)
            batched_amps1 = contract(expression, *operands)
            batched_amps2 = self.get_batched_amplitudes_from_simulator(fixed)
            assert self.backend.allclose(
                batched_amps1, batched_amps2, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
    
    def test_reduced_density_matrix(self):
        for where, fixed in where_fixed_generator(self.qubits, self.nfix_max, nsite_max=self.nsite_max):
            expression1, operands1 = self.converter.reduced_density_matrix(where, fixed=fixed, lightcone=True)
            expression2, operands2 = self.converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)
            assert len(operands1) <= len(operands2) + 2 # potential phase handling for qiskit Circuit            
            rdm1 = contract(expression1, *operands1)
            rdm2 = contract(expression2, *operands2)
            rdm3 = self.get_reduced_density_matrix_from_simulator(where, fixed=fixed)

            assert self.backend.allclose(
                rdm1, rdm2, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
            assert self.backend.allclose(
                rdm1, rdm3, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
            
            if self.backend is cp:
                rdm4 = self.get_reduced_density_matrix_from_cutn(where, fixed=fixed)
                if rdm4 is not None:
                    assert self.backend.allclose(
                        rdm1, rdm4, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
    
    def test_expectation(self):
        for pauli_string in random_pauli_string_generator(self.n_qubits, 2):
            expression1, operands1 = self.converter.expectation(pauli_string, lightcone=True)
            expression2, operands2 = self.converter.expectation(pauli_string, lightcone=False)
            assert len(operands1) <= len(operands2) + 2 # potential phase handling for qiskit Circuit
            expec1 = contract(expression1, *operands1)
            expec2 = contract(expression2, *operands2)
            expec3 = self.get_expectation_from_sv(pauli_string)

            assert self.backend.allclose(
                expec1, expec2, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
            assert self.backend.allclose(
                expec1, expec3, atol=atol_mapper[self.dtype], rtol=rtol_mapper[self.dtype])
    
    def test_sampling(self):
        full_qubits = list(self.qubits)
        np.random.shuffle(full_qubits)
        selected_qubits = full_qubits[:len(full_qubits)//2]

        for qubits_to_sample in (None, selected_qubits):
            seed = self.seed
            nshots = self.nshots
            max_try = 3
            overlap_best = 0.

            for counter in range(1, max_try+1):
                # build a histogram for the reference impl
                hist_ref = self._get_sampling_from_simulator(qubits_to_sample=qubits_to_sample, seed=seed)

                # do the same for cutensornet sampling
                hist_cutn = self.get_sampling_from_cutensornet(qubits_to_sample=qubits_to_sample, seed=seed)

                # compute overlap of the histograms (cutn vs ref)
                overlap = compute_histogram_overlap(hist_cutn, hist_ref, self.nshots)
                if overlap > overlap_best:
                    overlap_best = overlap
                else:
                    print("WARNING: overlap not improving as nshots increases!")

                # do the same for sampling from the (exactly computed) SV
                hist_sv = self.get_sampling_from_sv(qubits_to_sample=qubits_to_sample, seed=seed)

                # compute overlap of the histograms (sv vs ref)
                overlap_check = compute_histogram_overlap(hist_sv, hist_ref, self.nshots)
                print(f"with nshots = {self.nshots}, {overlap_best = }, {overlap_check = }")

                # to reduce test time we set 95% here, but 99% will also work
                if np.round(overlap, decimals=2) < 0.95:
                    self.nshots *= 10
                    print(f"retry with nshots = {self.nshots} ...")
                else:
                    self.nshots = nshots  # restore
                    break
            else:
                self.nshots = nshots  # restore
                assert False, f"{overlap_best=} after {counter} retries..."

    @manage_resource("handle")
    @manage_resource("state")
    @manage_resource("workspace")
    def run_tests(self):
        self.test_state_vector()
        self.test_amplitude()
        self.test_batched_amplitudes()
        self.test_reduced_density_matrix()
        self.test_expectation()
        self.test_gates()
        self.test_qubits()
        if self.backend is cp:
            # sampling only needed to be tested for cupy backend
            self.test_sampling()


class CirqTester(BaseTester):
    def _get_state_vector_from_simulator(self):
        qubits = self.qubits
        simulator = cirq.Simulator(dtype=self.dtype)
        circuit = circuit_parser_utils_cirq.remove_measurements(self.circuit)
        result = simulator.simulate(circuit, qubit_order=qubits)
        statevector = result.state_vector().reshape((2,)*self.n_qubits)
        if self.backend is torch:
            statevector = torch.as_tensor(statevector, dtype=getattr(torch, self.dtype), device='cuda')
        else:
            statevector = self.backend.asarray(statevector, dtype=self.dtype)
        return statevector
    
    def _get_sampling_from_simulator(self, qubits_to_sample=None, seed=None):
        if qubits_to_sample is None:
            qubits_to_sample = list(self.qubits)
        circuit = circuit_parser_utils_cirq.remove_measurements(self.circuit)
        circuit.append(cirq.measure_each(qubits_to_sample))
        circuit.append(cirq.measure(*qubits_to_sample, key='meas'))
        result = cirq.sample(
            circuit, repetitions=self.nshots, seed=seed, dtype=getattr(np, self.dtype))
        result = result.histogram(key='meas')
        sampling = {}
        nsamples = 0
        for bitstring, nsample in result.items():
            sampling[int(bitstring)] = nsample
            nsamples += nsample
        assert nsamples == self.nshots
        return sampling


class QiskitTester(BaseTester):    
    def _get_precision(self):
        precision = {'complex64': 'single',
                     'complex128': 'double'}[self.dtype]
        return precision
    
    def _get_state_vector_from_simulator(self):
        # requires qiskit >= 0.24.0
        precision = self._get_precision()
        circuit = circuit_parser_utils_qiskit.remove_measurements(self.circuit)
        try:
            # for qiskit >= 0.25.0
            simulator = qiskit.Aer.get_backend('aer_simulator_statevector', precision=precision)
            circuit = qiskit.transpile(circuit, simulator)
            circuit.save_statevector()
            result = simulator.run(circuit).result()
        except:
            # for qiskit 0.24.*
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
    
    def _get_sampling_from_simulator(self, qubits_to_sample=None, seed=None):
        if qubits_to_sample is None:
            qubits_to_sample = list(self.qubits)
        circuit = self.circuit.remove_final_measurements(inplace=False)
        new_creg = circuit._create_creg(len(qubits_to_sample), "meas")
        circuit.add_register(new_creg)
        circuit.measure(qubits_to_sample, new_creg)
        precision = self._get_precision()
        backend = qiskit.Aer.get_backend('qasm_simulator', precision=precision)
        result = backend.run(qiskit.transpile(circuit, backend), shots=self.nshots, seed=seed).result()
        counts  = result.get_counts(circuit)
        sampling = {}
        nsamples = 0
        for bitstring, nsample in counts.items():
            # little endian from qiskit
            value = int(bitstring[::-1], 2)
            sampling[value] = nsample
            nsamples += nsample
        assert nsamples == self.nshots
        return sampling
