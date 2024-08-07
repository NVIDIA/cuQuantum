# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import cupy as cp
import numpy as np
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
except ImportError:
    torch = None
import opt_einsum as oe

try:
    import cirq
    from cuquantum.cutensornet._internal import circuit_parser_utils_cirq
except ImportError:
    cirq = circuit_parser_utils_cirq = None
try:
    import qiskit
    from qiskit_aer import AerSimulator, QasmSimulator
    from cuquantum.cutensornet._internal import circuit_parser_utils_qiskit
except ImportError:
    qiskit = AerSimulator = QasmSimulator = circuit_parser_utils_qiskit = None

from cuquantum import cutensornet as cutn
from cuquantum import CircuitToEinsum, contract
from cuquantum.cutensornet._internal.utils import infer_object_package
from cuquantum.cutensornet._internal.circuit_converter_utils import get_pauli_gates

from .approxTN_utils import SVD_TOLERANCE
from .test_utils import atol_mapper, get_or_create_tensor_backend, rtol_mapper, DEFAULT_RNG, EMPTY_DICT, get_or_create_tensor_backend, get_dtype_name

# note: this implementation would cause pytorch tests being silently skipped
# if pytorch is not available, which is the desired effect since otherwise
# it'd be too noisy
backends = [np, cp]
if torch:
    backends.append(torch)

def get_contraction_tolerance(dtype):
    tolerance = {'atol': atol_mapper[dtype],
                 'rtol': rtol_mapper[dtype]}
    return tolerance

def get_mps_tolerance(dtype):
    tolerance = get_contraction_tolerance(dtype)
    # relax the tolerance for SVD based simulations
    if dtype in ('float64', 'complex128'):
    # for double precision, relax the tolerance
        tolerance['atol'] += SVD_TOLERANCE[dtype] ** .5
        tolerance['rtol'] += SVD_TOLERANCE[dtype] ** .5
    else:
        tolerance['atol'] += SVD_TOLERANCE[dtype]
        tolerance['rtol'] += SVD_TOLERANCE[dtype]
    return tolerance

def get_partial_indices(n, projected_modes=EMPTY_DICT):
    partial_indices = [slice(None)] * n
    for q, val in projected_modes.items():
        bit = int(val)
        partial_indices[q] = slice(bit, bit+1)
    return tuple(partial_indices)


def reduced_density_matrix_from_sv(sv, modes, projected_modes=EMPTY_DICT):
    n = sv.ndim
    sv = sv[get_partial_indices(n, projected_modes=projected_modes)]
    bra_modes = list(range(n))
    ket_modes = [i+n if i in modes else i for i in range(n)]
    output_modes = list(modes) + [i+n for i in modes]
    if infer_object_package(sv) == 'torch':
        inputs = [sv, bra_modes, sv.conj().resolve_conj(), ket_modes]
    else:
        inputs = [sv, bra_modes, sv.conj(), ket_modes]
    inputs.append(output_modes)
    return oe.contract(*inputs)


def batched_amplitude_from_sv(sv, projected_modes):
    n = sv.ndim
    shape = [sv.shape[i] for i in range(n) if i not in projected_modes]
    sv = sv[get_partial_indices(n, projected_modes)]
    return sv.reshape(shape)


def amplitude_from_sv(sv, bitstring):
    index = [int(ibit) for ibit in bitstring]
    return sv[tuple(index)].item()


def expectation_from_sv(sv, pauli_strings):
    if isinstance(pauli_strings, str):
        n = sv.ndim
        pauli_map = dict(zip(range(n), pauli_strings))
        backend = importlib.import_module(infer_object_package(sv))
        pauli_gates = get_pauli_gates(pauli_map, dtype=sv.dtype, backend=backend)
        # bra/ket indices
        if backend is torch:
            inputs = [sv, list(range(n)), sv.conj().resolve_conj(), list(range(n))]
        else:
            inputs = [sv, list(range(n)), sv.conj(), list(range(n))]
        for o, qs in pauli_gates:
            q = qs[0]
            inputs[3][q] += n # update ket indices
            if infer_object_package(sv) == 'torch' and str(sv.device) == 'cpu':
                o = o.to(device=sv.device)
            inputs.extend([o, [q+n, q]])
        return oe.contract(*inputs).item()
    else:
        value = 0
        for pauli_string, coeff in pauli_strings.items():
            value += expectation_from_sv(sv, pauli_string) * coeff
        return value


def probablity_from_sv(sv, modes_to_sample):
    backend = infer_object_package(sv)
    p = abs(sv) ** 2
    # convert p to double type in case probs does not add up to 1
    if backend == 'numpy':
        p = p.astype('float64')
    elif backend == 'cupy':
        p = cp.asnumpy(p).astype('float64')
    elif backend == 'torch':
        if p.device.type == 'cpu':
            p = p.numpy().astype('float64')
        else:
            p = p.cpu().numpy().astype('float64')
    if modes_to_sample is not None:
        sorted_modes_to_sample = sorted(modes_to_sample)
        axis = [q for q in range(sv.ndim) if q not in modes_to_sample]
        if axis:
            p = p.sum(tuple(axis))
        transpose_order = [sorted_modes_to_sample.index(q) for q in modes_to_sample]
        p = p.transpose(*transpose_order)
    # normalize
    p /= p.sum()
    return p

def sample_from_sv(sv, nshots, modes_to_sample=None, rng=DEFAULT_RNG):
    p = probablity_from_sv(sv, modes_to_sample)
    samples = rng.choice(np.arange(p.size), p=p.flat, size=nshots)
    bitstrings, counts = np.unique(samples, return_counts=True)
    samples = dict()
    for bitstring, count in zip(bitstrings, counts):
        bitstring = ''.join([str(bit) for bit in np.unravel_index(bitstring, p.shape)])
        samples[bitstring] = count
    return samples


class _BaseComputeEngine:
    ###################################################################
    # 
    #      Reference API signatures from all compute engines.
    #
    ###################################################################

    def __init__(self, circuit, backend, dtype='complex128', sample_rng=DEFAULT_RNG):
        self.circuit = circuit
        if not isinstance(backend, str):
            backend = backend.__name__
        self.backend = get_or_create_tensor_backend(backend)
        self.dtype = dtype
        self.sv = None
        self.norm = None
        self._tolerance = None
        self.sample_rng = sample_rng
    
    def free(self):
        pass
    
    @property
    def qubits(self):
        raise NotImplementedError
    
    @property
    def tolerance(self):
        if self._tolerance is None:
            #NOTE: This is default tolerance for contraction based simulation. 
            #For MPS simulations, tolerance needs to be relaxed
            self._tolerance = get_contraction_tolerance(self.dtype)
        return self._tolerance
    
    def _compute_state_vector(self):
        # implementation for different backends
        raise NotImplementedError
    
    def compute_sampling(self, nshots, modes=None):
        modes_to_sample = [self.qubits.index(q) for q in modes] if modes else None
        return sample_from_sv(self.compute_state_vector(), nshots, modes_to_sample=modes_to_sample, rng=self.sample_rng)
    
    def compute_state_vector(self):
        if self.sv is None:
            self.sv = self._compute_state_vector()
        return self.sv
    
    def compute_norm(self):
        if self.norm is None:
            sv = self.compute_state_vector()
            self.norm = self.backend.norm(sv).item() ** 2
        return self.norm
    
    def compute_amplitude(self, bitstring):
        return amplitude_from_sv(self.compute_state_vector(), bitstring)
    
    def compute_batched_amplitudes(self, fixed):
        fixed_modes = dict([(self.qubits.index(q), bit) for q, bit in fixed.items()])
        return batched_amplitude_from_sv(self.compute_state_vector(), fixed_modes)
    
    def compute_reduced_density_matrix(self, where, fixed=EMPTY_DICT):
        sv = self.compute_state_vector()
        modes = [self.qubits.index(q) for q in where]
        projected_modes = dict([(self.qubits.index(q), bit) for q, bit in fixed.items()])
        return reduced_density_matrix_from_sv(sv, modes, projected_modes=projected_modes)
    
    def compute_expectation(self, pauli_string):
        return expectation_from_sv(self.compute_state_vector(), pauli_string)


class CirqComputeEngine(_BaseComputeEngine):

    @property
    def qubits(self):
        return sorted(self.circuit.all_qubits())
    
    def _compute_state_vector(self):
        qubits = self.qubits
        simulator = cirq.Simulator(dtype=self.dtype)
        circuit = circuit_parser_utils_cirq.remove_measurements(self.circuit)
        result = simulator.simulate(circuit, qubit_order=qubits)
        statevector = result.state_vector().reshape((2,)*len(qubits))
        statevector = self.backend.asarray(statevector, dtype=self.dtype)
        return statevector
    
    def compute_sampling(self, nshots, modes=None):
        if modes is None:
            modes = self.qubits
        circuit = circuit_parser_utils_cirq.remove_measurements(self.circuit)
        circuit.append(cirq.measure(*modes, key='meas'))
        result = cirq.sample(
            circuit, repetitions=nshots, seed=int(self.sample_rng.integers(2023)), dtype=getattr(np, self.dtype))
        result = result.histogram(key='meas')
        sampling = {}
        nsamples = 0
        specifier = f"0{len(modes)}b"
        for bitstring, nsample in result.items():
            sampling[format(bitstring, specifier)] = nsample
            nsamples += nsample
        assert nsamples == nshots
        return sampling


class QiskitComputeEngine(_BaseComputeEngine):

    @property
    def qubits(self):
        return list(self.circuit.qubits)

    def _get_precision(self):
        precision = {'complex64': 'single',
                     'complex128': 'double'}[self.dtype]
        return precision
    
    def _compute_state_vector(self):
        precision = self._get_precision()
        circuit = circuit_parser_utils_qiskit.remove_measurements(self.circuit)
        circuit.save_statevector()
        simulator = AerSimulator(precision=precision)
        circuit = qiskit.transpile(circuit, simulator)
        result = simulator.run(circuit).result()
        sv = np.asarray(result.get_statevector()).reshape((2,)*circuit.num_qubits)
        # statevector returned by qiskit's simulator is labelled by the inverse of :attr:`qiskit.QuantumCircuit.qubits`
        # this is different from `cirq` and different from the implementation in :class:`CircuitToEinsum`
        sv = sv.transpose(list(range(circuit.num_qubits))[::-1])
        sv = self.backend.asarray(sv, dtype=self.dtype)
        return sv
    
    def compute_sampling(self, nshots, modes=None):
        if modes is None:
            modes = self.qubits
        circuit = self.circuit.remove_final_measurements(inplace=False)
        new_creg = circuit._create_creg(len(modes), "meas")
        circuit.add_register(new_creg)
        circuit.measure(modes, new_creg)
        precision = self._get_precision()
        backend = QasmSimulator(precision=precision)
        result = backend.run(qiskit.transpile(circuit, backend), shots=nshots, seed=self.sample_rng.integers(2023)).result()
        counts  = result.get_counts(circuit)
        sampling = {}
        nsamples = 0
        for bitstring, nsample in counts.items():
            # little endian from qiskit
            sampling[bitstring[::-1]] = nsample
            nsamples += nsample
        assert nsamples == nshots
        return sampling


class ConverterComputeEngine(_BaseComputeEngine):

    def __init__(self, circuit_or_converter, backend=cp, handle=None, dtype='complex128', sample_rng=DEFAULT_RNG):
        if isinstance(circuit_or_converter, CircuitToEinsum):
            circuit = None
            self.converter = circuit_or_converter
        else:
            circuit = circuit_or_converter
            self.converter = CircuitToEinsum(circuit, backend=backend, dtype=dtype)
        
        dtype_name = get_dtype_name(self.converter.dtype)
        super().__init__(circuit, self.converter.backend, dtype=dtype_name, sample_rng=sample_rng) 
        self._own_handle = handle is None
        if self._own_handle:
            handle = cutn.create()
        self.handle = handle
    
    @property
    def qubits(self):
        return self.converter.qubits
    
    def free(self):
        if self._own_handle and self.handle is not None:
            cutn.destroy(self.handle)
            self.handle = None
    
    def _compute_from_converter(self, task, *args, **kwargs):
        expression, operands = getattr(self.converter, task)(*args, **kwargs)
        out = contract(expression, *operands, options={'handle': self.handle})
        if task in {'amplitude', 'expectation'}:
            out = out.item()
        return out
    
    def _compute_state_vector(self):
        return self._compute_from_converter('state_vector')
    
    def compute_amplitude(self, bitstring):
        return self._compute_from_converter('amplitude', bitstring)
    
    def compute_batched_amplitudes(self, fixed):
        return self._compute_from_converter('batched_amplitudes', fixed)
    
    def compute_reduced_density_matrix(self, where, fixed=EMPTY_DICT, lightcone=True):
        return self._compute_from_converter('reduced_density_matrix', where, fixed=fixed, lightcone=lightcone)
    
    def compute_expectation(self, pauli_strings, lightcone=True):
        if isinstance(pauli_strings, str):
            pauli_strings = {pauli_strings: 1}
        val = 0
        for pauli_string, coeff in pauli_strings.items():
            val += coeff * self._compute_from_converter('expectation', pauli_string, lightcone=lightcone)
        return val