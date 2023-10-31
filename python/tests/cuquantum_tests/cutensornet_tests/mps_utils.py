# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# Note: This file must be self-contained and not import private helpers!

from dataclasses import asdict, dataclass
import importlib
import logging
from types import MappingProxyType
from typing import Optional

try:
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
import opt_einsum as oe
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
except ImportError:
    torch = None

from cuquantum import CircuitToEinsum
from cuquantum.cutensornet._internal.circuit_converter_utils import get_pauli_gates
from cuquantum.cutensornet._internal.utils import infer_object_package
from cuquantum.cutensornet._internal.tensor_wrapper import _get_backend_asarray_func
from .approxTN_utils import tensor_decompose, gate_decompose, SVD_TOLERANCE, verify_unitary
from .test_utils import gen_rand_svd_method, atol_mapper, rtol_mapper

####################################################
################# Helper functions #################
####################################################


EMPTY_DICT = MappingProxyType(dict())

def get_partial_indices(n, fixed=EMPTY_DICT):
    partial_indices = [slice(None)] * n
    index_map = {'0': slice(0, 1),
                 '1': slice(1, 2)}
    for q, val in fixed.items():
        partial_indices[q] = index_map[val]
    return tuple(partial_indices)


def reduced_density_matrix_from_sv(sv, where, fixed=EMPTY_DICT):
    n = sv.ndim
    sv = sv[get_partial_indices(n, fixed)]
    bra_modes = list(range(n))
    ket_modes = [i+n if i in where else i for i in range(n)]
    output_modes = list(where) + [i+n for i in where]
    if infer_object_package(sv) is torch:
        inputs = [sv, bra_modes, sv.conj().resolve_conj(), ket_modes]
    else:
        inputs = [sv, bra_modes, sv.conj(), ket_modes]
    inputs.append(output_modes)
    return oe.contract(*inputs)


def batched_amplitude_from_sv(sv, fixed):
    n = sv.ndim
    sv = sv[get_partial_indices(n, fixed)]
    return sv.reshape([2,]* (n-len(fixed)))


def amplitude_from_sv(sv, bitstring):
    index = [int(ibit) for ibit in bitstring]
    return sv[tuple(index)]


def expectation_from_sv(sv, pauli_string):
    n = sv.ndim
    pauli_map = dict(zip(range(n), pauli_string))
    backend = importlib.import_module(infer_object_package(sv))
    pauli_gates = get_pauli_gates(pauli_map, dtype=sv.dtype, backend=backend)
    # tentative bra/ket indices
    if backend is torch:
        inputs = [sv, list(range(n)), sv.conj().resolve_conj(), list(range(n))]
    else:
        inputs = [sv, list(range(n)), sv.conj(), list(range(n))]
    for o, qs in pauli_gates:
        q = qs[0]
        inputs[3][q] += n # update ket indices
        inputs.extend([o, [q+n, q]])
    return oe.contract(*inputs)


def sample_from_sv(sv, nshots, modes_to_sample=None, seed=None):
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
        # NOTE: bug here
        transpose_order = [sorted_modes_to_sample.index(q) for q in modes_to_sample]
        p = p.transpose(*transpose_order)
    # normalize
    p /= p.sum()
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.choice(np.arange(p.size), p=p.flat, size=nshots)
    hist_sv = np.unique(samples, return_counts=True)
    return dict(zip(*hist_sv))


def gen_random_mps(n_qubits, backend, rng, dtype, D=None):
    assert backend in (cp, np), "backend not supported"
    assert dtype in ('complex64', 'complex128'), f"dtype {dtype} not supported"
    real_dtype = 'float32' if dtype == 'complex64' else 'float64'
    mps_tensors = []
    for i in range(n_qubits):
        next_D = D if D is not None else rng.integers(1, 8)
        if i == 0:
            shape = (2, next_D)
        elif i == n_qubits - 1:
            shape = (prev_D, 2)
        else:
            shape = (prev_D, 2, next_D)
        t = rng.random(shape, dtype=real_dtype) + 1j * rng.random(shape, dtype=real_dtype)
        t = backend.asarray(t, order=rng.choice(['C', 'F']))
        t /= backend.linalg.norm(t)
        mps_tensors.append(t)
        prev_D = next_D
    return mps_tensors


def get_mps_tolerance(dtype):
    tolerance = {'rtol': rtol_mapper[dtype],
                 'atol': atol_mapper[dtype]}
    if dtype in ('float64', 'complex128'):
        # for double precision, relax the tolerance
        tolerance['rtol'] += SVD_TOLERANCE[dtype] ** .5
        tolerance['atol'] += SVD_TOLERANCE[dtype] ** .5
    else:
        tolerance['rtol'] += SVD_TOLERANCE[dtype]
        tolerance['atol'] += SVD_TOLERANCE[dtype]
    return tolerance


@dataclass
class MPSConfig:
    """Class for MPS simulation."""
    # final state
    canonical_center: Optional[int] = None
    # svd options
    max_extent: Optional[int] = None
    abs_cutoff: Optional[float] = 0
    rel_cutoff: Optional[float] = 0
    discarded_weight_cutoff: Optional[float] = 0
    normalization: Optional[str] = None
    algorithm: Optional[str] = 'gesvd'
    gesvdj_tol: Optional[float] = 0
    gesvdj_max_sweeps: Optional[int] = 0
    gesvdr_oversampling: Optional[int] = 0
    gesvdr_niters: Optional[int] = 0

    def __post_init__(self):
        # to be parsed to reference MPS implementation, algorithm and params not supported
        self.svd_options = {'max_extent': self.max_extent,
                            'abs_cutoff': self.abs_cutoff,
                            'rel_cutoff': self.rel_cutoff,
                            'discarded_weight_cutoff': self.discarded_weight_cutoff,
                            'normalization': self.normalization,
                            'partition': 'UV'} # must be enforced to UV partition
    
    @staticmethod
    def rand(n_qubits, rng, dtype, fixed=None, dict_format=True):
        config = dict()
        config['canonical_center'] = rng.integers(0, high=n_qubits)
        svd_method = asdict(gen_rand_svd_method(rng, dtype, fixed=fixed))
        # MPS simulation does not take allow partition other than 'UV'
        svd_method.pop('partition') 
        config.update(svd_method)
        # if found exact MPS simulation setting, shrink it down to truncated extent
        if config['max_extent'] >= 2**(n_qubits//2):
            config['max_extent'] = rng.integers(1, high=2**(n_qubits//2))
        if dict_format:
            return config
        else:
            return MPSConfig(**config)


class MPS:

    def __init__(
        self, 
        mps_tensors, 
        qubits=None,
        **mps_config
    ):
        self.n = len(mps_tensors)
        # avoid in-place modification
        self.mps_tensors = mps_tensors.copy()
        # potentially insert dummy labels for boundary tensors for consistent notation in this class
        if self.mps_tensors[0].ndim == 2:
            self.mps_tensors[0] = self.mps_tensors[0].reshape(1, *self.mps_tensors[0].shape)
        if self.mps_tensors[-1].ndim == 2:
            new_shape = self.mps_tensors[-1].shape + (1, ) 
            self.mps_tensors[-1] = self.mps_tensors[-1].reshape(*new_shape)
        self.qubits = qubits
        self.dtype = mps_tensors[0].dtype.name
        self.sv = None
        self.norm = None
        self.backend = importlib.import_module(infer_object_package(mps_tensors[0]))
        self.swap_gate = None
        self.mps_config = MPSConfig(**mps_config)
        self._tolerance = get_mps_tolerance(self.dtype)
    
    @property
    def tolerance(self):
        return self._tolerance
    
    def setup_resources(self, *args, **kwargs):
        pass
    
    def get_swap_gate(self):
        if self.swap_gate is None:
            asarray = _get_backend_asarray_func(self.backend)
            self.swap_gate = asarray([[1,0,0,0],
                                      [0,0,1,0],
                                      [0,1,0,0],
                                      [0,0,0,1]], dtype=self.dtype).reshape(2,2,2,2)
        return self.swap_gate
    
    def __getitem__(self, key):
        assert key >= 0 and key < self.n
        return self.mps_tensors[key]
    
    def __setitem__(self, key, val):
        assert key>=0 and key < self.n
        self.mps_tensors[key] = val
        # resetting SV and norm
        self.sv = self.norm = None
    
    def get_norm(self):
        if self.norm is None:
            self.norm = self.backend.linalg.norm(self.get_sv()) ** 2
        return self.norm
            
    def get_sv(self):
        if self.sv is None:
            inputs = []
            output_modes = []
            for i, o in enumerate(self.mps_tensors):
                modes = [2*i, 2*i+1, 2*i+2]
                inputs.extend([o, modes])
                output_modes.append(2*i+1)
            inputs.append(output_modes)
            self.sv = oe.contract(*inputs)
        return self.sv
    
    def get_amplitude(self, bitstring):
        return amplitude_from_sv(self.get_sv(), bitstring)
    
    def get_batched_amplitudes(self, fixed=EMPTY_DICT):
        if self.qubits is not None:
            _fixed = dict([(self.qubits.index(q), bit) for q, bit in fixed.items()])
        else:
            _fix = fixed
        return batched_amplitude_from_sv(self.get_sv(), fixed=_fixed)
    
    def get_reduced_density_matrix(self, where, fixed=EMPTY_DICT):
        if self.qubits is not None and not isinstance(where[0], int):
            _where = [self.qubits.index(q) for q in where]
            _fixed = dict([(self.qubits.index(q), bit) for q, bit in fixed.items()])
        else:
            _where = where
            _fixed = fixed
        return reduced_density_matrix_from_sv(self.get_sv(), _where, fixed=_fixed)
    
    def get_expectation(self, pauli_string):
        return expectation_from_sv(self.get_sv(), pauli_string)
    
    def get_sampling(self, qubits_to_sample=None, seed=None, nshots=5000):
        if qubits_to_sample is None:
            _qubits_to_sample = None
        else:
            _qubits_to_sample = [self.qubits.index(q) for q in qubits_to_sample]
        return sample_from_sv(self.get_sv(), nshots, modes_to_sample=_qubits_to_sample, seed=seed)
    
    def _apply_gate_1q(self, i, operand):
        self[i] = self.backend.einsum('ipj,Pp->iPj', self[i], operand)
    
    def _apply_gate_2q(self, i, j, operand):
        if i > j:
            return self._apply_gate_2q(j, i, operand.transpose(1,0,3,2))
        elif i == j:
            raise ValueError(f"gate acting on the same site {i} twice")
        elif i == j - 1:
            self[i], _, self[j] = gate_decompose('ipj,jqk,PQpq->iPj,jQk', self[i], self[j], operand, **self.mps_config.svd_options)
        else:
            # insert swap gates recursively
            swap_gate = self.get_swap_gate()
            if (j - i) % 2 == 0:
                self._apply_gate_2q(i, i+1, swap_gate)
                self._apply_gate_2q(i+1, j, operand)
                self._apply_gate_2q(i, i+1, swap_gate)
            else:
                self._apply_gate_2q(j-1, j, swap_gate)
                self._apply_gate_2q(i, j-1, operand)
                self._apply_gate_2q(j-1, j, swap_gate)
    
    def apply_gate(self, sites, operand):
        if len(sites) == 1:
            return self._apply_gate_1q(*sites, operand)
        elif len(sites) == 2:
            return self._apply_gate_2q(*sites, operand)
        else:
            raise NotImplementedError("Only single- and two- qubit gate supported")
    
    @staticmethod
    def from_converter(converter, initial_state=None, **mps_config):
        if initial_state is None:
            asarray = _get_backend_asarray_func(converter.backend)
            t = asarray([1,0], dtype=converter.dtype).reshape(1,2,1)
            initial_state = [t, ] * len(converter.qubits)
        mps = MPS(initial_state, qubits=list(converter.qubits), **mps_config)
        for operand, qs in converter.gates:
            sites = [converter.qubits.index(q) for q in qs]
            mps.apply_gate(sites, operand)
        mps.canonicalize()
        return mps
    
    def print(self):
        print([o.shape[2] for o in self.mps_tensors[:-1]])
    
    def canonicalize(self):
        center = self.mps_config.canonical_center
        if center is None:
            return
        max_extent = self.mps_config.max_extent
        svd_method = self.mps_config.svd_options.copy()
        svd_method['partition'] = 'V'
        for i in range(center):
            shared_extent = self[i+1].shape[0]
            if max_extent is not None and shared_extent > max_extent:
                self[i], r = tensor_decompose('ipj->ipx,xj', self[i], method='svd', **svd_method)
            else:
                self[i], r = tensor_decompose('ipj->ipx,xj', self[i], method='qr')
            self[i+1] = self.backend.einsum('xj,jqk->xqk', r, self[i+1])
        for i in range(self.n-1, center, -1):
            shared_extent = self[i].shape[0]
            if max_extent is not None and shared_extent > max_extent:
                self[i], r = tensor_decompose('ipj->xpj,ix', self[i], method='svd', **svd_method)
            else:
                self[i], r = tensor_decompose('ipj->xpj,ix', self[i], method='qr')
            self[i-1] = self.backend.einsum('mqi,ix->mqx', self[i-1], r)
    
    def check_canonicalization(self):
        center = self.mps_config.canonical_center
        if center is None:
            return
        modes = 'ipj'
        for i in range(self.n):
            if i < center:
                shared_mode = 'j'
            elif i > center:
                shared_mode = 'i'
            else:
                continue
            verify_unitary(self[i], modes, shared_mode, 
                SVD_TOLERANCE[self.dtype], tensor_name=f"Site {i} canonicalization")


if __name__ == '__main__':
    from cuquantum_benchmarks.frontends.frontend_qiskit import Qiskit as cuqnt_qiskit
    from cuquantum_benchmarks.benchmarks import qpe, quantum_volume, qaoa, random
    from cuquantum import contract
    generators = [qpe.QPE, quantum_volume.QuantumVolume, qaoa.QAOA]
    config = {'measure': True, 'unfold': True, 'p': 4}
    n_qubits = 8
    nshots = 10000
    
    # exact MPS for reference
    mps_config = {'abs_cutoff':1e-8, 'rel_cutoff':1e-5, 'canonical_center': 2}
    for generator in generators:
        seq = generator.generateGatesSequence(n_qubits, config)
        circuit = cuqnt_qiskit(n_qubits, config).generateCircuit(seq)
        converter = CircuitToEinsum(circuit)
        expr, operands = converter.state_vector()
        sv = contract(expr, *operands)
        mps0 = MPS.from_converter(converter, **mps_config)
        mps1 = MPS.from_converter(converter, max_extent=8, **mps_config)

        mps0.check_canonicalization()
        mps1.check_canonicalization()
        sv0 = mps0.get_sv()
        sv1 = mps1.get_sv() / mps1.get_norm()
        samples = sample_from_sv(sv, nshots, seed=1)
        samples0 = mps0.get_sampling(seed=1, nshots=nshots)
        samples1 = mps1.get_sampling(seed=1, nshots=nshots)

        print("Exact MPS bonds:")
        mps0.print()
        print("MPS bonds with max extent 8")
        mps1.print()
        print(f"Exact MPS SV error: {abs(sv0-sv).max()}; Approximate MPS SV error: {abs(sv1-sv).max()}")
        sv_ovlp = abs(cp.dot(sv.ravel(), sv1.ravel().conj()))
        print(f"approx MPS sv overlap: {sv_ovlp}")
    