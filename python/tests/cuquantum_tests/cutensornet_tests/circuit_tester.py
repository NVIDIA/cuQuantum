# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

try:
    import cirq
except ImportError:
    cirq = None
try:
    import qiskit
except ImportError:
    qiskit = None

from cuquantum import CircuitToEinsum
from cuquantum.cutensornet.experimental import NetworkState, MPSConfig
from cuquantum.cutensornet._internal.utils import infer_object_package

from .circuit_utils import CirqComputeEngine, ConverterComputeEngine, QiskitComputeEngine, get_contraction_tolerance, get_mps_tolerance, probablity_from_sv
from .test_utils import DEFAULT_RNG

def get_engine_iters(engine):
    """For NetworkState, compute the same property twice to ensure cache mechanism is correctly activated"""
    if isinstance(engine, NetworkState):
        return range(2)
    return range(1)


def bitstring_generator(n_qubits, num_bitstrings, rng=DEFAULT_RNG, state_dims=None):
    if state_dims is None:
        state_dims = (2, ) * n_qubits
    assert len(state_dims) == n_qubits
    for _ in range(num_bitstrings):
        bitstring = ''.join([str(rng.integers(0, dim)) for dim in state_dims])
        yield bitstring


def where_fixed_generator(qubits, num_fix_sites_max, num_rdm_sites_max=None, rng=DEFAULT_RNG, state_dims=None):
    n_qubits = len(qubits)
    if state_dims is None:
        state_dims = (2, ) * n_qubits
    assert len(state_dims) == n_qubits
    indices = np.arange(len(qubits))
    for nfix in range(num_fix_sites_max):
        for _ in range(2):
            rng.shuffle(indices)
            fixed_sites = [qubits[indices[ix]] for ix in range(nfix)]
            fixed_dims = [state_dims[indices[ix]] for ix in range(nfix)]
            bitstring = ''.join([str(rng.integers(0, dim)) for dim in fixed_dims])
            fixed = dict(zip(fixed_sites, bitstring))
            if num_rdm_sites_max is None:
                yield fixed
            else:
                for nsite in range(1, num_rdm_sites_max+1):
                    where = [qubits[indices[ix]] for ix in range(nfix, nfix+nsite)]
                    yield where, fixed


def get_random_pauli_strings(n, num_pauli_strings, rng=DEFAULT_RNG):
    def _get_pauli_string():
        return ''.join(rng.choice(['I','X', 'Y', 'Z'], n))
    
    if num_pauli_strings is None:
        return _get_pauli_string()
    else:
        # return in dictionary format
        pauli_strings = {}
        for _ in range(num_pauli_strings):
            pauli_string = _get_pauli_string()
            coeff = rng.random() + rng.random() * 1j
            if pauli_string in pauli_strings:
                pauli_strings[pauli_string] += coeff
            else:
                pauli_strings[pauli_string] = coeff
        return pauli_strings


def compute_sample_overlap(samples, sv, modes_to_sample):
    p = probablity_from_sv(sv, modes_to_sample)
    distribution = np.zeros(p.shape, dtype=p.dtype)
    for bitstring, count in samples.items():
        index = tuple(int(i) for i in bitstring)
        distribution[index] = count
    nshots = distribution.sum()
    distribution /= nshots
    ovlp = np.minimum(p, distribution).sum()
    return ovlp


class BaseTester:
    
    def __init__(self,
                 reference_engine,
                 target_engines,
                 converter=None,
                 state_dims=None,
                 num_tests_per_task=3, 
                 num_rdm_sites_max=3, 
                 num_fix_sites_max=3, 
                 nshots=5000,
                 rng=DEFAULT_RNG
    ):
        self.reference_engine = reference_engine
        self.target_engines = target_engines
        self.converter = converter
        self.backend = reference_engine.backend
        self.qubits = reference_engine.qubits
        self.n_qubits = len(self.qubits)
        if state_dims is None:
            state_dims = (2, ) * self.n_qubits
        self.state_dims = state_dims

        self.num_tests_per_task = num_tests_per_task
        self.num_rdm_sites_max = max(1, min(num_rdm_sites_max, self.n_qubits-1))
        self.num_fix_sites_max = max(min(num_fix_sites_max, self.n_qubits-num_rdm_sites_max-1), 0)
        self.nshots = nshots
        self.rng = rng

        for engine in self.all_engines:
            if isinstance(engine, NetworkState):
                if isinstance(engine.config, MPSConfig):
                    tolerance = get_mps_tolerance(engine.dtype)
                else:
                    tolerance = get_contraction_tolerance(engine.dtype)
                setattr(engine, 'tolerance', tolerance)
    
    @property
    def all_engines(self):
        return [self.reference_engine] + self.target_engines
    
    @property
    def is_qubit_system(self):
        return set(self.state_dims) == {2}
    
    def test_misc(self):
        raise NotImplementedError
    
    def test_norm(self):
        norm1 = self.reference_engine.compute_norm()
        for engine in self.target_engines:
            for _ in get_engine_iters(engine):
                norm2 = engine.compute_norm()
                message = f"{engine.__class__.__name__} maxDiff={abs(norm1-norm2)}"
                assert np.allclose(norm1, norm2, **engine.tolerance), message
    
    def test_state_vector(self):
        sv1 = self.reference_engine.compute_state_vector()
        for engine in self.target_engines:
            for _ in get_engine_iters(engine):
                sv2 = engine.compute_state_vector()
                message = f"{engine.__class__.__name__} maxDiff={abs(sv1-sv2).max()}"
                assert self.backend.allclose(sv1, sv2, **engine.tolerance), message
    
    
    def test_amplitude(self):
        for bitstring in bitstring_generator(self.n_qubits, self.num_tests_per_task):    
            amp1 = self.reference_engine.compute_amplitude(bitstring)
            for engine in self.target_engines:
                amp2 = engine.compute_amplitude(bitstring)
                message = f"{engine.__class__.__name__} diff={abs(amp1-amp2)}"
                assert np.allclose(amp1, amp2, **engine.tolerance), message
    
    def test_batched_amplitudes(self):
        for fixed in where_fixed_generator(self.qubits, self.num_fix_sites_max):
            batched_amps1 = self.reference_engine.compute_batched_amplitudes(fixed)
            for engine in self.target_engines:
                for _ in get_engine_iters(engine):
                    batched_amps2 = engine.compute_batched_amplitudes(fixed)
                    message = f"{engine.__class__.__name__} maxDiff={abs(batched_amps1-batched_amps2).max()}"
                    assert self.backend.allclose(batched_amps1, batched_amps2, **engine.tolerance), message
    
    def test_reduced_density_matrix(self):
        for where, fixed in where_fixed_generator(self.qubits, self.num_fix_sites_max, num_rdm_sites_max=self.num_rdm_sites_max):
            if self.converter is not None:
                operands1 = self.converter.reduced_density_matrix(where, fixed=fixed, lightcone=True)[1]
                operands2 = self.converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)[1]
                assert len(operands1) <= len(operands2) + 2 # potential phase handling for qiskit Circuit            
            
            rdm1 = self.reference_engine.compute_reduced_density_matrix(where, fixed=fixed)
            # comparision with different references
            for engine in self.target_engines:
                all_kwargs = ({'lightcone': True}, {'lightcone': False}) if isinstance(engine, ConverterComputeEngine) else ({},)
                for kwargs in all_kwargs:
                    for _ in get_engine_iters(engine):
                        rdm2 = engine.compute_reduced_density_matrix(where, fixed=fixed, **kwargs)
                        message = f"{engine.__class__.__name__} maxDiff={abs(rdm1-rdm2).max()}"
                        assert self.backend.allclose(rdm1, rdm2, **engine.tolerance), message
    
    def test_expectation(self):
        for pauli_string in get_random_pauli_strings(self.n_qubits, 6):
            if self.converter is not None:
                operands1 = self.converter.expectation(pauli_string, lightcone=True)[1]
                operands2 = self.converter.expectation(pauli_string, lightcone=False)[1]
                assert len(operands1) <= len(operands2) + 2 # potential phase handling for qiskit Circuit
            
            expec1 = self.reference_engine.compute_expectation(pauli_string)
            for engine in self.target_engines:
                for lightcone in (True, False):
                    expec2 = engine.compute_expectation(pauli_string, lightcone=lightcone)
                    message = f"{engine.__class__.__name__} maxDiff={abs(expec1-expec2)}"
                    assert np.allclose(expec1, expec2, **engine.tolerance), message
    
    def test_sampling(self):
        full_qubits = list(self.qubits)
        self.rng.shuffle(full_qubits)
        selected_qubits = full_qubits[:len(full_qubits)//2]
        sv = self.reference_engine.compute_state_vector()

        for engine in self.all_engines:
            for qubits_to_sample in (None, selected_qubits):
                nshots = self.nshots
                max_try = 3
                overlap_best = 0.
                modes_to_sample = None if qubits_to_sample is None else [self.qubits.index(q) for q in qubits_to_sample]

                for counter in range(1, max_try+1):
                    # build the sampling
                    samples = engine.compute_sampling(self.nshots, modes=qubits_to_sample)

                    # compute overlap of the histograms
                    overlap = compute_sample_overlap(samples, sv, modes_to_sample)
                    if overlap > overlap_best:
                        overlap_best = overlap
                    else:
                        print(f"WARNING: overlap not improving {counter=} {overlap_best=} {overlap=} as nshots increases!")
                    
                    # to reduce test time we set 95% here, but 99% will also work
                    if np.round(overlap, decimals=2) < 0.95:
                        self.nshots *= 10
                        print(f"{overlap=}, retry with nshots = {self.nshots} ...")
                    else:
                        print(f"{overlap=} with nshots = {self.nshots}")
                        self.nshots = nshots  # restore
                        break
                else:
                    self.nshots = nshots  # restore
                    assert False, f"{overlap_best=} after {counter} retries..."
    
    def run_tests(self):       
        self.test_state_vector()
        self.test_amplitude()
        self.test_batched_amplitudes()
        self.test_reduced_density_matrix()
        self.test_expectation()
        self.test_norm()
        self.test_misc()
        self.test_sampling()
        # release resources for all compute engines
        for engine in self.all_engines:
            engine.free()


class CircuitToEinsumTester(BaseTester):

    @classmethod
    def from_circuit(cls, circuit, dtype, backend, handle=None, **kwargs):
        converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend)
        # Framework provider as reference
        if qiskit and isinstance(circuit, qiskit.QuantumCircuit):
            reference_engine = QiskitComputeEngine(circuit, backend, dtype=dtype)
        elif cirq and isinstance(circuit, cirq.Circuit):
            reference_engine = CirqComputeEngine(circuit, backend, dtype=dtype)
        else:
            raise ValueError(f"circuit type {type(circuit)} not supported")
        # engines to test on
        target_engines = [ConverterComputeEngine(converter, backend=backend, handle=handle)]
        return cls(reference_engine, target_engines, converter=converter, **kwargs)
    
    def test_misc(self):
        self.test_qubits()
        self.test_gates()
        norm = self.reference_engine.compute_norm()
        assert np.allclose(norm, 1, **self.reference_engine.tolerance)
    
    def test_qubits(self):
        assert len(self.qubits) == self.n_qubits
    
    def test_gates(self):
        for (gate_operand, qubits) in self.converter.gates:
            assert gate_operand.ndim == len(qubits) * 2
            assert infer_object_package(gate_operand) == self.backend.__name__