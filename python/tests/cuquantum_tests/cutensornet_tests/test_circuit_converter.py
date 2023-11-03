# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import cupy as cp
from cuquantum import CircuitToEinsum

from .circuit_utils import ApproximateMPSTester, backends, CircuitToEinsumTester
from .circuit_utils import cirq_circuits, cirq_circuits_mps
from .circuit_utils import qiskit_circuits, qiskit_circuits_mps
from .circuit_utils import GLOBAL_RNG, is_converter_mps_compatible
from .mps_utils import MPSConfig


class TestCircuitToEinsum:
    # If PyTorch/Qiskit/Cirq is not installed, the corresponding tests are silently
    # skipped.

    @pytest.mark.parametrize("circuit", cirq_circuits + qiskit_circuits)
    @pytest.mark.parametrize("dtype", ('complex64', 'complex128',))
    @pytest.mark.parametrize("backend", backends)
    def test_circuit_converter(self, circuit, dtype, backend, nsample=3, nsite_max=3, nfix_max=3):
        # Results from CircuitToEinsum are compared with Cirq/Qiskit
        # If the backend is set to cupy, additional references below are also tested: 
        #       1. Tensor network simulation based on cutensornet state APIs if backend is cupy
        #       2. State vector simulation based on cutensornet state APIs
        #       3. Exact MPS simulation based on cutensornet state APIs if no mulit-qubit gates exist in the circuit
        #       4. Exact MPS simulation based on a reference cupy implementation in `mps_utils.MPS` if no multi-qubit gates exist in the circuit
        circuit_tests = CircuitToEinsumTester(circuit, dtype, backend, nsample, nsite_max, nfix_max)
        circuit_tests.run_tests()

class TestMPSStateAPIs:

    @pytest.mark.parametrize("circuit", cirq_circuits_mps + qiskit_circuits_mps)
    def test_exact_mps(self, circuit, nsamples=3, nsite_max=3, nfix_max=3):
        # Computation results from approaches below are compared:
        #       1. Exact MPS simulation based on cutensornet state APIs if no mulit-qubit gates exist in the circuit
        #       2. Exact MPS simulation based on a reference cupy-gesvd implementation 
        # Here we only perform exact MPS simulation for double precision as 
        # different SVD algorithms may lead to drastically different results for single precision circuits
        converter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)
        n_qubits = converter.n_qubits
        if not is_converter_mps_compatible(converter):
            pytest.skip("MPS test skipped due to multi-qubit gate")
        
        mps_options = {'algorithm': GLOBAL_RNG.choice(('gesvd', 'gesvdr', 'gesvdp', 'gesvdj'))}
        exact_mps_tests = ApproximateMPSTester(converter, nsamples, nsite_max, nfix_max, **mps_options)
        exact_mps_tests.run_tests()
    
    @pytest.mark.parametrize("circuit", cirq_circuits_mps + qiskit_circuits_mps)
    @pytest.mark.parametrize("dtype", ('complex64', 'complex128',))
    def test_approximate_mps(self, circuit, dtype, nsamples=3, nsite_max=3, nfix_max=3):
        # Computation results from approaches below are compared:
        #       1. Approximate MPS simulation based on cutensornet state APIs if no mulit-qubit gates exist in the circuit
        #       2. Approximate MPS simulation based on a reference cupy implementation in `mps_utils.MPS` if no multi-qubit gates exist in the circuit
        converter = CircuitToEinsum(circuit, dtype=dtype, backend=cp)
        n_qubits = converter.n_qubits
        if not is_converter_mps_compatible(converter):
            pytest.skip("MPS test skipped due to multi-qubit gate")
        
        # test two different types of randomly generated MPS options
        for _ in range(2):
            # restrict to gesvd algorithm to avoid accuracy fallout
            mps_options = MPSConfig.rand(n_qubits, GLOBAL_RNG, dtype, fixed={'algorithm': 'gesvd'}, dict_format=True)
            approximate_mps_tests = ApproximateMPSTester(converter, nsamples, nsite_max, nfix_max, **mps_options)
            approximate_mps_tests.run_tests()
