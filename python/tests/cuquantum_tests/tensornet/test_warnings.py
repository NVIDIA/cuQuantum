# Copyright (c) 2025-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest

from cuquantum import (
    CircuitToEinsum,
    Network,
    NetworkOptions,
    OptimizerOptions,
    PathFinderOptions,
    ReconfigOptions,
    SlicerOptions,
    contract,
    contract_path,
    einsum,
    einsum_path,
    tensor,
)
from cuquantum.cutensornet.experimental import (
    ContractDecomposeAlgorithm,
    MPSConfig,
    TNConfig,
    NetworkOperator,
    NetworkState,
    contract_decompose,
)
from .utils.circuit_data import testing_circuits



def check_user_warning(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = func(*args, **kwargs)
        assert len(w) == 1
        # make sure one deprecation warning is issued
        assert issubclass(w[0].category, UserWarning)
        return out


class TestContraction:

    @pytest.mark.parametrize("func", [
        contract, einsum, contract_path, einsum_path, Network
    ])
    def test_contraction(self, func):
        a = np.random.random((2,2))
        out = check_user_warning(func, 'ij,jk->ik', a, a)
        if func is Network:
            out.free()
    
    @pytest.mark.parametrize("cls", [
        NetworkOptions, 
        OptimizerOptions, 
        PathFinderOptions, 
        ReconfigOptions, 
        SlicerOptions,
    ])
    def test_options(self, cls):
        check_user_warning(cls)
    
    def test_circuit_to_einsum(self):
        if testing_circuits:
            check_user_warning(CircuitToEinsum, testing_circuits[0])
    

class TestTensor:

    def test_decompose(self):
        a = np.random.random((2,2))
        check_user_warning(tensor.decompose, 'ij->ik,jk', a)
    
    @pytest.mark.parametrize("cls", [
        tensor.DecompositionOptions,
        tensor.QRMethod,
        tensor.SVDMethod,
    ])
    def test_options(self, cls):
        check_user_warning(cls)
    

class TestExperimental:

    def test_contract_decompose(self):
        a = np.random.random((2,2))
        check_user_warning(contract_decompose, 'ij,jk->il,lk', a, a)
    
    def test_network_state(self):
        if testing_circuits:
            state = check_user_warning(NetworkState.from_circuit, testing_circuits[0])
            state.free()
    
    def test_network_operator(self):
        pauli_strings = {
            'IX': 0.2,
            'XY': 0.3,
            'YZ': 0.4,
        }
        check_user_warning(NetworkOperator.from_pauli_strings, pauli_strings)
    
    @pytest.mark.parametrize("cls", [
        ContractDecomposeAlgorithm,
        MPSConfig,
        TNConfig,
    ])
    def test_options(self, cls):
        check_user_warning(cls)