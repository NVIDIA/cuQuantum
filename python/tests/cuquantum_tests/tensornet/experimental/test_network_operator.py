# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np

from cuquantum.tensornet.experimental import NetworkOperator
from cuquantum.tensornet.experimental._internal.network_state_utils import STATE_SUPPORTED_DTYPE_NAMES

from ..utils.circuit_ifc import CircuitHelper
from ..utils.data import BACKEND_MEMSPACE, ARRAY_BACKENDS
from ..utils.helpers import (
    deselect_invalid_network_operator_tests, 
    deselect_network_operator_from_pauli_string_tests, 
    get_state_internal_backend_device
)
from ._internal.state_factory import get_random_network_operator


# Correctness tests will be performed in TestNetworkState
class TestNetworkOperator:
    @pytest.mark.uncollect_if(func=deselect_invalid_network_operator_tests)
    @pytest.mark.parametrize("backend", BACKEND_MEMSPACE)
    @pytest.mark.parametrize("state_dim_extents",(3, 4, 7, (3, 2, 4, 5), (4, 5, 2, 3, 2)))
    @pytest.mark.parametrize("dtype", STATE_SUPPORTED_DTYPE_NAMES)
    @pytest.mark.parametrize('device_id', (None, 0, 2))
    def test_network_operator(self, backend, state_dim_extents, dtype, device_id):
        if isinstance(state_dim_extents, int):
            state_dim_extents = (2, ) * state_dim_extents
        network_operator = get_random_network_operator(state_dim_extents, 
            backend=backend, rng=np.random.default_rng(2), num_repeats=2, dtype=dtype, options={'device_id': device_id})
        expected_backend, expected_device = get_state_internal_backend_device(backend, device_id)
        for tensors, _, _ in network_operator.mpos + network_operator.tensor_products:
            for o in tensors:
                assert (o.name, o.device_id) == (expected_backend, expected_device)
    
    @pytest.mark.uncollect_if(func=deselect_network_operator_from_pauli_string_tests)
    @pytest.mark.parametrize("backend", BACKEND_MEMSPACE)
    @pytest.mark.parametrize("n_qubits", (3, 4, 5, 8, 12))
    @pytest.mark.parametrize("num_pauli_strings", (None, 1, 4))
    @pytest.mark.parametrize("dtype", STATE_SUPPORTED_DTYPE_NAMES)
    @pytest.mark.parametrize('device_id', (None, 0, 2))
    def test_from_pauli_strings(self, backend, n_qubits, num_pauli_strings, dtype, device_id):
        expected_backend, expected_device = get_state_internal_backend_device(backend, device_id)
        if backend == 'torch-gpu':
            backend = 'torch'
        pauli_strings = CircuitHelper.get_random_pauli_strings(n_qubits, num_pauli_strings, np.random.default_rng(4))
        network_operator = NetworkOperator.from_pauli_strings(pauli_strings, dtype=dtype, backend=backend, options={'device_id': device_id})
        expected_device = 0 if device_id is None else device_id
        for tensors, _, _ in network_operator.mpos + network_operator.tensor_products:
            for o in tensors:
                assert (o.name, o.device_id) == (expected_backend, expected_device)
    
    @pytest.mark.parametrize(
        "kwargs", ({}, {'backend': 'auto'}),
    )
    def test_auto_backend(self, kwargs):
        pauli_strings = {'IX': 0.3, 'ZZ': 0.2, 'XX': 0.1, 'YY': 0.4}
        network_operator = NetworkOperator.from_pauli_strings(pauli_strings, **kwargs)
        expected_backend = "cupy" if "cupy" in ARRAY_BACKENDS else "numpy"
        assert network_operator.backend == expected_backend

        for tensors, _, _ in network_operator.mpos + network_operator.tensor_products:
            for o in tensors:
                assert o.name == expected_backend
    
    @pytest.mark.parametrize("backend", ARRAY_BACKENDS)
    @pytest.mark.parametrize("dtype", ("float32", "float64"))
    def test_real_dtype(self, backend, dtype):
        real_pauli_strings = {'IX': 0.3, 'XZ': 0.2, 'II': 0.1, 'ZZ': 0.4}
        imag_pauli_strings = {'YX': 0.3, 'ZY': 0.2, 'YY': 0.1, 'XX': 0.4}
        network_operator = NetworkOperator.from_pauli_strings(real_pauli_strings, backend=backend, dtype=dtype)
        del network_operator

        with pytest.raises(ValueError) as e:
            NetworkOperator.from_pauli_strings(imag_pauli_strings, backend=backend, dtype=dtype)
        assert "Pauli Y operator" in str(e.value)
