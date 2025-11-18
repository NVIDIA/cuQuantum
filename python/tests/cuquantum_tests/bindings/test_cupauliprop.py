# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import os

import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None
    pytest.skip("skipping binding tests when cupy is not installed", allow_module_level=True)
from cupy import testing

from nvmath.internal.typemaps import NAME_TO_DATA_TYPE
from cuquantum.bindings import cupauliprop as cupp


###################################################################
#
# Following the test_cutensornet.py philosophy:
# The sole goal is to ensure the Python arguments are properly
# passed to the C level. We do not ensure coverage nor correctness.
#
###################################################################

def manage_resource(name):
    """Decorator to manage resource lifecycle in tests."""
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kwargs):
            try:
                if name == 'handle':
                    h = cupp.create()
                elif name == 'workspace':
                    h = cupp.create_workspace_descriptor(self.handle)
                elif name == 'pauli_expansion':
                    # Create a simple Pauli expansion on device
                    num_qubits = getattr(self, 'num_qubits', 4)
                    num_terms = getattr(self, 'num_terms', 2)
                    dtype = getattr(self, 'dtype', np.complex128)
                    
                    num_packed_ints = cupp.get_num_packed_integers(num_qubits)
                    xz_bits_size = num_terms * 2 * num_packed_ints * np.dtype(np.uint64).itemsize
                    coef_size = num_terms * np.dtype(dtype).itemsize
                    
                    d_xz_bits = cp.cuda.alloc(xz_bits_size)
                    d_coefs = cp.cuda.alloc(coef_size)
                    
                    h = cupp.create_pauli_expansion(
                        self.handle, num_qubits,
                        d_xz_bits.ptr, xz_bits_size,
                        d_coefs.ptr, coef_size,
                        NAME_TO_DATA_TYPE[np.dtype(dtype).name],
                        num_terms, 0, 1)  # not sorted, has duplicates
                    
                    # Keep memory alive
                    self._xz_bits_mem = d_xz_bits
                    self._coefs_mem = d_coefs
                    
                elif name == 'pauli_expansion_view':
                    # Create view of entire expansion
                    num_terms = cupp.pauli_expansion_get_num_terms(
                        self.handle, self.pauli_expansion)
                    h = cupp.pauli_expansion_get_contiguous_range(
                        self.handle, self.pauli_expansion, 0, num_terms)
                    
                elif name == 'clifford_operator':
                    # Create a simple Hadamard gate on qubit 0
                    h = cupp.create_clifford_gate_operator(
                        self.handle, cupp.CliffordGateKind.CLIFFORD_GATE_H, [0])
                    
                elif name == 'pauli_rotation_operator':
                    # Create a Pauli rotation gate
                    angle = getattr(self, 'angle', np.pi / 4)
                    h = cupp.create_pauli_rotation_gate_operator(
                        self.handle, angle, 2, [0, 1],
                        [cupp.PauliKind.PAULI_X, cupp.PauliKind.PAULI_X])
                    
                elif name == 'noise_channel_operator':
                    # Create a single-qubit depolarizing channel
                    probs = [0.9, 0.033, 0.033, 0.034]  # I, X, Y, Z
                    h = cupp.create_pauli_noise_channel_operator(
                        self.handle, 1, [0], probs)
                else:
                    assert False, f'name "{name}" not recognized'
                    
                setattr(self, name, h)
                impl(self, *args, **kwargs)
            except:
                print(f'managing resource {name} failed')
                raise
            finally:
                if name == 'handle' and hasattr(self, name):
                    cupp.destroy(self.handle)
                    del self.handle
                elif name == 'workspace' and hasattr(self, name):
                    cupp.destroy_workspace_descriptor(self.workspace)
                    del self.workspace
                elif name == 'pauli_expansion' and hasattr(self, name):
                    cupp.destroy_pauli_expansion(self.pauli_expansion)
                    del self.pauli_expansion
                    if hasattr(self, '_xz_bits_mem'):
                        del self._xz_bits_mem
                    if hasattr(self, '_coefs_mem'):
                        del self._coefs_mem
                elif name == 'pauli_expansion_view' and hasattr(self, name):
                    cupp.destroy_pauli_expansion_view(self.pauli_expansion_view)
                    del self.pauli_expansion_view
                elif name in ('clifford_operator', 'pauli_rotation_operator', 'noise_channel_operator'):
                    if hasattr(self, name):
                        cupp.destroy_operator(getattr(self, name))
                        delattr(self, name)
        return test_func
    return decorator


class TestLibHelper:
    
    def test_get_version(self):
        ver = cupp.get_version()
        assert isinstance(ver, int)
    
    def test_get_error_string(self):
        msg = cupp.get_error_string(cupp.Status.SUCCESS)
        assert isinstance(msg, str)
    
    def test_get_num_packed_integers(self):
        for num_qubits in [1, 10, 64, 100]:
            num_packed = cupp.get_num_packed_integers(num_qubits)
            assert isinstance(num_packed, int)
            assert num_packed > 0


class TestHandle:
    
    @manage_resource('handle')
    def test_handle_create_destroy(self):
        # Simple round-trip test
        pass
    
    @manage_resource('handle')
    def test_set_stream(self):
        stream = cp.cuda.Stream()
        cupp.set_stream(self.handle, stream.ptr)
        stream.synchronize()


class TestWorkspaceDescriptor:
    
    @manage_resource('handle')
    @manage_resource('workspace')
    def test_workspace_create_destroy(self):
        # Simple round-trip test
        pass
    
    @manage_resource('handle')
    @manage_resource('workspace')
    def test_workspace_set_get_memory(self):
        # Allocate some memory
        mem_size = 1024 * 1024  # 1 MB
        d_mem = cp.cuda.alloc(mem_size)
        
        # Set memory
        cupp.workspace_set_memory(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH,
            d_mem.ptr, mem_size)
        
        # Get memory back
        ptr, size = cupp.workspace_get_memory(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH)
        
        assert ptr == d_mem.ptr
        assert size == mem_size


@testing.parameterize(*testing.product({
    'num_qubits': (4, 8, 16),
    'num_terms': (1, 10, 100),
    'dtype': (np.complex64, np.complex128),
}))
class TestPauliExpansion:
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    def test_pauli_expansion_create_destroy(self):
        # Simple round-trip test
        pass
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    def test_pauli_expansion_get_properties(self):
        # Test all getter functions
        num_qubits = cupp.pauli_expansion_get_num_qubits(
            self.handle, self.pauli_expansion)
        assert num_qubits == self.num_qubits
        
        num_terms = cupp.pauli_expansion_get_num_terms(
            self.handle, self.pauli_expansion)
        assert num_terms == self.num_terms
        
        data_type = cupp.pauli_expansion_get_data_type(
            self.handle, self.pauli_expansion)
        assert data_type == NAME_TO_DATA_TYPE[np.dtype(self.dtype).name]
        
        is_sorted = cupp.pauli_expansion_is_sorted(
            self.handle, self.pauli_expansion)
        assert isinstance(is_sorted, (int, np.integer))
        
        is_dedup = cupp.pauli_expansion_is_deduplicated(
            self.handle, self.pauli_expansion)
        assert isinstance(is_dedup, (int, np.integer))
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    def test_pauli_expansion_get_storage_buffer(self):
        xz_ptr, xz_size, coef_ptr, coef_size, num_terms, location = \
            cupp.pauli_expansion_get_storage_buffer(
                self.handle, self.pauli_expansion)
        
        assert xz_ptr == self._xz_bits_mem.ptr
        assert coef_ptr == self._coefs_mem.ptr
        assert num_terms == self.num_terms
        assert location == cupp.Memspace.DEVICE
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    @manage_resource('pauli_expansion_view')
    def test_pauli_expansion_create_view(self):
        # View is created by decorator
        num_terms = cupp.pauli_expansion_view_get_num_terms(
            self.handle, self.pauli_expansion_view)
        assert num_terms == self.num_terms


@testing.parameterize(*testing.product({
    'dtype': (np.complex64, np.complex128),
}))
class TestPauliExpansionViewOperations:
    
    num_qubits = 4
    num_terms = 10
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    @manage_resource('pauli_expansion_view')
    @manage_resource('workspace')
    def test_view_prepare_deduplication(self):
        max_workspace = 1 << 30  # 1 GB
        cupp.pauli_expansion_view_prepare_deduplication(
            self.handle, self.pauli_expansion_view,
            1,  # make_sorted
            max_workspace, self.workspace)
        
        # Query workspace size
        ws_size = cupp.workspace_get_memory_size(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH)
        assert isinstance(ws_size, (int, np.integer))
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    @manage_resource('pauli_expansion_view')
    @manage_resource('workspace')
    def test_view_prepare_canonical_sort(self):
        max_workspace = 1 << 30
        cupp.pauli_expansion_view_prepare_canonical_sort(
            self.handle, self.pauli_expansion_view,
            max_workspace, self.workspace)
        
        ws_size = cupp.workspace_get_memory_size(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH)
        assert isinstance(ws_size, (int, np.integer))
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    @manage_resource('pauli_expansion_view')
    @manage_resource('workspace')
    def test_view_prepare_trace_with_zero_state(self):
        max_workspace = 1 << 30
        cupp.pauli_expansion_view_prepare_trace_with_zero_state(
            self.handle, self.pauli_expansion_view,
            max_workspace, self.workspace)
        
        ws_size = cupp.workspace_get_memory_size(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH)
        assert isinstance(ws_size, (int, np.integer))


@pytest.mark.parametrize('gate_kind', [
    cupp.CliffordGateKind.CLIFFORD_GATE_H,
    cupp.CliffordGateKind.CLIFFORD_GATE_X,
    cupp.CliffordGateKind.CLIFFORD_GATE_CX,
    cupp.CliffordGateKind.CLIFFORD_GATE_SWAP,
])
class TestCliffordOperators:
    
    @manage_resource('handle')
    def test_clifford_operator_create_destroy(self, gate_kind):
        # Determine number of qubits based on gate
        if gate_kind in (cupp.CliffordGateKind.CLIFFORD_GATE_CX,
                         cupp.CliffordGateKind.CLIFFORD_GATE_CZ,
                         cupp.CliffordGateKind.CLIFFORD_GATE_SWAP):
            qubits = [0, 1]
        else:
            qubits = [0]
        
        oper = cupp.create_clifford_gate_operator(
            self.handle, gate_kind, qubits)
        
        try:
            # Query operator kind
            kind = cupp.quantum_operator_get_kind(self.handle, oper)
            assert kind == cupp.QuantumOperatorKind.EXPANSION_KIND_CLIFFORD_GATE
        finally:
            cupp.destroy_operator(oper)


@pytest.mark.parametrize('angle', [0.0, np.pi/4, np.pi/2, np.pi])
@pytest.mark.parametrize('paulis', [
    [cupp.PauliKind.PAULI_X],
    [cupp.PauliKind.PAULI_Z, cupp.PauliKind.PAULI_Z],
    [cupp.PauliKind.PAULI_X, cupp.PauliKind.PAULI_Y],
])
class TestPauliRotationOperators:
    
    @manage_resource('handle')
    def test_pauli_rotation_create_destroy(self, angle, paulis):
        num_qubits = len(paulis)
        qubits = list(range(num_qubits))
        
        oper = cupp.create_pauli_rotation_gate_operator(
            self.handle, angle, num_qubits, qubits, paulis)
        
        try:
            kind = cupp.quantum_operator_get_kind(self.handle, oper)
            assert kind == cupp.QuantumOperatorKind.EXPANSION_KIND_PAULI_ROTATION_GATE
        finally:
            cupp.destroy_operator(oper)


class TestNoiseChannelOperators:
    
    @manage_resource('handle')
    def test_single_qubit_depolarizing_channel(self):
        probs = [0.9, 0.033, 0.033, 0.034]
        oper = cupp.create_pauli_noise_channel_operator(
            self.handle, 1, [0], probs)
        
        try:
            kind = cupp.quantum_operator_get_kind(self.handle, oper)
            assert kind == cupp.QuantumOperatorKind.EXPANSION_KIND_PAULI_NOISE_CHANNEL
        finally:
            cupp.destroy_operator(oper)
    
    @manage_resource('handle')
    def test_two_qubit_noise_channel(self):
        # 16 probabilities for 2-qubit channel
        probs = [1.0/16] * 16
        oper = cupp.create_pauli_noise_channel_operator(
            self.handle, 2, [0, 1], probs)
        
        try:
            kind = cupp.quantum_operator_get_kind(self.handle, oper)
            assert kind == cupp.QuantumOperatorKind.EXPANSION_KIND_PAULI_NOISE_CHANNEL
        finally:
            cupp.destroy_operator(oper)


@testing.parameterize(*testing.product({
    'dtype': (np.complex64, np.complex128),
}))
class TestOperatorApplicationWorkflow:
    """Test end-to-end operator application workflow."""
    
    num_qubits = 4
    num_terms = 5
    
    @manage_resource('handle')
    @manage_resource('workspace')
    def test_operator_application_workflow(self):
        # Create input expansion
        num_packed_ints = cupp.get_num_packed_integers(self.num_qubits)
        xz_bits_size = self.num_terms * 2 * num_packed_ints * 8
        coef_size = self.num_terms * np.dtype(self.dtype).itemsize
        
        d_xz_in = cp.cuda.alloc(xz_bits_size)
        d_coef_in = cp.cuda.alloc(coef_size)
        
        expansion_in = cupp.create_pauli_expansion(
            self.handle, self.num_qubits,
            d_xz_in.ptr, xz_bits_size,
            d_coef_in.ptr, coef_size,
            NAME_TO_DATA_TYPE[np.dtype(self.dtype).name],
            self.num_terms, 0, 1)
        
        try:
            # Create view
            view_in = cupp.pauli_expansion_get_contiguous_range(
                self.handle, expansion_in, 0, self.num_terms)
            
            try:
                # Create operator (Hadamard on qubit 0)
                oper = cupp.create_clifford_gate_operator(
                    self.handle, cupp.CliffordGateKind.CLIFFORD_GATE_H, [0])
                
                try:
                    # Prepare operator application
                    max_workspace = 1 << 30
                    xz_out_size, coef_out_size = cupp.pauli_expansion_view_prepare_operator_application(
                        self.handle, view_in, oper,
                        0,  # make_sorted
                        1,  # keep_duplicates
                        0,  # num_truncation_strategies
                        None,  # truncation_strategies (no truncation)
                        max_workspace, self.workspace)
                    
                    assert isinstance(xz_out_size, (int, np.integer))
                    assert isinstance(coef_out_size, (int, np.integer))
                    assert xz_out_size > 0
                    assert coef_out_size > 0
                    
                finally:
                    cupp.destroy_operator(oper)
            finally:
                cupp.destroy_pauli_expansion_view(view_in)
        finally:
            cupp.destroy_pauli_expansion(expansion_in)


class TestTruncationParams:
    """Test truncation parameter structures."""
    
    def test_coefficient_truncation_params(self):
        params = cupp.CoefficientTruncationParams()
        params.cutoff = 1e-6
        assert params.cutoff == 1e-6
        assert params.ptr != 0
    
    def test_pauli_weight_truncation_params(self):
        params = cupp.PauliWeightTruncationParams()
        params.cutoff = 10
        assert params.cutoff == 10
        assert params.ptr != 0
    
    def test_truncation_strategy(self):
        # Test creating a truncation strategy struct
        params = cupp.CoefficientTruncationParams()
        params.cutoff = 1e-4
        
        strategy = cupp.TruncationStrategy()
        strategy.strategy = cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_COEFFICIENT_BASED
        strategy.param_struct = params.ptr
        
        assert strategy.strategy == cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_COEFFICIENT_BASED
        assert strategy.param_struct == params.ptr


@testing.parameterize(*testing.product({
    'dtype': (np.complex64, np.complex128),
}))
class TestOperatorApplicationWithTruncation:
    """Test operator application with truncation strategies."""
    
    num_qubits = 4
    num_terms = 5
    
    @manage_resource('handle')
    @manage_resource('workspace')
    def test_operator_application_with_truncation(self):
        # Create input expansion
        num_packed_ints = cupp.get_num_packed_integers(self.num_qubits)
        xz_bits_size = self.num_terms * 2 * num_packed_ints * 8
        coef_size = self.num_terms * np.dtype(self.dtype).itemsize
        
        d_xz_in = cp.cuda.alloc(xz_bits_size)
        d_coef_in = cp.cuda.alloc(coef_size)
        
        expansion_in = cupp.create_pauli_expansion(
            self.handle, self.num_qubits,
            d_xz_in.ptr, xz_bits_size,
            d_coef_in.ptr, coef_size,
            NAME_TO_DATA_TYPE[np.dtype(self.dtype).name],
            self.num_terms, 0, 1)
        
        try:
            # Create view
            view_in = cupp.pauli_expansion_get_contiguous_range(
                self.handle, expansion_in, 0, self.num_terms)
            
            try:
                # Create operator (Hadamard on qubit 0)
                oper = cupp.create_clifford_gate_operator(
                    self.handle, cupp.CliffordGateKind.CLIFFORD_GATE_H, [0])
                
                try:
                    # Create truncation strategies
                    coef_params = cupp.CoefficientTruncationParams()
                    coef_params.cutoff = 1e-4
                    
                    weight_params = cupp.PauliWeightTruncationParams()
                    weight_params.cutoff = 8
                    
                    # Create truncation strategy structs
                    coef_strategy = cupp.TruncationStrategy()
                    coef_strategy.strategy = cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_COEFFICIENT_BASED
                    coef_strategy.param_struct = coef_params.ptr
                    
                    weight_strategy = cupp.TruncationStrategy()
                    weight_strategy.strategy = cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED
                    weight_strategy.param_struct = weight_params.ptr
                    
                    # Create list of strategy objects
                    strategies = [coef_strategy, weight_strategy]
                    
                    # Prepare operator application with truncation
                    max_workspace = 1 << 30
                    xz_out_size, coef_out_size = cupp.pauli_expansion_view_prepare_operator_application(
                        self.handle, view_in, oper,
                        0,  # make_sorted
                        1,  # keep_duplicates
                        2,  # num_truncation_strategies
                        strategies,  # truncation_strategies
                        max_workspace, self.workspace)
                    
                    assert isinstance(xz_out_size, (int, np.integer))
                    assert isinstance(coef_out_size, (int, np.integer))
                    assert xz_out_size > 0
                    assert coef_out_size > 0
                    
                finally:
                    cupp.destroy_operator(oper)
            finally:
                cupp.destroy_pauli_expansion_view(view_in)
        finally:
            cupp.destroy_pauli_expansion(expansion_in)


class TestEnums:
    """Test that all enums are accessible."""
    
    def test_status_enum(self):
        assert hasattr(cupp.Status, 'SUCCESS')
        assert hasattr(cupp.Status, 'INVALID_VALUE')
    
    def test_compute_type_enum(self):
        assert hasattr(cupp.ComputeType, 'COMPUTE_32F')
        assert hasattr(cupp.ComputeType, 'COMPUTE_64F')
    
    def test_memspace_enum(self):
        assert hasattr(cupp.Memspace, 'DEVICE')
        assert hasattr(cupp.Memspace, 'HOST')
    
    def test_pauli_kind_enum(self):
        assert hasattr(cupp.PauliKind, 'PAULI_I')
        assert hasattr(cupp.PauliKind, 'PAULI_X')
        assert hasattr(cupp.PauliKind, 'PAULI_Y')
        assert hasattr(cupp.PauliKind, 'PAULI_Z')
    
    def test_clifford_gate_enum(self):
        assert hasattr(cupp.CliffordGateKind, 'CLIFFORD_GATE_H')
        assert hasattr(cupp.CliffordGateKind, 'CLIFFORD_GATE_CX')
    
    def test_truncation_strategy_kind_enum(self):
        assert hasattr(cupp.TruncationStrategyKind, 'TRUNCATION_STRATEGY_COEFFICIENT_BASED')
        assert hasattr(cupp.TruncationStrategyKind, 'TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED')