# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.11.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycupauliprop cimport *


###############################################################################
# Types
###############################################################################

ctypedef cupaulipropHandle_t Handle
ctypedef cupaulipropWorkspaceDescriptor_t WorkspaceDescriptor
ctypedef cupaulipropPauliExpansion_t PauliExpansion
ctypedef cupaulipropPauliExpansionView_t PauliExpansionView
ctypedef cupaulipropQuantumOperator_t QuantumOperator

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef cupaulipropStatus_t _Status
ctypedef cupaulipropComputeType_t _ComputeType
ctypedef cupaulipropMemspace_t _Memspace
ctypedef cupaulipropWorkspaceKind_t _WorkspaceKind
ctypedef cupaulipropTruncationStrategyKind_t _TruncationStrategyKind
ctypedef cupaulipropPauliKind_t _PauliKind
ctypedef cupaulipropCliffordGateKind_t _CliffordGateKind
ctypedef cupaulipropQuantumOperatorKind_t _QuantumOperatorKind


###############################################################################
# Functions
###############################################################################

cpdef size_t get_version() except? 0
cpdef str get_error_string(int error)
cpdef int32_t get_num_packed_integers(int32_t num_qubits) except? -1
cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef set_stream(intptr_t handle, stream)
cpdef intptr_t create_workspace_descriptor(intptr_t handle) except? 0
cpdef destroy_workspace_descriptor(intptr_t workspace_desc)
cpdef int64_t workspace_get_memory_size(intptr_t handle, intptr_t workspace_desc, int mem_space, int workspace_kind) except? -1
cpdef workspace_set_memory(intptr_t handle, intptr_t workspace_desc, int mem_space, int workspace_kind, intptr_t memory_buffer, int64_t memory_buffer_size)
cpdef tuple workspace_get_memory(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind)
cpdef intptr_t create_pauli_expansion(intptr_t handle, int32_t num_qubits, intptr_t xz_bits_buffer, int64_t xz_bits_buffer_size, intptr_t coef_buffer, int64_t coef_buffer_size, int data_type, int64_t num_terms, int32_t is_sorted, int32_t has_duplicates) except? 0
cpdef destroy_pauli_expansion(intptr_t pauli_expansion)
cpdef tuple pauli_expansion_get_storage_buffer(intptr_t handle, intptr_t pauli_expansion)
cpdef int32_t pauli_expansion_get_num_qubits(intptr_t handle, intptr_t pauli_expansion) except? -1
cpdef int64_t pauli_expansion_get_num_terms(intptr_t handle, intptr_t pauli_expansion) except? -1
cpdef int pauli_expansion_get_data_type(intptr_t handle, intptr_t pauli_expansion) except? -1
cpdef int32_t pauli_expansion_is_sorted(intptr_t handle, intptr_t pauli_expansion) except? -1
cpdef int32_t pauli_expansion_is_deduplicated(intptr_t handle, intptr_t pauli_expansion) except? -1
cpdef intptr_t pauli_expansion_get_contiguous_range(intptr_t handle, intptr_t pauli_expansion, int64_t start_ind_ex, int64_t end_ind_ex) except? 0
cpdef destroy_pauli_expansion_view(intptr_t view)
cpdef int64_t pauli_expansion_view_get_num_terms(intptr_t handle, intptr_t view) except? -1
cpdef int pauli_expansion_view_get_location(intptr_t view) except? -1
cpdef pauli_expansion_view_prepare_deduplication(intptr_t handle, intptr_t view_in, int32_t make_sorted, int64_t max_workspace_size, intptr_t workspace)
cpdef pauli_expansion_view_execute_deduplication(intptr_t handle, intptr_t view_in, intptr_t expansion_out, int32_t make_sorted, intptr_t workspace)
cpdef pauli_expansion_view_prepare_canonical_sort(intptr_t handle, intptr_t view_in, int64_t max_workspace_size, intptr_t workspace)
cpdef pauli_expansion_view_execute_canonical_sort(intptr_t handle, intptr_t view_in, intptr_t expansion_out, intptr_t workspace)
cpdef pauli_expansion_populate_from_view(intptr_t handle, intptr_t view_in, intptr_t expansion_out)
cpdef pauli_expansion_view_prepare_trace_with_expansion_view(intptr_t handle, intptr_t view1, intptr_t view2, int64_t max_workspace_size, intptr_t workspace)
cpdef pauli_expansion_view_compute_trace_with_expansion_view(intptr_t handle, intptr_t view1, intptr_t view2, int32_t take_adjoint1, intptr_t trace, intptr_t workspace)
cpdef pauli_expansion_view_prepare_trace_with_zero_state(intptr_t handle, intptr_t view, int64_t max_workspace_size, intptr_t workspace)
cpdef pauli_expansion_view_compute_trace_with_zero_state(intptr_t handle, intptr_t view, intptr_t trace, intptr_t workspace)
cpdef intptr_t create_clifford_gate_operator(intptr_t handle, int clifford_gate_kind, qubit_indices) except? 0
cpdef intptr_t create_pauli_rotation_gate_operator(intptr_t handle, double angle, int32_t num_qubits, qubit_indices, paulis) except? 0
cpdef intptr_t create_pauli_noise_channel_operator(intptr_t handle, int32_t num_qubits, qubit_indices, probabilities) except? 0
cpdef int quantum_operator_get_kind(intptr_t handle, intptr_t oper) except? -1
cpdef destroy_operator(intptr_t oper)
