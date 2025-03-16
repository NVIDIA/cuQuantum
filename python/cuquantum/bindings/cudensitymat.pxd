# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.03.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycudensitymat cimport *


###############################################################################
# Types
###############################################################################

ctypedef cudensitymatHandle_t Handle
ctypedef cudensitymatState_t State
ctypedef cudensitymatElementaryOperator_t ElementaryOperator
ctypedef cudensitymatMatrixOperator_t MatrixOperator
ctypedef cudensitymatOperatorTerm_t OperatorTerm
ctypedef cudensitymatOperator_t Operator
ctypedef cudensitymatOperatorAction_t OperatorAction
ctypedef cudensitymatExpectation_t Expectation
ctypedef cudensitymatWorkspaceDescriptor_t WorkspaceDescriptor
ctypedef cudensitymatDistributedRequest_t DistributedRequest
ctypedef cudensitymatTimeRange_t TimeRange
ctypedef cudensitymatDistributedCommunicator_t DistributedCommunicator
ctypedef cudensitymatScalarCallback_t ScalarCallback
ctypedef cudensitymatTensorCallback_t TensorCallback
ctypedef cudensitymatDistributedInterface_t DistributedInterface
ctypedef cudensitymatLoggerCallback_t LoggerCallback
ctypedef cudensitymatLoggerCallbackData_t LoggerCallbackData

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType

cdef class WrappedScalarCallback:
    cdef object callback
    cdef cudensitymatCallbackDevice_t device
    cdef cudensitymatWrappedScalarCallback_t* c_struct

cdef class WrappedTensorCallback:
    cdef object callback
    cdef cudensitymatCallbackDevice_t device
    cdef cudensitymatWrappedTensorCallback_t* c_struct


###############################################################################
# Enum
###############################################################################

ctypedef cudensitymatStatus_t _Status
ctypedef cudensitymatComputeType_t _ComputeType
ctypedef cudensitymatDistributedProvider_t _DistributedProvider
ctypedef cudensitymatCallbackDevice_t _CallbackDevice
ctypedef cudensitymatStatePurity_t _StatePurity
ctypedef cudensitymatElementaryOperatorSparsity_t _ElementaryOperatorSparsity
ctypedef cudensitymatMemspace_t _Memspace
ctypedef cudensitymatWorkspaceKind_t _WorkspaceKind


###############################################################################
# Functions
###############################################################################

cpdef get_version()
cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef reset_distributed_configuration(intptr_t handle, int provider, intptr_t comm_ptr, size_t comm_size)
cpdef int32_t get_num_ranks(intptr_t handle) except? -1
cpdef int32_t get_proc_rank(intptr_t handle) except? -1
cpdef reset_random_seed(intptr_t handle, int32_t random_seed)
cpdef intptr_t create_state(intptr_t handle, int purity, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int data_type) except? 0
cpdef destroy_state(intptr_t state)
cpdef int32_t state_get_num_components(intptr_t handle, intptr_t state) except? -1
cpdef state_attach_component_storage(intptr_t handle, intptr_t state, int32_t num_state_components, component_buffer, component_buffer_size)
cpdef state_get_component_num_modes(intptr_t handle, intptr_t state, int32_t state_component_local_id, intptr_t state_component_global_id, intptr_t state_component_num_modes, intptr_t batch_mode_location)
cpdef state_get_component_info(intptr_t handle, intptr_t state, int32_t state_component_local_id, intptr_t state_component_global_id, intptr_t state_component_num_modes, intptr_t state_component_mode_extents, intptr_t state_component_mode_offsets)
cpdef state_initialize_zero(intptr_t handle, intptr_t state, intptr_t stream)
cpdef state_compute_scaling(intptr_t handle, intptr_t state, intptr_t scaling_factors, intptr_t stream)
cpdef state_compute_norm(intptr_t handle, intptr_t state, intptr_t norm, intptr_t stream)
cpdef state_compute_trace(intptr_t handle, intptr_t state, intptr_t trace, intptr_t stream)
cpdef state_compute_accumulation(intptr_t handle, intptr_t state_in, intptr_t state_out, intptr_t scaling_factors, intptr_t stream)
cpdef state_compute_inner_product(intptr_t handle, intptr_t state_left, intptr_t state_right, intptr_t inner_product, intptr_t stream)
cpdef intptr_t create_operator_term(intptr_t handle, int32_t num_space_modes, space_mode_extents) except? 0
cpdef intptr_t create_operator(intptr_t handle, int32_t num_space_modes, space_mode_extents) except? 0
cpdef operator_prepare_action(intptr_t handle, intptr_t superoperator, intptr_t state_in, intptr_t state_out, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream)
cpdef operator_compute_action(intptr_t handle, intptr_t superoperator, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state_in, intptr_t state_out, intptr_t workspace, intptr_t stream)
cpdef intptr_t create_operator_action(intptr_t handle, int32_t num_operators, operators) except? 0
cpdef destroy_operator_action(intptr_t operator_action)
cpdef operator_action_prepare(intptr_t handle, intptr_t operator_action, state_in, intptr_t state_out, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream)
cpdef operator_action_compute(intptr_t handle, intptr_t operator_action, double time, int64_t batch_size, int32_t num_params, intptr_t params, state_in, intptr_t state_out, intptr_t workspace, intptr_t stream)
cpdef intptr_t create_expectation(intptr_t handle, intptr_t superoperator) except? 0
cpdef destroy_expectation(intptr_t expectation)
cpdef expectation_prepare(intptr_t handle, intptr_t expectation, intptr_t state, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream)
cpdef expectation_compute(intptr_t handle, intptr_t expectation, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state, intptr_t expectation_value, intptr_t workspace, intptr_t stream)
cpdef intptr_t create_workspace(intptr_t handle) except? 0
cpdef destroy_workspace(intptr_t workspace_descr)
cpdef size_t workspace_get_memory_size(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind) except? -1
cpdef workspace_set_memory(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind, intptr_t memory_buffer, size_t memory_buffer_size)
cpdef tuple workspace_get_memory(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind)

###############################################################################

cpdef tuple state_get_component_storage_size(intptr_t handle, intptr_t state, int32_t num_state_components)
cpdef intptr_t create_elementary_operator(intptr_t handle, int32_t num_space_modes, space_mode_extents, int sparsity, int32_t num_diagonals, diagonal_offsets, int data_type, intptr_t tensor_data, wrapped_tensor_callback) except? 0
cpdef intptr_t create_elementary_operator_batch(intptr_t handle, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int sparsity, int32_t num_diagonals, diagonal_offsets, int data_type, intptr_t tensor_data, wrapped_tensor_callback) except? 0
cpdef intptr_t create_matrix_operator_dense_local(intptr_t handle, int32_t num_space_modes, space_mode_extents, int data_type, intptr_t matrix_data, wrapped_matrix_callback) except? 0
cpdef intptr_t create_matrix_operator_dense_local_batch(intptr_t handle, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int data_type, intptr_t matrix_data, wrapped_matrix_callback) except? 0
cpdef operator_term_append_elementary_product(intptr_t handle, intptr_t operator_term, int32_t num_elem_operators, elem_operators, state_modes_acted_on, mode_action_duality, coefficient, wrapped_coefficient_callback)
cpdef operator_term_append_elementary_product_batch(intptr_t handle, intptr_t operator_term, int32_t num_elem_operators, elem_operators, state_modes_acted_on, mode_action_duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, wrapped_coefficient_callback)
cpdef operator_term_append_general_product(intptr_t handle, intptr_t operator_term, int32_t num_elem_operators, num_operator_modes, operator_mode_extents, operator_mode_strides, state_modes_acted_on, mode_action_duality, int data_type, tensor_data, wrapped_tensor_callbacks, coefficient, wrapped_coefficient_callback)
cpdef operator_term_append_matrix_product(intptr_t handle, intptr_t operator_term, int32_t num_matrix_operators, matrix_operators, matrix_conjugation, action_duality, coefficient, wrapped_coefficient_callback)
cpdef operator_term_append_matrix_product_batch(intptr_t handle, intptr_t operator_term, int32_t num_matrix_operators, matrix_operators, matrix_conjugation, action_duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, wrapped_coefficient_callback)
cpdef operator_append_term(intptr_t handle, intptr_t superoperator, intptr_t operator_term, int32_t duality, coefficient, coefficient_callback)
cpdef operator_append_term_batch(intptr_t handle, intptr_t superoperator, intptr_t operator_term, int32_t duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, coefficient_callback)
cpdef destroy_elementary_operator(intptr_t elem_operator)
cpdef destroy_matrix_operator(intptr_t matrix_operator)
cpdef destroy_operator_term(intptr_t operator_term)
cpdef destroy_operator(intptr_t superoperator)
