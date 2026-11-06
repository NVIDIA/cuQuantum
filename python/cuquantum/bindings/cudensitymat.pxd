# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 26.06.0 to 26.11.0. Do not modify it directly.


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    int32_t,
    int64_t,
    intptr_t,
)


# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport intptr_t

from .cycudensitymat cimport *


###############################################################################
# Types
###############################################################################

ctypedef cudensitymatHandle_t Handle
ctypedef cudensitymatState_t State
ctypedef cudensitymatElementaryOperator_t ElementaryOperator
ctypedef cudensitymatMatrixOperator_t MatrixOperator
ctypedef cudensitymatMatrixProductOperator_t MatrixProductOperator
ctypedef cudensitymatOperatorTerm_t OperatorTerm
ctypedef cudensitymatOperator_t Operator
ctypedef cudensitymatOperatorAction_t OperatorAction
ctypedef cudensitymatExpectation_t Expectation
ctypedef cudensitymatOperatorSpectrum_t OperatorSpectrum
ctypedef cudensitymatWorkspaceDescriptor_t WorkspaceDescriptor
ctypedef cudensitymatTimePropagation_t TimePropagation
ctypedef cudensitymatEigenDecomposition_t EigenDecomposition
ctypedef cudensitymatTimePropagationApproachKrylovConfig_t TimePropagationApproachKrylovConfig
ctypedef cudensitymatEigenDecompositionApproachKrylovConfig_t EigenDecompositionApproachKrylovConfig
ctypedef cudensitymatTimePropagationScopeSplitTDVPConfig_t TimePropagationScopeSplitTDVPConfig
ctypedef cudensitymatEigenDecompositionScopeSplitDMRGConfig_t EigenDecompositionScopeSplitDMRGConfig
ctypedef cudensitymatEigenDecompositionApproachLinearConfig_t EigenDecompositionApproachLinearConfig
ctypedef cudensitymatStateFittingScopeSplitALSConfig_t StateFittingScopeSplitALSConfig
ctypedef cudensitymatStateFittingApproachLinSolveConfig_t StateFittingApproachLinSolveConfig
ctypedef cudensitymatSVDConfig_t SVDConfig
ctypedef cudensitymatDistributedRequest_t DistributedRequest
ctypedef cudensitymatDistributedCommunicator_t DistributedCommunicator
ctypedef cudensitymatScalarCallback_t ScalarCallback
ctypedef cudensitymatTensorCallback_t TensorCallback
ctypedef cudensitymatScalarGradientCallback_t ScalarGradientCallback
ctypedef cudensitymatTensorGradientCallback_t TensorGradientCallback
ctypedef cudensitymatDistributedInterface_t DistributedInterface
ctypedef cudensitymatLoggerCallback_t LoggerCallback
ctypedef cudensitymatLoggerCallbackData_t LoggerCallbackData
ctypedef cudensitymatWrappedScalarCallback_t _WrappedScalarCallback
ctypedef cudensitymatWrappedTensorCallback_t _WrappedTensorCallback
ctypedef cudensitymatWrappedScalarGradientCallback_t _WrappedScalarGradientCallback
ctypedef cudensitymatWrappedTensorGradientCallback_t _WrappedTensorGradientCallback

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType

cdef class WrappedScalarCallback:
    cdef public object callback
    cdef cudensitymatCallbackDevice_t device
    cdef cudensitymatWrappedScalarCallback_t _struct

cdef class WrappedTensorCallback:
    cdef public object callback
    cdef cudensitymatCallbackDevice_t device
    cdef cudensitymatWrappedTensorCallback_t _struct

cdef class WrappedScalarGradientCallback:
    cdef public object callback
    cdef cudensitymatCallbackDevice_t device
    cdef cudensitymatWrappedScalarGradientCallback_t _struct

cdef class WrappedTensorGradientCallback:
    cdef public object callback
    cdef cudensitymatCallbackDevice_t device
    cdef cudensitymatWrappedTensorGradientCallback_t _struct


###############################################################################
# Enum
###############################################################################

ctypedef cudensitymatStatus_t _Status
ctypedef cudensitymatComputeType_t _ComputeType
ctypedef cudensitymatDistributedProvider_t _DistributedProvider
ctypedef cudensitymatCallbackDevice_t _CallbackDevice
ctypedef cudensitymatDifferentiationDir_t _DifferentiationDir
ctypedef cudensitymatStatePurity_t _StatePurity
ctypedef cudensitymatElementaryOperatorSparsity_t _ElementaryOperatorSparsity
ctypedef cudensitymatOperatorSpectrumKind_t _OperatorSpectrumKind
ctypedef cudensitymatOperatorSpectrumConfig_t _OperatorSpectrumConfig
ctypedef cudensitymatBoundaryCondition_t _BoundaryCondition
ctypedef cudensitymatTimePropagationScopeKind_t _TimePropagationScopeKind
ctypedef cudensitymatEigenDecompositionScopeKind_t _EigenDecompositionScopeKind
ctypedef cudensitymatTimePropagationScopeSplitKind_t _TimePropagationScopeSplitKind
ctypedef cudensitymatEigenDecompositionScopeSplitKind_t _EigenDecompositionScopeSplitKind
ctypedef cudensitymatTimePropagationApproachKind_t _TimePropagationApproachKind
ctypedef cudensitymatEigenDecompositionApproachKind_t _EigenDecompositionApproachKind
ctypedef cudensitymatTimePropagationAttribute_t _TimePropagationAttribute
ctypedef cudensitymatEigenDecompositionAttribute_t _EigenDecompositionAttribute
ctypedef cudensitymatTimePropagationApproachKrylovConfigAttribute_t _TimePropagationApproachKrylovConfigAttribute
ctypedef cudensitymatEigenDecompositionApproachKrylovConfigAttribute_t _EigenDecompositionApproachKrylovConfigAttribute
ctypedef cudensitymatTimePropagationScopeSplitTDVPConfigAttribute_t _TimePropagationScopeSplitTDVPConfigAttribute
ctypedef cudensitymatEigenDecompositionScopeSplitDMRGConfigAttribute_t _EigenDecompositionScopeSplitDMRGConfigAttribute
ctypedef cudensitymatEigenDecompositionApproachLinearConfigAttribute_t _EigenDecompositionApproachLinearConfigAttribute
ctypedef cudensitymatStateFittingScopeKind_t _StateFittingScopeKind
ctypedef cudensitymatStateFittingScopeSplitKind_t _StateFittingScopeSplitKind
ctypedef cudensitymatStateFittingApproachKind_t _StateFittingApproachKind
ctypedef cudensitymatStateFittingAttribute_t _StateFittingAttribute
ctypedef cudensitymatStateFittingScopeSplitALSConfigAttribute_t _StateFittingScopeSplitALSConfigAttribute
ctypedef cudensitymatStateFittingApproachLinSolveConfigAttribute_t _StateFittingApproachLinSolveConfigAttribute
ctypedef cudensitymatSVDConfigAttribute_t _SVDConfigAttribute
ctypedef cudensitymatEigenDecompositionSpectrumKind_t _EigenDecompositionSpectrumKind
ctypedef cudensitymatMemspace_t _Memspace
ctypedef cudensitymatWorkspaceKind_t _WorkspaceKind


###############################################################################
# Functions
###############################################################################

cpdef size_t get_version() except? 0
cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef reset_distributed_configuration(intptr_t handle, int provider, intptr_t comm_ptr, size_t comm_size)
cpdef int32_t get_num_ranks(intptr_t handle) except? -1
cpdef int32_t get_proc_rank(intptr_t handle) except? -1
cpdef reset_random_seed(intptr_t handle, int32_t random_seed)
cpdef intptr_t create_state(intptr_t handle, int purity, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int data_type) except? 0
cpdef intptr_t create_state_mps(intptr_t handle, int purity, int32_t num_space_modes, space_mode_extents, int boundary_condition, bond_extents, int data_type, int64_t batch_size) except? 0
cpdef state_mps_set_current_bond_extents(intptr_t handle, intptr_t state, bond_extents)
cpdef state_mps_get_current_bond_extents(intptr_t handle, intptr_t state, intptr_t bond_extents)
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
cpdef intptr_t create_elementary_operator(intptr_t handle, int32_t num_space_modes, space_mode_extents, int sparsity, int32_t num_diagonals, diagonal_offsets, int data_type, intptr_t tensor_data, tensor_callback, tensor_gradient_callback) except? 0
cpdef intptr_t create_elementary_operator_batch(intptr_t handle, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int sparsity, int32_t num_diagonals, diagonal_offsets, int data_type, intptr_t tensor_data, tensor_callback, tensor_gradient_callback) except? 0
cpdef destroy_elementary_operator(intptr_t elem_operator)
cpdef intptr_t create_matrix_operator_dense_local(intptr_t handle, int32_t num_space_modes, space_mode_extents, int data_type, intptr_t matrix_data, matrix_callback, matrix_gradient_callback) except? 0
cpdef intptr_t create_matrix_operator_dense_local_batch(intptr_t handle, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int data_type, intptr_t matrix_data, matrix_callback, matrix_gradient_callback) except? 0
cpdef destroy_matrix_operator(intptr_t matrix_operator)
cpdef intptr_t create_matrix_product_operator(intptr_t handle, int32_t num_space_modes, space_mode_extents, int boundary_condition, bond_extents, int data_type, tensor_data, tensor_callbacks, tensor_gradient_callbacks) except? 0
cpdef destroy_matrix_product_operator(intptr_t matrix_product_operator)
cpdef intptr_t create_operator_term(intptr_t handle, int32_t num_space_modes, space_mode_extents) except? 0
cpdef destroy_operator_term(intptr_t operator_term)
cpdef operator_term_append_elementary_product(intptr_t handle, intptr_t operator_term, int32_t num_elem_operators, elem_operators, state_modes_acted_on, mode_action_duality, complex coefficient, coefficient_callback, coefficient_gradient_callback)
cpdef operator_term_append_elementary_product_batch(intptr_t handle, intptr_t operator_term, int32_t num_elem_operators, elem_operators, state_modes_acted_on, mode_action_duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, coefficient_callback, coefficient_gradient_callback)
cpdef operator_term_append_matrix_product(intptr_t handle, intptr_t operator_term, int32_t num_matrix_operators, matrix_operators, matrix_conjugation, action_duality, complex coefficient, coefficient_callback, coefficient_gradient_callback)
cpdef operator_term_append_matrix_product_batch(intptr_t handle, intptr_t operator_term, int32_t num_matrix_operators, matrix_operators, matrix_conjugation, action_duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, coefficient_callback, coefficient_gradient_callback)
cpdef operator_term_append_mpo_product(intptr_t handle, intptr_t operator_term, int32_t num_mpo_operators, mpo_operators, mpo_conjugation, state_modes_acted_on, mode_action_duality, complex coefficient, coefficient_callback, coefficient_gradient_callback)
cpdef intptr_t create_operator(intptr_t handle, int32_t num_space_modes, space_mode_extents) except? 0
cpdef destroy_operator(intptr_t superoperator)
cpdef operator_append_term(intptr_t handle, intptr_t superoperator, intptr_t operator_term, int32_t duality, complex coefficient, coefficient_callback, coefficient_gradient_callback)
cpdef operator_append_term_batch(intptr_t handle, intptr_t superoperator, intptr_t operator_term, int32_t duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, coefficient_callback, coefficient_gradient_callback)
cpdef attach_batched_coefficients(intptr_t handle, intptr_t superoperator, int32_t num_operator_term_batched_coeffs, operator_term_batched_coeffs_tmp, operator_term_batched_coeffs, int32_t num_operator_product_batched_coeffs, operator_product_batched_coeffs_tmp, operator_product_batched_coeffs)
cpdef intptr_t create_state_fitting_scope_split_als_config(intptr_t handle) except? 0
cpdef destroy_state_fitting_scope_split_als_config(intptr_t config)
cpdef get_state_fitting_scope_split_als_config_attribute_dtype(int attr)
cpdef state_fitting_scope_split_als_config_set_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef state_fitting_scope_split_als_config_get_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef intptr_t create_state_fitting_approach_lin_solve_config(intptr_t handle) except? 0
cpdef destroy_state_fitting_approach_lin_solve_config(intptr_t config)
cpdef get_state_fitting_approach_lin_solve_config_attribute_dtype(int attr)
cpdef state_fitting_approach_lin_solve_config_set_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef state_fitting_approach_lin_solve_config_get_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef operator_prepare_action(intptr_t handle, intptr_t superoperator, intptr_t state_in, intptr_t state_out, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream)
cpdef operator_compute_action(intptr_t handle, intptr_t superoperator, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state_in, intptr_t state_out, intptr_t workspace, intptr_t stream)
cpdef operator_prepare_action_backward_diff(intptr_t handle, intptr_t superoperator, intptr_t state_in, intptr_t state_out_adj, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream)
cpdef operator_compute_action_backward_diff(intptr_t handle, intptr_t superoperator, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state_in, intptr_t state_out_adj, intptr_t state_in_adj, intptr_t params_grad, intptr_t workspace, intptr_t stream)
cpdef intptr_t create_operator_action(intptr_t handle, int32_t num_operators, operators, int scope_kind, int approach_kind) except? 0
cpdef destroy_operator_action(intptr_t operator_action)
cpdef get_state_fitting_attribute_dtype(int attr)
cpdef operator_action_configure(intptr_t handle, intptr_t operator_action, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef operator_action_prepare(intptr_t handle, intptr_t operator_action, state_in, intptr_t state_out, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream)
cpdef operator_action_compute(intptr_t handle, intptr_t operator_action, double time, int64_t batch_size, int32_t num_params, intptr_t params, state_in, intptr_t state_out, intptr_t workspace, intptr_t stream)
cpdef intptr_t create_expectation(intptr_t handle, intptr_t superoperator) except? 0
cpdef destroy_expectation(intptr_t expectation)
cpdef expectation_prepare(intptr_t handle, intptr_t expectation, intptr_t state, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream)
cpdef expectation_compute(intptr_t handle, intptr_t expectation, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state, intptr_t expectation_value, intptr_t workspace, intptr_t stream)
cpdef intptr_t create_operator_spectrum(intptr_t handle, intptr_t superoperator, int32_t is_hermitian, int spectrum_kind) except? 0
cpdef destroy_operator_spectrum(intptr_t spectrum)
cpdef get_operator_spectrum_config_dtype(int attr)
cpdef operator_spectrum_configure(intptr_t handle, intptr_t spectrum, int attribute, intptr_t attribute_value, size_t attribute_value_size)
cpdef operator_spectrum_prepare(intptr_t handle, intptr_t spectrum, int32_t max_eigen_states, intptr_t state, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream)
cpdef operator_spectrum_compute(intptr_t handle, intptr_t spectrum, double time, int64_t batch_size, int32_t num_params, intptr_t params, int32_t num_eigen_states, eigenstates, intptr_t eigenvalues, intptr_t tolerances, intptr_t workspace, intptr_t stream)
cpdef intptr_t create_time_propagation_scope_split_tdvp_config(intptr_t handle) except? 0
cpdef destroy_time_propagation_scope_split_tdvp_config(intptr_t config)
cpdef get_time_propagation_scope_split_tdvp_config_attribute_dtype(int attr)
cpdef time_propagation_scope_split_tdvp_config_set_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef time_propagation_scope_split_tdvp_config_get_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef intptr_t create_time_propagation_approach_krylov_config(intptr_t handle) except? 0
cpdef destroy_time_propagation_approach_krylov_config(intptr_t config)
cpdef get_time_propagation_approach_krylov_config_attribute_dtype(int attr)
cpdef time_propagation_approach_krylov_config_set_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef time_propagation_approach_krylov_config_get_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef intptr_t create_time_propagation(intptr_t handle, intptr_t superoperator, int32_t is_hermitian, int scope_kind, int approach_kind) except? 0
cpdef destroy_time_propagation(intptr_t time_propagation)
cpdef get_time_propagation_attribute_dtype(int attr)
cpdef time_propagation_configure(intptr_t handle, intptr_t time_propagation, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef time_propagation_prepare(intptr_t handle, intptr_t time_propagation, intptr_t state_in, intptr_t state_out, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream)
cpdef time_propagation_compute(intptr_t handle, intptr_t time_propagation, double time_step_real, double time_step_imag, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state_in, intptr_t state_out, intptr_t workspace, intptr_t stream)
cpdef intptr_t create_svd_config(intptr_t handle) except? 0
cpdef destroy_svd_config(intptr_t config)
cpdef get_svd_config_attribute_dtype(int attr)
cpdef svd_config_set_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef svd_config_get_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef intptr_t create_eigen_decomposition_scope_split_dmrg_config(intptr_t handle) except? 0
cpdef destroy_eigen_decomposition_scope_split_dmrg_config(intptr_t config)
cpdef get_eigen_decomposition_scope_split_dmrg_config_attribute_dtype(int attr)
cpdef eigen_decomposition_scope_split_dmrg_config_set_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef eigen_decomposition_scope_split_dmrg_config_get_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef intptr_t create_eigen_decomposition_approach_krylov_config(intptr_t handle) except? 0
cpdef destroy_eigen_decomposition_approach_krylov_config(intptr_t config)
cpdef get_eigen_decomposition_approach_krylov_config_attribute_dtype(int attr)
cpdef eigen_decomposition_approach_krylov_config_set_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef eigen_decomposition_approach_krylov_config_get_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef intptr_t create_eigen_decomposition_approach_linear_config(intptr_t handle) except? 0
cpdef destroy_eigen_decomposition_approach_linear_config(intptr_t config)
cpdef get_eigen_decomposition_approach_linear_config_attribute_dtype(int attr)
cpdef eigen_decomposition_approach_linear_config_set_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef eigen_decomposition_approach_linear_config_get_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef intptr_t create_eigen_decomposition(intptr_t handle, intptr_t superoperator, int32_t is_hermitian, int spectrum_kind, int scope_kind, int approach_kind) except? 0
cpdef destroy_eigen_decomposition(intptr_t eigen_decomposition)
cpdef get_eigen_decomposition_attribute_dtype(int attr)
cpdef eigen_decomposition_configure(intptr_t handle, intptr_t eigen_decomposition, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef eigen_decomposition_prepare(intptr_t handle, intptr_t eigen_decomposition, int32_t max_eigen_states, intptr_t state, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream)
cpdef eigen_decomposition_compute(intptr_t handle, intptr_t eigen_decomposition, double time, int64_t batch_size, int32_t num_params, intptr_t params, int32_t num_eigen_states, eigenstates, intptr_t eigenvalues, intptr_t tolerances, intptr_t workspace, intptr_t stream)
cpdef intptr_t create_workspace(intptr_t handle) except? 0
cpdef destroy_workspace(intptr_t workspace_descr)
cpdef size_t workspace_get_memory_size(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind) except? -1
cpdef workspace_set_memory(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind, intptr_t memory_buffer, size_t memory_buffer_size)
cpdef tuple workspace_get_memory(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind)
cpdef elementary_operator_attach_buffer(intptr_t handle, intptr_t elem_operator, intptr_t buffer, size_t buffer_size)
cpdef matrix_operator_dense_local_attach_buffer(intptr_t handle, intptr_t matrix_operator, intptr_t buffer, size_t buffer_size)

###############################################################################

cpdef tuple state_get_component_storage_size(intptr_t handle, intptr_t state, int32_t num_state_components)
