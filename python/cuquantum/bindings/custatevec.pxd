# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 25.03.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycustatevec cimport *


###############################################################################
# Types
###############################################################################

ctypedef custatevecHandle_t Handle
ctypedef custatevecSamplerDescriptor_t SamplerDescriptor
ctypedef custatevecAccessorDescriptor_t AccessorDescriptor
ctypedef custatevecCommunicatorDescriptor_t CommunicatorDescriptor
ctypedef custatevecDistIndexBitSwapSchedulerDescriptor_t DistIndexBitSwapSchedulerDescriptor
ctypedef custatevecSVSwapWorkerDescriptor_t SVSwapWorkerDescriptor
ctypedef custatevecSubSVMigratorDescriptor_t SubSVMigratorDescriptor
ctypedef custatevecDeviceMemHandler_t _DeviceMemHandler
ctypedef custatevecSVSwapParameters_t _SVSwapParameters
ctypedef custatevecLoggerCallback_t LoggerCallback
ctypedef custatevecLoggerCallbackData_t LoggerCallbackData

ctypedef cudaStream_t Stream
ctypedef cudaEvent_t Event
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef custatevecStatus_t _Status
ctypedef custatevecPauli_t _Pauli
ctypedef custatevecMatrixLayout_t _MatrixLayout
ctypedef custatevecMatrixType_t _MatrixType
ctypedef custatevecCollapseOp_t _CollapseOp
ctypedef custatevecComputeType_t _ComputeType
ctypedef custatevecSamplerOutput_t _SamplerOutput
ctypedef custatevecDeviceNetworkType_t _DeviceNetworkType
ctypedef custatevecCommunicatorType_t _CommunicatorType
ctypedef custatevecDataTransferType_t _DataTransferType
ctypedef custatevecMatrixMapType_t _MatrixMapType
ctypedef custatevecStateVectorType_t _StateVectorType
ctypedef custatevecMathMode_t _MathMode


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef size_t get_default_workspace_size(intptr_t handle) except? 0
cpdef set_workspace(intptr_t handle, intptr_t workspace, size_t workspace_size_in_bytes)
cpdef str get_error_name(int status)
cpdef str get_error_string(int status)
cpdef int32_t get_property(int type) except? -1
cpdef size_t get_version() except? 0
cpdef set_stream(intptr_t handle, intptr_t stream_id)
cpdef intptr_t get_stream(intptr_t handle) except? 0
cpdef logger_open_file(log_file)
cpdef logger_set_level(int32_t level)
cpdef logger_set_mask(int32_t mask)
cpdef logger_force_disable()
cpdef abs2sum_array(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t abs2sum, bit_ordering, uint32_t bit_ordering_len, mask_bit_string, mask_ordering, uint32_t mask_len)
cpdef collapse_on_z_basis(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, int32_t parity, basis_bits, uint32_t n_basis_bits, double norm)
cpdef collapse_by_bit_string(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, bit_string, bit_ordering, uint32_t bit_string_len, double norm)
cpdef int32_t measure_on_z_basis(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, basis_bits, uint32_t n_basis_bits, double randnum, int collapse) except? -1
cpdef batch_measure(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t bit_string, bit_ordering, uint32_t bit_string_len, double randnum, int collapse)
cpdef batch_measure_with_offset(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t bit_string, bit_ordering, uint32_t bit_string_len, double randnum, int collapse, double offset, double abs2sum)
cpdef apply_pauli_rotation(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, double theta, paulis, targets, uint32_t n_targets, controls, control_bit_values, uint32_t n_controls)
cpdef size_t apply_matrix_get_workspace_size(intptr_t handle, int sv_data_type, uint32_t n_index_bits, intptr_t matrix, int matrix_data_type, int layout, int32_t adjoint, uint32_t n_targets, uint32_t n_controls, int compute_type) except? 0
cpdef apply_matrix(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t matrix, int matrix_data_type, int layout, int32_t adjoint, targets, uint32_t n_targets, controls, control_bit_values, uint32_t n_controls, int compute_type, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes)
cpdef size_t compute_expectation_get_workspace_size(intptr_t handle, int sv_data_type, uint32_t n_index_bits, intptr_t matrix, int matrix_data_type, int layout, uint32_t n_basis_bits, int compute_type) except? 0
cpdef double compute_expectation(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t expectation_value, int expectation_data_type, intptr_t matrix, int matrix_data_type, int layout, basis_bits, uint32_t n_basis_bits, int compute_type, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes) except? 0
cpdef tuple sampler_create(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_max_shots)
cpdef sampler_destroy(intptr_t sampler)
cpdef sampler_preprocess(intptr_t handle, intptr_t sampler, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes)
cpdef double sampler_get_squared_norm(intptr_t handle, intptr_t sampler) except? -1
cpdef sampler_apply_sub_sv_offset(intptr_t handle, intptr_t sampler, int32_t sub_sv_ord, uint32_t n_sub_svs, double offset, double norm)
cpdef sampler_sample(intptr_t handle, intptr_t sampler, intptr_t bit_strings, bit_ordering, uint32_t bit_string_len, randnums, uint32_t n_shots, int output)
cpdef size_t apply_generalized_permutation_matrix_get_workspace_size(intptr_t handle, int sv_data_type, uint32_t n_index_bits, permutation, intptr_t diagonals, int diagonals_data_type, targets, uint32_t n_targets, uint32_t n_controls) except? 0
cpdef apply_generalized_permutation_matrix(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, permutation, intptr_t diagonals, int diagonals_data_type, int32_t adjoint, targets, uint32_t n_targets, controls, control_bit_values, uint32_t n_controls, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes)
cpdef compute_expectations_on_pauli_basis(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t expectation_values, pauli_operators_array, uint32_t n_pauli_operator_arrays, basis_bits_array, n_basis_bits_array)
cpdef tuple accessor_create(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, bit_ordering, uint32_t bit_ordering_len, mask_bit_string, mask_ordering, uint32_t mask_len)
cpdef tuple accessor_create_view(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, bit_ordering, uint32_t bit_ordering_len, mask_bit_string, mask_ordering, uint32_t mask_len)
cpdef accessor_destroy(intptr_t accessor)
cpdef accessor_set_extra_workspace(intptr_t handle, intptr_t accessor, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes)
cpdef accessor_get(intptr_t handle, intptr_t accessor, intptr_t external_buffer, int64_t begin, int64_t end)
cpdef accessor_set(intptr_t handle, intptr_t accessor, intptr_t external_buffer, int64_t begin, int64_t end)
cpdef size_t test_matrix_type_get_workspace_size(intptr_t handle, int matrix_type, intptr_t matrix, int matrix_data_type, int layout, uint32_t n_targets, int32_t adjoint, int compute_type) except? 0
cpdef double test_matrix_type(intptr_t handle, int matrix_type, intptr_t matrix, int matrix_data_type, int layout, uint32_t n_targets, int32_t adjoint, int compute_type, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes) except? -1
cpdef intptr_t communicator_create(intptr_t handle, int communicator_type, soname) except? 0
cpdef communicator_destroy(intptr_t handle, intptr_t communicator)
cpdef intptr_t dist_index_bit_swap_scheduler_create(intptr_t handle, uint32_t n_global_index_bits, uint32_t n_local_index_bits) except? 0
cpdef dist_index_bit_swap_scheduler_destroy(intptr_t handle, intptr_t scheduler)
cpdef tuple sv_swap_worker_create(intptr_t handle, intptr_t communicator, intptr_t org_sub_sv, int32_t org_sub_sv_ind_ex, intptr_t org_event, int sv_data_type, intptr_t stream)
cpdef sv_swap_worker_destroy(intptr_t handle, intptr_t sv_swap_worker)
cpdef sv_swap_worker_set_extra_workspace(intptr_t handle, intptr_t sv_swap_worker, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes)
cpdef sv_swap_worker_set_transfer_workspace(intptr_t handle, intptr_t sv_swap_worker, intptr_t transfer_workspace, size_t transfer_workspace_size_in_bytes)
cpdef sv_swap_worker_set_sub_svs_p2p(intptr_t handle, intptr_t sv_swap_worker, dst_sub_svs_p2p, dst_sub_sv_indices_p2p, dst_events, uint32_t n_dst_sub_svs_p2p)
cpdef sv_swap_worker_execute(intptr_t handle, intptr_t sv_swap_worker, int64_t begin, int64_t end)
cpdef initialize_state_vector(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, int sv_type)
cpdef size_t apply_matrix_batched_get_workspace_size(intptr_t handle, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, int map_type, matrix_indices, intptr_t matrices, int matrix_data_type, int layout, int32_t adjoint, uint32_t n_matrices, uint32_t n_targets, uint32_t n_controls, int compute_type) except? 0
cpdef apply_matrix_batched(intptr_t handle, intptr_t batched_sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, int map_type, matrix_indices, intptr_t matrices, int matrix_data_type, int layout, int32_t adjoint, uint32_t n_matrices, targets, uint32_t n_targets, controls, control_bit_values, uint32_t n_controls, int compute_type, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes)
cpdef abs2sum_array_batched(intptr_t handle, intptr_t batched_sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, intptr_t abs2sum_arrays, int64_t abs2sum_array_stride, bit_ordering, uint32_t bit_ordering_len, mask_bit_strings, mask_ordering, uint32_t mask_len)
cpdef size_t collapse_by_bit_string_batched_get_workspace_size(intptr_t handle, uint32_t n_svs, bit_strings, norms) except? 0
cpdef collapse_by_bit_string_batched(intptr_t handle, intptr_t batched_sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, bit_strings, bit_ordering, uint32_t bit_string_len, norms, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes)
cpdef measure_batched(intptr_t handle, intptr_t batched_sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, intptr_t bit_strings, bit_ordering, uint32_t bit_string_len, randnums, int collapse)
cpdef intptr_t sub_sv_migrator_create(intptr_t handle, intptr_t device_slots, int sv_data_type, int n_device_slots, int n_local_index_bits) except? 0
cpdef sub_sv_migrator_destroy(intptr_t handle, intptr_t migrator)
cpdef sub_sv_migrator_migrate(intptr_t handle, intptr_t migrator, int device_slot_ind_ex, intptr_t src_sub_sv, intptr_t dst_sub_sv, int64_t begin, int64_t end)
cpdef size_t compute_expectation_batched_get_workspace_size(intptr_t handle, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, intptr_t matrices, int matrix_data_type, int layout, uint32_t n_matrices, uint32_t n_basis_bits, int compute_type) except? 0
cpdef compute_expectation_batched(intptr_t handle, intptr_t batched_sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, intptr_t expectation_values, intptr_t matrices, int matrix_data_type, int layout, uint32_t n_matrices, basis_bits, uint32_t n_basis_bits, int compute_type, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes)
cpdef set_math_mode(intptr_t handle, int mode)
cpdef int get_math_mode(intptr_t handle) except? -1
