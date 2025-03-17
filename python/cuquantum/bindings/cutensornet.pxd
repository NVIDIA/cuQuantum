# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 25.03.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycutensornet cimport *


###############################################################################
# Types
###############################################################################

ctypedef cutensornetNetworkDescriptor_t NetworkDescriptor
ctypedef cutensornetContractionPlan_t ContractionPlan
ctypedef cutensornetHandle_t Handle
ctypedef cutensornetWorkspaceDescriptor_t WorkspaceDescriptor
ctypedef cutensornetContractionOptimizerConfig_t ContractionOptimizerConfig
ctypedef cutensornetContractionOptimizerInfo_t ContractionOptimizerInfo
ctypedef cutensornetContractionAutotunePreference_t ContractionAutotunePreference
ctypedef cutensornetSliceGroup_t SliceGroup
ctypedef cutensornetTensorDescriptor_t TensorDescriptor
ctypedef cutensornetTensorSVDConfig_t TensorSVDConfig
ctypedef cutensornetTensorSVDInfo_t TensorSVDInfo
ctypedef cutensornetState_t State
ctypedef cutensornetStateMarginal_t StateMarginal
ctypedef cutensornetStateSampler_t StateSampler
ctypedef cutensornetStateAccessor_t StateAccessor
ctypedef cutensornetStateExpectation_t StateExpectation
ctypedef cutensornetNetworkOperator_t NetworkOperator
ctypedef cutensornetNodePair_t NodePair
ctypedef cutensornetSliceInfoPair_t SliceInfoPair
ctypedef cutensornetTensorQualifiers_t TensorQualifiers
ctypedef cutensornetDeviceMemHandler_t _DeviceMemHandler
ctypedef cutensornetDistributedCommunicator_t DistributedCommunicator
ctypedef cutensornetDistributedInterface_t DistributedInterface
ctypedef cutensornetTensorIDList_t TensorIDList
ctypedef cutensornetGesvdjParams_t GesvdjParams
ctypedef cutensornetGesvdrParams_t GesvdrParams
ctypedef cutensornetGesvdjStatus_t GesvdjStatus
ctypedef cutensornetGesvdpStatus_t GesvdpStatus
ctypedef cutensornetLoggerCallback_t LoggerCallback
ctypedef cutensornetLoggerCallbackData_t LoggerCallbackData
ctypedef cutensornetContractionPath_t ContractionPath
ctypedef cutensornetSlicingConfig_t SlicingConfig

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef cutensornetStatus_t _Status
ctypedef cutensornetComputeType_t _ComputeType
ctypedef cutensornetGraphAlgo_t _GraphAlgo
ctypedef cutensornetMemoryModel_t _MemoryModel
ctypedef cutensornetOptimizerCost_t _OptimizerCost
ctypedef cutensornetContractionOptimizerConfigAttributes_t _ContractionOptimizerConfigAttribute
ctypedef cutensornetContractionOptimizerInfoAttributes_t _ContractionOptimizerInfoAttribute
ctypedef cutensornetContractionAutotunePreferenceAttributes_t _ContractionAutotunePreferenceAttribute
ctypedef cutensornetWorksizePref_t _WorksizePref
ctypedef cutensornetMemspace_t _Memspace
ctypedef cutensornetWorkspaceKind_t _WorkspaceKind
ctypedef cutensornetTensorSVDConfigAttributes_t _TensorSVDConfigAttribute
ctypedef cutensornetTensorSVDPartition_t _TensorSVDPartition
ctypedef cutensornetTensorSVDNormalization_t _TensorSVDNormalization
ctypedef cutensornetTensorSVDInfoAttributes_t _TensorSVDInfoAttribute
ctypedef cutensornetGateSplitAlgo_t _GateSplitAlgo
ctypedef cutensornetNetworkAttributes_t _NetworkAttribute
ctypedef cutensornetSmartOption_t _SmartOption
ctypedef cutensornetTensorSVDAlgo_t _TensorSVDAlgo
ctypedef cutensornetStatePurity_t _StatePurity
ctypedef cutensornetMarginalAttributes_t _MarginalAttribute
ctypedef cutensornetSamplerAttributes_t _SamplerAttribute
ctypedef cutensornetAccessorAttributes_t _AccessorAttribute
ctypedef cutensornetExpectationAttributes_t _ExpectationAttribute
ctypedef cutensornetBoundaryCondition_t _BoundaryCondition
ctypedef cutensornetStateAttributes_t _StateAttribute
ctypedef cutensornetStateMPOApplication_t _StateMPOApplication
ctypedef cutensornetStateMPSGaugeOption_t _StateMPSGaugeOption


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef intptr_t create_network_descriptor(intptr_t handle, int32_t num_inputs, num_modes_in, extents_in, strides_in, modes_in, qualifiers_in, int32_t num_modes_out, extents_out, strides_out, modes_out, int data_type, int compute_type) except? 0
cpdef destroy_network_descriptor(intptr_t desc)
cpdef intptr_t get_output_tensor_descriptor(intptr_t handle, intptr_t desc_net) except? 0
cpdef intptr_t create_workspace_descriptor(intptr_t handle) except? 0
cpdef workspace_compute_contraction_sizes(intptr_t handle, intptr_t desc_net, intptr_t optimizer_info, intptr_t work_desc)
cpdef int64_t workspace_get_memory_size(intptr_t handle, intptr_t work_desc, int work_pref, int mem_space, int work_kind) except? -1
cpdef workspace_set_memory(intptr_t handle, intptr_t work_desc, int mem_space, int work_kind, intptr_t memory_ptr, int64_t memory_size)
cpdef tuple workspace_get_memory(intptr_t handle, intptr_t work_desc, int mem_space, int work_kind)
cpdef destroy_workspace_descriptor(intptr_t desc)
cpdef intptr_t create_contraction_optimizer_config(intptr_t handle) except? 0
cpdef destroy_contraction_optimizer_config(intptr_t optimizer_config)
cpdef get_contraction_optimizer_config_attribute_dtype(int attr)
cpdef contraction_optimizer_config_get_attribute(intptr_t handle, intptr_t optimizer_config, int attr, intptr_t buf, size_t size_in_bytes)
cpdef contraction_optimizer_config_set_attribute(intptr_t handle, intptr_t optimizer_config, int attr, intptr_t buf, size_t size_in_bytes)
cpdef destroy_contraction_optimizer_info(intptr_t optimizer_info)
cpdef intptr_t create_contraction_optimizer_info(intptr_t handle, intptr_t desc_net) except? 0
cpdef contraction_optimize(intptr_t handle, intptr_t desc_net, intptr_t optimizer_config, uint64_t workspace_size_constraint, intptr_t optimizer_info)
cpdef get_contraction_optimizer_info_attribute_dtype(int attr)
cpdef contraction_optimizer_info_get_attribute(intptr_t handle, intptr_t optimizer_info, int attr, intptr_t buf, size_t size_in_bytes)
cpdef contraction_optimizer_info_set_attribute(intptr_t handle, intptr_t optimizer_info, int attr, intptr_t buf, size_t size_in_bytes)
cpdef size_t contraction_optimizer_info_get_packed_size(intptr_t handle, intptr_t optimizer_info) except? 0
cpdef contraction_optimizer_info_pack_data(intptr_t handle, intptr_t optimizer_info, buffer, size_t size_in_bytes)
cpdef intptr_t create_contraction_optimizer_info_from_packed_data(intptr_t handle, intptr_t desc_net, buffer, size_t size_in_bytes) except? 0
cpdef update_contraction_optimizer_info_from_packed_data(intptr_t handle, buffer, size_t size_in_bytes, intptr_t optimizer_info)
cpdef intptr_t create_contraction_plan(intptr_t handle, intptr_t desc_net, intptr_t optimizer_info, intptr_t work_desc) except? 0
cpdef destroy_contraction_plan(intptr_t plan)
cpdef contraction_autotune(intptr_t handle, intptr_t plan, raw_data_in, intptr_t raw_data_out, intptr_t work_desc, intptr_t pref, intptr_t stream)
cpdef intptr_t create_contraction_autotune_preference(intptr_t handle) except? 0
cpdef get_contraction_autotune_preference_attribute_dtype(int attr)
cpdef contraction_autotune_preference_get_attribute(intptr_t handle, intptr_t autotune_preference, int attr, intptr_t buf, size_t size_in_bytes)
cpdef contraction_autotune_preference_set_attribute(intptr_t handle, intptr_t autotune_preference, int attr, intptr_t buf, size_t size_in_bytes)
cpdef destroy_contraction_autotune_preference(intptr_t autotune_preference)
cpdef intptr_t create_slice_group_from_id_range(intptr_t handle, int64_t slice_id_start, int64_t slice_id_stop, int64_t slice_id_step) except? 0
cpdef destroy_slice_group(intptr_t slice_group)
cpdef contract_slices(intptr_t handle, intptr_t plan, raw_data_in, intptr_t raw_data_out, int32_t accumulate_output, intptr_t work_desc, intptr_t slice_group, intptr_t stream)
cpdef intptr_t create_tensor_descriptor(intptr_t handle, int32_t num_modes, extents, strides, modes, int data_type) except? 0
cpdef destroy_tensor_descriptor(intptr_t desc_tensor)
cpdef intptr_t create_tensor_svd_config(intptr_t handle) except? 0
cpdef destroy_tensor_svd_config(intptr_t svd_config)
cpdef get_tensor_svd_config_attribute_dtype(int attr)
cpdef tensor_svd_config_get_attribute(intptr_t handle, intptr_t svd_config, int attr, intptr_t buf, size_t size_in_bytes)
cpdef tensor_svd_config_set_attribute(intptr_t handle, intptr_t svd_config, int attr, intptr_t buf, size_t size_in_bytes)
cpdef workspace_compute_svd_sizes(intptr_t handle, intptr_t desc_tensor_in, intptr_t desc_tensor_u, intptr_t desc_tensor_v, intptr_t svd_config, intptr_t work_desc)
cpdef workspace_compute_qr_sizes(intptr_t handle, intptr_t desc_tensor_in, intptr_t desc_tensor_q, intptr_t desc_tensor_r, intptr_t work_desc)
cpdef intptr_t create_tensor_svd_info(intptr_t handle) except? 0
cpdef get_tensor_svd_info_attribute_dtype(int attr)
cpdef tensor_svd_info_get_attribute(intptr_t handle, intptr_t svd_info, int attr, intptr_t buf, size_t size_in_bytes)
cpdef destroy_tensor_svd_info(intptr_t svd_info)
cpdef tensor_svd(intptr_t handle, intptr_t desc_tensor_in, intptr_t raw_data_in, intptr_t desc_tensor_u, intptr_t u, intptr_t s, intptr_t desc_tensor_v, intptr_t v, intptr_t svd_config, intptr_t svd_info, intptr_t work_desc, intptr_t stream)
cpdef tensor_qr(intptr_t handle, intptr_t desc_tensor_in, intptr_t raw_data_in, intptr_t desc_tensor_q, intptr_t q, intptr_t desc_tensor_r, intptr_t r, intptr_t work_desc, intptr_t stream)
cpdef workspace_compute_gate_split_sizes(intptr_t handle, intptr_t desc_tensor_in_a, intptr_t desc_tensor_in_b, intptr_t desc_tensor_in_g, intptr_t desc_tensor_u, intptr_t desc_tensor_v, int gate_algo, intptr_t svd_config, int compute_type, intptr_t work_desc)
cpdef gate_split(intptr_t handle, intptr_t desc_tensor_in_a, intptr_t raw_data_in_a, intptr_t desc_tensor_in_b, intptr_t raw_data_in_b, intptr_t desc_tensor_in_g, intptr_t raw_data_in_g, intptr_t desc_tensor_u, intptr_t u, intptr_t s, intptr_t desc_tensor_v, intptr_t v, int gate_algo, intptr_t svd_config, int compute_type, intptr_t svd_info, intptr_t work_desc, intptr_t stream)
cpdef logger_set_file(intptr_t file)
cpdef logger_open_file(log_file)
cpdef logger_set_level(int32_t level)
cpdef logger_set_mask(int32_t mask)
cpdef logger_force_disable()
cpdef size_t get_version() except? 0
cpdef size_t get_cudart_version() except? 0
cpdef str get_error_string(int error)
cpdef distributed_reset_configuration(intptr_t handle, intptr_t comm_ptr, size_t comm_size)
cpdef int32_t distributed_get_num_ranks(intptr_t handle) except? -1
cpdef int32_t distributed_get_proc_rank(intptr_t handle) except? -1
cpdef distributed_synchronize(intptr_t handle)
cpdef get_network_attribute_dtype(int attr)
cpdef network_get_attribute(intptr_t handle, intptr_t network_desc, int attr, intptr_t buf, size_t size_in_bytes)
cpdef network_set_attribute(intptr_t handle, intptr_t network_desc, int attr, intptr_t buf, size_t size_in_bytes)
cpdef workspace_purge_cache(intptr_t handle, intptr_t work_desc, int mem_space)
cpdef compute_gradients_backward(intptr_t handle, intptr_t plan, raw_data_in, intptr_t output_gradient, gradients, int32_t accumulate_output, intptr_t work_desc, intptr_t stream)
cpdef intptr_t create_state(intptr_t handle, int purity, int32_t num_state_modes, state_mode_extents, int data_type) except? 0
cpdef int64_t state_apply_tensor(intptr_t handle, intptr_t tensor_network_state, int32_t num_state_modes, state_modes, intptr_t tensor_data, tensor_mode_strides, int32_t immutable, int32_t adjoint, int32_t unitary) except? -1
cpdef state_update_tensor(intptr_t handle, intptr_t tensor_network_state, int64_t tensor_id, intptr_t tensor_data, int32_t unitary)
cpdef destroy_state(intptr_t tensor_network_state)
cpdef intptr_t create_marginal(intptr_t handle, intptr_t tensor_network_state, int32_t num_marginal_modes, marginal_modes, int32_t num_projected_modes, projected_modes, marginal_tensor_strides) except? 0
cpdef get_marginal_attribute_dtype(int attr)
cpdef marginal_configure(intptr_t handle, intptr_t tensor_network_marginal, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef marginal_prepare(intptr_t handle, intptr_t tensor_network_marginal, size_t max_workspace_size_device, intptr_t work_desc, intptr_t cuda_stream)
cpdef marginal_compute(intptr_t handle, intptr_t tensor_network_marginal, projected_mode_values, intptr_t work_desc, intptr_t marginal_tensor, intptr_t cuda_stream)
cpdef destroy_marginal(intptr_t tensor_network_marginal)
cpdef intptr_t create_sampler(intptr_t handle, intptr_t tensor_network_state, int32_t num_modes_to_sample, modes_to_sample) except? 0
cpdef get_sampler_attribute_dtype(int attr)
cpdef sampler_configure(intptr_t handle, intptr_t tensor_network_sampler, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef sampler_prepare(intptr_t handle, intptr_t tensor_network_sampler, size_t max_workspace_size_device, intptr_t work_desc, intptr_t cuda_stream)
cpdef sampler_sample(intptr_t handle, intptr_t tensor_network_sampler, int64_t num_shots, intptr_t work_desc, intptr_t samples, intptr_t cuda_stream)
cpdef destroy_sampler(intptr_t tensor_network_sampler)
cpdef state_finalize_mps(intptr_t handle, intptr_t tensor_network_state, int boundary_condition, extents_out, strides_out)
cpdef get_state_attribute_dtype(int attr)
cpdef state_configure(intptr_t handle, intptr_t tensor_network_state, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef state_prepare(intptr_t handle, intptr_t tensor_network_state, size_t max_workspace_size_device, intptr_t work_desc, intptr_t cuda_stream)
cpdef intptr_t create_network_operator(intptr_t handle, int32_t num_state_modes, state_mode_extents, int data_type) except? 0
cpdef int64_t network_operator_append_product(intptr_t handle, intptr_t tensor_network_operator, complex coefficient, int32_t num_tensors, num_state_modes, state_modes, tensor_mode_strides, tensor_data) except? -1
cpdef destroy_network_operator(intptr_t tensor_network_operator)
cpdef intptr_t create_accessor(intptr_t handle, intptr_t tensor_network_state, int32_t num_projected_modes, projected_modes, amplitudes_tensor_strides) except? 0
cpdef get_accessor_attribute_dtype(int attr)
cpdef accessor_configure(intptr_t handle, intptr_t tensor_network_accessor, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef accessor_prepare(intptr_t handle, intptr_t tensor_network_accessor, size_t max_workspace_size_device, intptr_t work_desc, intptr_t cuda_stream)
cpdef accessor_compute(intptr_t handle, intptr_t tensor_network_accessor, projected_mode_values, intptr_t work_desc, intptr_t amplitudes_tensor, intptr_t state_norm, intptr_t cuda_stream)
cpdef destroy_accessor(intptr_t tensor_network_accessor)
cpdef intptr_t create_expectation(intptr_t handle, intptr_t tensor_network_state, intptr_t tensor_network_operator) except? 0
cpdef get_expectation_attribute_dtype(int attr)
cpdef expectation_configure(intptr_t handle, intptr_t tensor_network_expectation, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef expectation_prepare(intptr_t handle, intptr_t tensor_network_expectation, size_t max_workspace_size_device, intptr_t work_desc, intptr_t cuda_stream)
cpdef expectation_compute(intptr_t handle, intptr_t tensor_network_expectation, intptr_t work_desc, intptr_t expectation_value, intptr_t state_norm, intptr_t cuda_stream)
cpdef destroy_expectation(intptr_t tensor_network_expectation)
cpdef int64_t state_apply_tensor_operator(intptr_t handle, intptr_t tensor_network_state, int32_t num_state_modes, state_modes, intptr_t tensor_data, tensor_mode_strides, int32_t immutable, int32_t adjoint, int32_t unitary) except? -1
cpdef int64_t state_apply_controlled_tensor_operator(intptr_t handle, intptr_t tensor_network_state, int32_t num_control_modes, state_control_modes, state_control_values, int32_t num_target_modes, state_target_modes, intptr_t tensor_data, tensor_mode_strides, int32_t immutable, int32_t adjoint, int32_t unitary) except? -1
cpdef state_update_tensor_operator(intptr_t handle, intptr_t tensor_network_state, int64_t tensor_id, intptr_t tensor_data, int32_t unitary)
cpdef int64_t state_apply_network_operator(intptr_t handle, intptr_t tensor_network_state, intptr_t tensor_network_operator, int32_t immutable, int32_t adjoint, int32_t unitary) except? -1
cpdef state_initialize_mps(intptr_t handle, intptr_t tensor_network_state, int boundary_condition, extents_in, strides_in, state_tensors_in)
cpdef state_get_info(intptr_t handle, intptr_t tensor_network_state, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef int64_t network_operator_append_mpo(intptr_t handle, intptr_t tensor_network_operator, complex coefficient, int32_t num_state_modes, state_modes, tensor_mode_extents, tensor_mode_strides, tensor_data, int boundary_condition) except? -1
cpdef accessor_get_info(intptr_t handle, intptr_t tensor_network_accessor, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef expectation_get_info(intptr_t handle, intptr_t tensor_network_expectation, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef marginal_get_info(intptr_t handle, intptr_t tensor_network_marginal, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef sampler_get_info(intptr_t handle, intptr_t tensor_network_sampler, int attribute, intptr_t attribute_value, size_t attribute_size)
cpdef int64_t state_apply_unitary_channel(intptr_t handle, intptr_t tensor_network_state, int32_t num_state_modes, state_modes, int32_t num_tensors, tensor_data, tensor_mode_strides, probabilities) except? -1
cpdef state_capture_mps(intptr_t handle, intptr_t tensor_network_state)
cpdef int64_t state_apply_general_channel(intptr_t handle, intptr_t tensor_network_state, int32_t num_state_modes, state_modes, int32_t num_tensors, tensor_data, tensor_mode_strides) except? -1
