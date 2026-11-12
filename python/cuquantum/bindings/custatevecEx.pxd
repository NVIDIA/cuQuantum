# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 26.09.0 to 26.11.0. Do not modify it directly.


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    int32_t,
    int64_t,
    intptr_t,
    uint32_t,
)


# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport intptr_t

from .cycustatevecEx cimport *


###############################################################################
# Types
###############################################################################

ctypedef custatevecExDictionaryDescriptor_t DictionaryDescriptor
ctypedef custatevecExCommunicatorDescriptor_t CommunicatorDescriptor
ctypedef custatevecExStateVectorDescriptor_t StateVectorDescriptor
ctypedef custatevecExSVUpdaterDescriptor_t SVUpdaterDescriptor
ctypedef custatevecExResourceManagerDescriptor_t ResourceManagerDescriptor

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType

ctypedef custatevecExSVUpdaterConfigItem_t _SVUpdaterConfigItem


###############################################################################
# Enum
###############################################################################

ctypedef custatevecExCommunicatorStatus_t _CommunicatorStatus
ctypedef custatevecExStateVectorCapability_t _StateVectorCapability
ctypedef custatevecExStateVectorDistributionType_t _StateVectorDistributionType
ctypedef custatevecExIndexBitDomain_t _IndexBitDomain
ctypedef custatevecExWireInitMode_t _WireInitMode
ctypedef custatevecExGlobalIndexBitClass_t _GlobalIndexBitClass
ctypedef custatevecExStateVectorProperty_t _StateVectorProperty
ctypedef custatevecExPermutationType_t _PermutationType
ctypedef custatevecExExposeResources_t _ExposeResources
ctypedef custatevecExMatrixType_t _MatrixType
ctypedef custatevecExSVUpdaterConfigName_t _SVUpdaterConfigName
ctypedef custatevecExMemorySharingMethod_t _MemorySharingMethod
ctypedef custatevecExMemoryPlacement_t _MemoryPlacement
ctypedef custatevecExSynchronizationScope_t _SynchronizationScope


###############################################################################
# Functions
###############################################################################

cpdef dictionary_destroy(intptr_t dictionary)
cpdef int communicator_finalize() except? -1
cpdef tuple communicator_get_size_and_rank()
cpdef intptr_t communicator_create() except? 0
cpdef communicator_destroy(intptr_t ex_communicator)
cpdef intptr_t configure_state_vector_single_device(int sv_data_type, int32_t num_wires, int32_t num_device_wires, int32_t device_id, uint32_t capability) except? 0
cpdef intptr_t configure_state_vector_multi_device(int sv_data_type, int32_t num_wires, int32_t num_device_wires, device_ids, int32_t num_devices, int network_type, uint32_t capability) except? 0
cpdef intptr_t configure_state_vector_multi_process(int sv_data_type, int32_t num_wires, int32_t num_device_wires, int32_t device_id, int memory_sharing_method, global_index_bit_classes, num_global_index_bits_per_layer, int32_t num_global_index_bit_layers, size_t transfer_workspace_size_in_bytes, intptr_t aux_config, uint32_t capability) except? 0
cpdef intptr_t state_vector_create_single_process(intptr_t sv_config, streams, int32_t num_streams, intptr_t resource_manager) except? 0
cpdef intptr_t state_vector_create_multi_process(intptr_t sv_config, intptr_t stream, intptr_t ex_communicator, intptr_t resource_manager) except? 0
cpdef state_vector_destroy(intptr_t state_vector)
cpdef state_vector_get_property(intptr_t state_vector, int property, intptr_t value, size_t size_in_bytes)
cpdef state_vector_set_math_mode(intptr_t state_vector, int mode)
cpdef state_vector_set_zero_state(intptr_t state_vector)
cpdef state_vector_get_state(intptr_t state_vector, intptr_t state, int data_type, int64_t begin, int64_t end, int32_t max_num_concurrent_copies)
cpdef state_vector_set_state(intptr_t state_vector, intptr_t state, int data_type, int64_t begin, int64_t end, int32_t max_num_concurrent_copies)
cpdef state_vector_reassign_wire_ordering(intptr_t state_vector, wire_ordering, int32_t wire_ordering_len)
cpdef state_vector_permute_index_bits(intptr_t state_vector, permutation, int32_t permutation_len, int permutation_type)
cpdef state_vector_stage_sub_sv(intptr_t state_vector, int32_t sub_sv_index)
cpdef state_vector_expose_resources(intptr_t state_vector, int expose_resources)
cpdef state_vector_synchronize(intptr_t state_vector)
cpdef state_vector_synchronize_scoped(intptr_t state_vector, int scope)
cpdef state_vector_add_wires(intptr_t state_vector, int index_bit_domain, int32_t num_wires_to_add, int wire_init_mode, intptr_t wires_added)
cpdef abs2sum_array(intptr_t state_vector, intptr_t abs2sum, output_ordering, int32_t output_ordering_len, mask_bit_string, mask_wire_ordering, int32_t mask_len)
cpdef int64_t measure(intptr_t state_vector, bit_string_ordering, int32_t bit_string_ordering_len, double randnum, int collapse, intptr_t reserved) except? -1
cpdef sample(intptr_t state_vector, intptr_t bit_strings, bit_string_ordering, int32_t bit_string_ordering_len, randnums, int32_t num_shots, int output, abs2sums)
cpdef apply_matrix(intptr_t state_vector, intptr_t matrix, int matrix_data_type, int ex_matrix_type, int layout, int32_t adjoint, targets, int32_t num_targets, controls, control_bit_values, int32_t num_controls)
cpdef apply_pauli_rotation(intptr_t state_vector, double theta, paulis, targets, int32_t num_targets, controls, control_bit_values, int32_t num_controls)
cpdef compute_expectation_on_pauli_basis(intptr_t state_vector, intptr_t expectation_values, pauli_operator_arrays, int32_t num_pauli_operator_arrays, basis_wires_array, num_basis_wires_array)
cpdef compute_expectation(intptr_t state_vector, intptr_t expectation_values, intptr_t matrices, int matrix_data_type, int layout, int32_t num_matrices, basis_wires, int32_t num_basis_wires)
cpdef intptr_t sv_updater_create(intptr_t sv_updater_config, intptr_t resource_manager) except? 0
cpdef sv_updater_destroy(intptr_t sv_updater)
cpdef sv_updater_clear(intptr_t sv_updater)
cpdef sv_updater_enqueue_matrix(intptr_t sv_updater, intptr_t matrix, int matrix_data_type, int ex_matrix_type, int layout, int32_t adjoint, targets, int32_t num_targets, controls, control_bit_values, int32_t num_controls)
cpdef sv_updater_enqueue_unitary_channel(intptr_t sv_updater, unitaries, int unitaries_data_type, ex_matrix_types, int32_t num_unitaries, int layout, probabilities, channel_wires, int32_t num_channel_wires)
cpdef sv_updater_enqueue_general_channel(intptr_t sv_updater, matrices, int matrix_data_type, ex_matrix_types, int32_t num_matrices, int layout, channel_wires, int32_t num_channel_wires)
cpdef int32_t sv_updater_get_max_num_required_randnums(intptr_t sv_updater) except? -1
cpdef sv_updater_apply(intptr_t sv_updater, intptr_t state_vector, randnums, int32_t num_randnums)
