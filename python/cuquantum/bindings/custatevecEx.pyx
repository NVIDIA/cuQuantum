# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 26.09.0 to 26.11.0. Do not modify it directly.


# <<<< PREAMBLE CONTENT >>>>

from cuquantum.bindings._utils cimport (
    get_nested_resource_ptr as _cyb_get_nested_resource_ptr,
    get_resource_ptr as _cyb_get_resource_ptr,
    get_resource_ptrs as _cyb_get_resource_ptrs,
    nested_resource as _cyb_nested_resource,
    nullable_unique_ptr as _cyb_nullable_unique_ptr,
)
from libc.stdint cimport (
    int32_t,
    int64_t,
    intptr_t,
    uint32_t,
)
from libcpp.vector cimport vector as _cyb_vector

from enum import IntEnum as _cyb_IntEnum


# <<<< END OF PREAMBLE CONTENT >>>>

cimport cython  # NOQA
from libcpp.vector cimport vector

from ._utils cimport (get_resource_ptr, get_nested_resource_ptr, nested_resource, nullable_unique_ptr,
                      get_resource_ptrs)

###############################################################################
# Enum
###############################################################################

class CommunicatorStatus(_cyb_IntEnum):
    """
    Status code returned by communicator method functions.  This status
    code is returned by communicator functions defined in
    custatevecEx_ext.h. This enum only implements the success code. Other
    status codes are implementation dependent.

    See `custatevecExCommunicatorStatus_t`.
    """
    SUCCESS = CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS

class StateVectorCapability(_cyb_IntEnum):
    """
    Bitmask that specifies state vector capability. This enum is reserved
    for future use.

    See `custatevecExStateVectorCapability_t`.
    """
    NONE = CUSTATEVEC_EX_SV_CAPABILITY_NONE
    RESIZABLE = CUSTATEVEC_EX_SV_CAPABILITY_RESIZABLE

class StateVectorDistributionType(_cyb_IntEnum):
    """
    Enum that specifies the distribution type of state vector.

    See `custatevecExStateVectorDistributionType_t`.
    """
    SINGLE_DEVICE = CUSTATEVEC_EX_SV_DISTRIBUTION_SINGLE_DEVICE
    MULTI_DEVICE = CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_DEVICE
    MULTI_PROCESS = CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS

class IndexBitDomain(_cyb_IntEnum):
    """
    Specifies a domain that an index bit belongs to.  Each wire of a state
    vector is mapped to an index bit, and every index bit belongs to one of
    the following domains that determines how operations on the wire are
    executed:  - Local: the index bit is local to a sub state vector.
    Operations are executed without inter-device or inter-process
    communication.  - Migration: the index bit distinguishes sub state
    vectors placed on host versus device memory. Operations require host-
    device migration.  - Global device: the index bit distinguishes sub
    state vectors held by different devices or processes. Operations
    require data transfer routed by in-process GPUDirect P2P, inter-process
    GPUDirect P2P, or a communicator depending on the configuration.  The
    applicable index bit domains for each state vector distribution model
    are described in the documentation of
    `custatevecExConfigureStateVectorSingleDevice()`,
    `custatevecExConfigureStateVectorMultiDevice()`, and
    `custatevecExConfigureStateVectorMultiProcess()`.  This enum is used by
    `custatevecExStateVectorAddWires()` to specify the domain of index bits
    assigned to the newly added wires.

    See `custatevecExIndexBitDomain_t`.
    """
    LOCAL = CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL
    MIGRATION = CUSTATEVEC_EX_INDEX_BIT_DOMAIN_MIGRATION
    GLOBAL_DEVICE = CUSTATEVEC_EX_INDEX_BIT_DOMAIN_GLOBAL_DEVICE

class WireInitMode(_cyb_IntEnum):
    """
    Specifies the initialization mode of newly added wires.  This enum is
    used by `custatevecExStateVectorAddWires()` to specify how the newly
    added wires are initialized. The initialization is applied per wire:
    each new wire is independently set to the specified single-qubit state,
    and the state vector is updated as the tensor product with the
    resulting block of new wires.

    See `custatevecExWireInitMode_t`.
    """
    ZERO = CUSTATEVEC_EX_WIRE_INIT_MODE_ZERO

class GlobalIndexBitClass(_cyb_IntEnum):
    """
    Communication method for global index bit operations in multi-process
    distributions.  Operations on global index bits require data transfers.
    This enum specifies the communication method to use. It is used by
    `custatevecExConfigureStateVectorMultiProcess()` to describe the per-
    layer routing of inter-process global index bits.

    See `custatevecExGlobalIndexBitClass_t`.
    """
    INTERPROC_P2P = CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P
    COMMUNICATOR = CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_COMMUNICATOR
    MIGRATION = CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_MIGRATION

class StateVectorProperty(_cyb_IntEnum):
    """
    Specifies the name of state vector property.

    See `custatevecExStateVectorProperty_t`.
    """
    DISTRIBUTION_TYPE = CUSTATEVEC_EX_SV_PROP_DISTRIBUTION_TYPE
    DATA_TYPE = CUSTATEVEC_EX_SV_PROP_DATA_TYPE
    NUM_WIRES = CUSTATEVEC_EX_SV_PROP_NUM_WIRES
    WIRE_ORDERING = CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING
    NUM_LOCAL_WIRES = CUSTATEVEC_EX_SV_PROP_NUM_LOCAL_WIRES
    NUM_DEVICE_SUBSVS = CUSTATEVEC_EX_SV_PROP_NUM_DEVICE_SUBSVS
    DEVICE_SUBSV_INDICES = CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES
    NUM_INTERPROC_DEVICE_WIRES = CUSTATEVEC_EX_SV_PROP_NUM_INTERPROC_DEVICE_WIRES
    NUM_INPROC_DEVICE_WIRES = CUSTATEVEC_EX_SV_PROP_NUM_INPROC_DEVICE_WIRES
    NUM_MIGRATION_WIRES = CUSTATEVEC_EX_SV_PROP_NUM_MIGRATION_WIRES
    NUM_SUBSV_SLICES = CUSTATEVEC_EX_SV_PROP_NUM_SUBSV_SLICES
    NUM_SUBSVS = CUSTATEVEC_EX_SV_PROP_NUM_SUBSVS
    SUBSV_INDICES = CUSTATEVEC_EX_SV_PROP_SUBSV_INDICES
    NUM_UNSTAGED_SUBSVS = CUSTATEVEC_EX_SV_PROP_NUM_UNSTAGED_SUBSVS
    UNSTAGED_SUBSV_INDICES = CUSTATEVEC_EX_SV_PROP_UNSTAGED_SUBSV_INDICES
    MAX_NUM_WIRES = CUSTATEVEC_EX_SV_PROP_MAX_NUM_WIRES
    MAX_NUM_LOCAL_WIRES = CUSTATEVEC_EX_SV_PROP_MAX_NUM_LOCAL_WIRES
    MAX_NUM_INTERPROC_DEVICE_WIRES = CUSTATEVEC_EX_SV_PROP_MAX_NUM_INTERPROC_DEVICE_WIRES
    MAX_NUM_INPROC_DEVICE_WIRES = CUSTATEVEC_EX_SV_PROP_MAX_NUM_INPROC_DEVICE_WIRES
    MAX_NUM_MIGRATION_WIRES = CUSTATEVEC_EX_SV_PROP_MAX_NUM_MIGRATION_WIRES

class PermutationType(_cyb_IntEnum):
    """
    Specifies the permutation type.

    See `custatevecExPermutationType_t`.
    """
    SCATTER = CUSTATEVEC_EX_PERMUTATION_SCATTER
    GATHER = CUSTATEVEC_EX_PERMUTATION_GATHER

class ExposeResources(_cyb_IntEnum):
    """
    Resource exposure type.  Specifies the exposure type of state vector
    resources.

    See `custatevecExExposeResources_t`.
    """
    ACCESSIBLE = CUSTATEVEC_EX_EXPOSE_RESOURCES_ACCESSIBLE

class MatrixType(_cyb_IntEnum):
    """
    Specifies the type of matrix.

    See `custatevecExMatrixType_t`.
    """
    DENSE = CUSTATEVEC_EX_MATRIX_DENSE
    DIAGONAL = CUSTATEVEC_EX_MATRIX_DIAGONAL
    ANTI_DIAGONAL = CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL

class SVUpdaterConfigName(_cyb_IntEnum):
    """
    Specifies the configuration argument type of SVUpdater.

    See `custatevecExSVUpdaterConfigName_t`.
    """
    MAX_NUM_HOST_THREADS = CUSTATEVEC_EX_SVUPDATER_CONFIG_MAX_NUM_HOST_THREADS
    DENSE_FUSION_SIZE = CUSTATEVEC_EX_SVUPDATER_CONFIG_DENSE_FUSION_SIZE
    DIAGONAL_FUSION_SIZE = CUSTATEVEC_EX_SVUPDATER_CONFIG_DIAGONAL_FUSION_SIZE

class MemorySharingMethod(_cyb_IntEnum):
    """
    Specifies the method to share device virtual memory among processes.

    See `custatevecExMemorySharingMethod_t`.
    """
    AUTODETECT = CUSTATEVEC_EX_MEMORY_SHARING_METHOD_AUTODETECT
    NONE = CUSTATEVEC_EX_MEMORY_SHARING_METHOD_NONE
    FABRIC_HANDLE = CUSTATEVEC_EX_MEMORY_SHARING_METHOD_FABRIC_HANDLE
    PIDFD = CUSTATEVEC_EX_MEMORY_SHARING_METHOD_PIDFD

class MemoryPlacement(_cyb_IntEnum):
    """
    Memory placement type.  Specifies where a memory chunk is placed.

    See `custatevecExMemoryPlacement_t`.
    """
    ON_HOST = CUSTATEVEC_EX_MEMORY_PLACEMENT_ON_HOST

class SynchronizationScope(_cyb_IntEnum):
    """
    Specifies the scope of synchronization.  This enum is used by
    `custatevecExStateVectorSynchronizeScoped()`.

    See `custatevecExSynchronizationScope_t`.
    """
    PROCESS_LOCAL = CUSTATEVEC_EX_SYNCHRONIZATION_SCOPE_PROCESS_LOCAL
    GLOBAL = CUSTATEVEC_EX_SYNCHRONIZATION_SCOPE_GLOBAL


###############################################################################
# Error handling
###############################################################################

from .custatevec import cuStateVecError


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuStateVecError(status)


###############################################################################
# SVUpdater
###############################################################################

cdef struct _SVUpdaterConfigItemAlignmentProbe:
    char prefix
    _SVUpdaterConfigItem item


cdef inline void _check_sv_updater_config_item_abi() except *:
    """Fail at import if the handwritten Cython declaration drifts from the C ABI."""
    cdef _SVUpdaterConfigItem items[2]
    cdef _SVUpdaterConfigItemAlignmentProbe alignment_probe
    
    cdef intptr_t item_address = <intptr_t>&items[0]
    cdef intptr_t value_address = <intptr_t>&items[0].value
    cdef intptr_t next_item_address = <intptr_t>&items[1]

    cdef intptr_t probe_address = <intptr_t>&alignment_probe
    cdef intptr_t aligned_item_address = <intptr_t>&alignment_probe.item

    if (
        aligned_item_address - probe_address != 16
        or value_address - item_address != 16
        or next_item_address - item_address != 48
    ):
        raise RuntimeError(
            "custatevecExSVUpdaterConfigItem_t ABI mismatch: "
            "expected 16-byte alignment, value offset 16, and size 48"
        )


_check_sv_updater_config_item_abi()


cdef class SVUpdaterConfigItem:

    """One SVUpdater configuration name and its ``int32`` value.

    Pass a sequence of them to :func:`configure_sv_updater`; the binding creates
    the aligned native array.

    Attributes:
        name (int): The configuration item name; a
            :class:`SVUpdaterConfigName` value. See
            `custatevecExSVUpdaterConfigItem_t::name`.
        value (int): The item's ``int32`` value. See
            `custatevecExSVUpdaterConfigItem_t::value`.

    .. seealso:: `custatevecExSVUpdaterConfigItem_t`
    """

    cdef:
        readonly int name
        readonly int32_t value

    def __init__(self, int name, int32_t value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"SVUpdaterConfigItem(name={self.name}, value={self.value})"

    def __eq__(self, other):
        if not isinstance(other, SVUpdaterConfigItem):
            return NotImplemented
        return (
            self.name == (<SVUpdaterConfigItem>other).name
            and self.value == (<SVUpdaterConfigItem>other).value
        )


###############################################################################
# Wrapper functions
###############################################################################

cpdef dictionary_destroy(intptr_t dictionary):
    """Destroy dictionary instance.

    Args:
        dictionary (intptr_t): dictionary descriptor instance.

    .. seealso:: `custatevecExDictionaryDestroy`
    """
    with nogil:
        __status__ = custatevecExDictionaryDestroy(<DictionaryDescriptor>dictionary)
    check_status(__status__)


cpdef int communicator_finalize() except? -1:
    """Finalize inter-process communication library.

    Returns:
        int: status from communicator's finalize() method.

    .. seealso:: `custatevecExCommunicatorFinalize`
    """
    cdef _CommunicatorStatus ex_comm_status
    with nogil:
        __status__ = custatevecExCommunicatorFinalize(&ex_comm_status)
    check_status(__status__)
    return <int>ex_comm_status


cpdef tuple communicator_get_size_and_rank():
    """Get the global size and rank.

    Returns:
        A 3-tuple containing:
        - int32_t: global number of processes.
        - int32_t: global rank of the calling process.
        - int: status from communicator operations.

    .. seealso:: `custatevecExCommunicatorGetSizeAndRank`
    """
    cdef int32_t size
    cdef int32_t rank
    cdef _CommunicatorStatus ex_comm_status
    with nogil:
        __status__ = custatevecExCommunicatorGetSizeAndRank(&size, &rank, &ex_comm_status)
    check_status(__status__)
    return (size, rank, <int>ex_comm_status)


cpdef intptr_t communicator_create() except? 0:
    """Create communicator instance.

    Returns:
        intptr_t: created communicator instance.

    .. seealso:: `custatevecExCommunicatorCreate`
    """
    cdef CommunicatorDescriptor ex_communicator
    with nogil:
        __status__ = custatevecExCommunicatorCreate(&ex_communicator)
    check_status(__status__)
    return <intptr_t>ex_communicator


cpdef communicator_destroy(intptr_t ex_communicator):
    """Destroy communicator instance.

    Args:
        ex_communicator (intptr_t): communicator instance to destroy.

    .. seealso:: `custatevecExCommunicatorDestroy`
    """
    with nogil:
        __status__ = custatevecExCommunicatorDestroy(<CommunicatorDescriptor>ex_communicator)
    check_status(__status__)


cpdef intptr_t configure_state_vector_single_device(int sv_data_type, int32_t num_wires, int32_t num_device_wires, int32_t device_id, uint32_t capability) except? 0:
    """Create configuration for single device state vector.

    Args:
        sv_data_type (int): state vector data type.
        num_wires (int32_t): number of wires of state vector.
        num_device_wires (int32_t): number of wires of state vector on
            device.
        device_id (int32_t): device id where the entire state vector will
            be allocated.
        capability (uint32_t): bit mask to specify optional features of
            state vector.

    Returns:
        intptr_t: dictionary instance that holds state vector
            configuration.

    .. seealso:: `custatevecExConfigureStateVectorSingleDevice`
    """
    cdef DictionaryDescriptor sv_config
    with nogil:
        __status__ = custatevecExConfigureStateVectorSingleDevice(&sv_config, <DataType>sv_data_type, num_wires, num_device_wires, device_id, capability)
    check_status(__status__)
    return <intptr_t>sv_config


cpdef intptr_t configure_state_vector_multi_device(int sv_data_type, int32_t num_wires, int32_t num_device_wires, device_ids, int32_t num_devices, int network_type, uint32_t capability) except? 0:
    """Create configuration for multi-device state vector.

    Args:
        sv_data_type (int): state vector data type.
        num_wires (int32_t): number of wires of state vector.
        num_device_wires (int32_t): number of wires of state vector on
            each device.
        device_ids (object): host pointer to an array of device ids. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_devices (int32_t): number of devices.
        network_type (int): device network topology type.
        capability (uint32_t): bit mask to specify optional features of
            state vector.

    Returns:
        intptr_t: dictionary instance that holds state vector
            configuration.

    .. seealso:: `custatevecExConfigureStateVectorMultiDevice`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _device_ids_
    _cyb_get_resource_ptr[int32_t](_device_ids_, device_ids, <int32_t*>NULL)
    cdef DictionaryDescriptor sv_config
    with nogil:
        __status__ = custatevecExConfigureStateVectorMultiDevice(&sv_config, <DataType>sv_data_type, num_wires, num_device_wires, <const int32_t*>(_device_ids_.data()), num_devices, <custatevecDeviceNetworkType_t>network_type, capability)
    check_status(__status__)
    return <intptr_t>sv_config


cpdef intptr_t configure_state_vector_multi_process(int sv_data_type, int32_t num_wires, int32_t num_device_wires, int32_t device_id, int memory_sharing_method, global_index_bit_classes, num_global_index_bits_per_layer, int32_t num_global_index_bit_layers, size_t transfer_workspace_size_in_bytes, intptr_t aux_config, uint32_t capability) except? 0:
    """Create configuration for multi-process distributed state vector.

    Args:
        sv_data_type (int): state vector data type.
        num_wires (int32_t): number of wires of state vector.
        num_device_wires (int32_t): number of wires of state vector on
            each device.
        device_id (int32_t): device id for this process.
        memory_sharing_method (MemorySharingMethod): method for sharing
            GPU virtual device memory between processes.
        global_index_bit_classes (object): host pointer to an array of
            global index bit classes for each layer. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``_GlobalIndexBitClass``.
        num_global_index_bits_per_layer (object): host pointer to an array
            specifying number of global index bits per layer. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_global_index_bit_layers (int32_t): number of global index bit
            layers.
        transfer_workspace_size_in_bytes (size_t): size in bytes of
            workspace memory for inter-process data transfers.
        aux_config (intptr_t): pointer to auxiliary configuration.
        capability (uint32_t): bit mask to specify optional features of
            state vector.

    Returns:
        intptr_t: dictionary instance that holds state vector
            configuration.

    .. seealso:: `custatevecExConfigureStateVectorMultiProcess`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int] ] _global_index_bit_classes_
    _cyb_get_resource_ptr[int](_global_index_bit_classes_, global_index_bit_classes, <int*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _num_global_index_bits_per_layer_
    _cyb_get_resource_ptr[int32_t](_num_global_index_bits_per_layer_, num_global_index_bits_per_layer, <int32_t*>NULL)
    cdef DictionaryDescriptor sv_config
    with nogil:
        __status__ = custatevecExConfigureStateVectorMultiProcess(&sv_config, <DataType>sv_data_type, num_wires, num_device_wires, device_id, <_MemorySharingMethod>memory_sharing_method, <const _GlobalIndexBitClass*>(_global_index_bit_classes_.data()), <const int32_t*>(_num_global_index_bits_per_layer_.data()), num_global_index_bit_layers, transfer_workspace_size_in_bytes, <const void*>aux_config, capability)
    check_status(__status__)
    return <intptr_t>sv_config


cpdef intptr_t state_vector_create_single_process(intptr_t sv_config, streams, int32_t num_streams, intptr_t resource_manager) except? 0:
    """Create state vector.

    Args:
        sv_config (intptr_t): state vector configuration created by a
            state vector configuration function.
        streams (object): a pointer to a host array that holds CUDA
            streams. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``intptr_t``.
        num_streams (int32_t): the number of streams given by the streams
            argument.
        resource_manager (intptr_t): resource manager.

    Returns:
        intptr_t: state vector instance.

    .. seealso:: `custatevecExStateVectorCreateSingleProcess`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[intptr_t] ] _streams_
    _cyb_get_resource_ptr[intptr_t](_streams_, streams, <intptr_t*>NULL)
    cdef StateVectorDescriptor state_vector
    with nogil:
        __status__ = custatevecExStateVectorCreateSingleProcess(&state_vector, <const DictionaryDescriptor>sv_config, <const Stream*>(_streams_.data()), num_streams, <ResourceManagerDescriptor>resource_manager)
    check_status(__status__)
    return <intptr_t>state_vector


cpdef intptr_t state_vector_create_multi_process(intptr_t sv_config, intptr_t stream, intptr_t ex_communicator, intptr_t resource_manager) except? 0:
    """Create multi-process distributed state vector.

    Args:
        sv_config (intptr_t): state vector configuration created by
            custatevecExConfigureStateVectorMultiProcess.
        stream (intptr_t): CUDA stream for this process.
        ex_communicator (intptr_t): communicator descriptor for inter-
            process communication.
        resource_manager (intptr_t): resource manager.

    Returns:
        intptr_t: state vector instance.

    .. seealso:: `custatevecExStateVectorCreateMultiProcess`
    """
    cdef StateVectorDescriptor state_vector
    with nogil:
        __status__ = custatevecExStateVectorCreateMultiProcess(&state_vector, <const DictionaryDescriptor>sv_config, <Stream>stream, <CommunicatorDescriptor>ex_communicator, <ResourceManagerDescriptor>resource_manager)
    check_status(__status__)
    return <intptr_t>state_vector


cpdef state_vector_destroy(intptr_t state_vector):
    """Destroy state vector instance.

    Args:
        state_vector (intptr_t): state vector instance.

    .. seealso:: `custatevecExStateVectorDestroy`
    """
    with nogil:
        __status__ = custatevecExStateVectorDestroy(<StateVectorDescriptor>state_vector)
    check_status(__status__)


cpdef state_vector_get_property(intptr_t state_vector, int property, intptr_t value, size_t size_in_bytes):
    """Retrieve state vector properties.

    Args:
        state_vector (intptr_t): state vector instance.
        property (StateVectorProperty): a value of
            ``custatevecExStateVectorProperty_t``.
        value (intptr_t): host pointer to a host buffer that receives the
            value of the specified property.
        size_in_bytes (size_t): byte size of the value buffer.

    .. seealso:: `custatevecExStateVectorGetProperty`
    """
    with nogil:
        __status__ = custatevecExStateVectorGetProperty(<const StateVectorDescriptor>state_vector, <_StateVectorProperty>property, <void*>value, size_in_bytes)
    check_status(__status__)


cpdef state_vector_set_math_mode(intptr_t state_vector, int mode):
    """Set the compute precision mode for a state vector instance.

    Args:
        state_vector (intptr_t): state vector instance.
        mode (int): Compute precision mode as defined by
            ``custatevecMathMode_t``.

    .. seealso:: `custatevecExStateVectorSetMathMode`
    """
    with nogil:
        __status__ = custatevecExStateVectorSetMathMode(<StateVectorDescriptor>state_vector, <custatevecMathMode_t>mode)
    check_status(__status__)


cpdef state_vector_set_zero_state(intptr_t state_vector):
    """Set the zero state to state vector.

    Args:
        state_vector (intptr_t): state vector instance.

    .. seealso:: `custatevecExStateVectorSetZeroState`
    """
    with nogil:
        __status__ = custatevecExStateVectorSetZeroState(<StateVectorDescriptor>state_vector)
    check_status(__status__)


cpdef state_vector_get_state(intptr_t state_vector, intptr_t state, int data_type, int64_t begin, int64_t end, int32_t max_num_concurrent_copies):
    """Copy state vector elements to host buffer.

    Args:
        state_vector (intptr_t): state vector instance.
        state (intptr_t): pointer to a host buffer that receives state
            vector elements.
        data_type (int): data_type of the state vector elements.
        begin (int64_t): index of state vector element where the copy
            begins.
        end (int64_t): index of state vector element where the copy ends.
        max_num_concurrent_copies (int32_t): Max number of parallel
            copies.

    .. seealso:: `custatevecExStateVectorGetState`
    """
    with nogil:
        __status__ = custatevecExStateVectorGetState(<const StateVectorDescriptor>state_vector, <void*>state, <DataType>data_type, <custatevecIndex_t>begin, <custatevecIndex_t>end, max_num_concurrent_copies)
    check_status(__status__)


cpdef state_vector_set_state(intptr_t state_vector, intptr_t state, int data_type, int64_t begin, int64_t end, int32_t max_num_concurrent_copies):
    """Set complex value array on host to state vector.

    Args:
        state_vector (intptr_t): state vector instance.
        state (intptr_t): pointer to a complex vector on host.
        data_type (int): data_type of the state vector elements.
        begin (int64_t): index of state vector element where the copy
            begins.
        end (int64_t): index of state vector element where the copy ends.
        max_num_concurrent_copies (int32_t): Max number of parallel
            copies.

    .. seealso:: `custatevecExStateVectorSetState`
    """
    with nogil:
        __status__ = custatevecExStateVectorSetState(<StateVectorDescriptor>state_vector, <const void*>state, <DataType>data_type, <custatevecIndex_t>begin, <custatevecIndex_t>end, max_num_concurrent_copies)
    check_status(__status__)


cpdef state_vector_reassign_wire_ordering(intptr_t state_vector, wire_ordering, int32_t wire_ordering_len):
    """Reassign wire ordering to state vector.

    Args:
        state_vector (intptr_t): state vector instance.
        wire_ordering (object): the pointer to a integer host array that
            holds wire ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        wire_ordering_len (int32_t): the length of wire ordering.

    .. seealso:: `custatevecExStateVectorReassignWireOrdering`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _wire_ordering_
    _cyb_get_resource_ptr[int32_t](_wire_ordering_, wire_ordering, <int32_t*>NULL)
    with nogil:
        __status__ = custatevecExStateVectorReassignWireOrdering(<StateVectorDescriptor>state_vector, <const int32_t*>(_wire_ordering_.data()), wire_ordering_len)
    check_status(__status__)


cpdef state_vector_permute_index_bits(intptr_t state_vector, permutation, int32_t permutation_len, int permutation_type):
    """Permute index bits and wires of state vector.

    Args:
        state_vector (intptr_t): StateVector instance.
        permutation (object): a host pointer to an integer array
            specifying the permutation. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        permutation_len (int32_t): length of the permutation.
        permutation_type (PermutationType): permutation type (scatter or
            gather).

    .. seealso:: `custatevecExStateVectorPermuteIndexBits`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _permutation_
    _cyb_get_resource_ptr[int32_t](_permutation_, permutation, <int32_t*>NULL)
    with nogil:
        __status__ = custatevecExStateVectorPermuteIndexBits(<StateVectorDescriptor>state_vector, <const int32_t*>(_permutation_.data()), permutation_len, <_PermutationType>permutation_type)
    check_status(__status__)


cpdef state_vector_stage_sub_sv(intptr_t state_vector, int32_t sub_sv_index):
    """Stage a sub state vector onto device memory.

    Args:
        state_vector (intptr_t): StateVector instance.
        sub_sv_index (int32_t): Sub state vector index to stage onto
            device.

    .. seealso:: `custatevecExStateVectorStageSubSV`
    """
    with nogil:
        __status__ = custatevecExStateVectorStageSubSV(<StateVectorDescriptor>state_vector, sub_sv_index)
    check_status(__status__)


cpdef state_vector_expose_resources(intptr_t state_vector, int expose_resources):
    """Expose state vector resources for access.

    Args:
        state_vector (intptr_t): StateVector instance.
        expose_resources (ExposeResources): Layout type for exposing
            resources.

    .. seealso:: `custatevecExStateVectorExposeResources`
    """
    with nogil:
        __status__ = custatevecExStateVectorExposeResources(<StateVectorDescriptor>state_vector, <_ExposeResources>expose_resources)
    check_status(__status__)


cpdef state_vector_synchronize(intptr_t state_vector):
    """Flush all operations and synchronize.

    Args:
        state_vector (intptr_t): state vector instance.

    .. seealso:: `custatevecExStateVectorSynchronize`
    """
    with nogil:
        __status__ = custatevecExStateVectorSynchronize(<StateVectorDescriptor>state_vector)
    check_status(__status__)


cpdef state_vector_synchronize_scoped(intptr_t state_vector, int scope):
    """Flush all operations and synchronize within the specified scope.

    Args:
        state_vector (intptr_t): state vector instance.
        scope (SynchronizationScope): the scope of synchronization.

    .. seealso:: `custatevecExStateVectorSynchronizeScoped`
    """
    with nogil:
        __status__ = custatevecExStateVectorSynchronizeScoped(<StateVectorDescriptor>state_vector, <_SynchronizationScope>scope)
    check_status(__status__)


cpdef state_vector_add_wires(intptr_t state_vector, int index_bit_domain, int32_t num_wires_to_add, int wire_init_mode, intptr_t wires_added):
    """Add wires to a state vector.

    Args:
        state_vector (intptr_t): state vector instance.
        index_bit_domain (IndexBitDomain): the domain of index bits to
            which the new wires are assigned.
        num_wires_to_add (int32_t): the number of wires to add.
        wire_init_mode (WireInitMode): initialization mode applied to each
            newly added wire.
        wires_added (intptr_t): host pointer to an int32_t array that
            receives the IDs of the newly added wires. The array length
            must be at least ``num_wires_to_add``.

    .. seealso:: `custatevecExStateVectorAddWires`
    """
    with nogil:
        __status__ = custatevecExStateVectorAddWires(<StateVectorDescriptor>state_vector, <_IndexBitDomain>index_bit_domain, num_wires_to_add, <_WireInitMode>wire_init_mode, <int32_t*>wires_added)
    check_status(__status__)


cpdef abs2sum_array(intptr_t state_vector, intptr_t abs2sum, output_ordering, int32_t output_ordering_len, mask_bit_string, mask_wire_ordering, int32_t mask_len):
    """Calculate abs2sum array for a given set of wires.

    Args:
        state_vector (intptr_t): StateVector instance.
        abs2sum (intptr_t): pointer to a host or device array of sums of
            squared absolute values.
        output_ordering (object): pointer to a host array of output tensor
            ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        output_ordering_len (int32_t): the length of output_ordering.
        mask_bit_string (object): pointer to a host array for a bit string
            to specify mask bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        mask_wire_ordering (object): pointer to a host array that
            specifies the wire ordering of mask_bit_string. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        mask_len (int32_t): the length of mask.

    .. seealso:: `custatevecExAbs2SumArray`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _output_ordering_
    _cyb_get_resource_ptr[int32_t](_output_ordering_, output_ordering, <int32_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _mask_bit_string_
    _cyb_get_resource_ptr[int32_t](_mask_bit_string_, mask_bit_string, <int32_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _mask_wire_ordering_
    _cyb_get_resource_ptr[int32_t](_mask_wire_ordering_, mask_wire_ordering, <int32_t*>NULL)
    with nogil:
        __status__ = custatevecExAbs2SumArray(<StateVectorDescriptor>state_vector, <double*>abs2sum, <const int32_t*>(_output_ordering_.data()), output_ordering_len, <const int32_t*>(_mask_bit_string_.data()), <const int32_t*>(_mask_wire_ordering_.data()), mask_len)
    check_status(__status__)


cpdef int64_t measure(intptr_t state_vector, bit_string_ordering, int32_t bit_string_ordering_len, double randnum, int collapse, intptr_t reserved) except? -1:
    """Perform qubit measurements.

    Args:
        state_vector (intptr_t): StateVector instance.
        bit_string_ordering (object): pointer to a host array of bit
            string ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        bit_string_ordering_len (int32_t): length of bit_stringOrdering.
        randnum (double): random number, [0, 1).
        collapse (int): Collapse operation.
        reserved (intptr_t): Reserved argument. A null pointer should be
            passed.

    Returns:
        int64_t: measured bit string.

    .. seealso:: `custatevecExMeasure`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _bit_string_ordering_
    _cyb_get_resource_ptr[int32_t](_bit_string_ordering_, bit_string_ordering, <int32_t*>NULL)
    cdef custatevecIndex_t bit_string
    with nogil:
        __status__ = custatevecExMeasure(<StateVectorDescriptor>state_vector, &bit_string, <const int32_t*>(_bit_string_ordering_.data()), bit_string_ordering_len, randnum, <custatevecCollapseOp_t>collapse, <const void*>reserved)
    check_status(__status__)
    return <int64_t>bit_string


cpdef sample(intptr_t state_vector, intptr_t bit_strings, bit_string_ordering, int32_t bit_string_ordering_len, randnums, int32_t num_shots, int output, abs2sums):
    """Sample bit strings from the state vector.

    Args:
        state_vector (intptr_t): State vector instance.
        bit_strings (intptr_t): pointer to a host array to store sampled
            bit strings.
        bit_string_ordering (object): pointer to a host array of bit
            string ordering for sampling. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        bit_string_ordering_len (int32_t): length of bit_string_ordering.
        randnums (object): pointer to an array of random numbers. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.
        num_shots (int32_t): the number of shots.
        output (int): the order of sampled bit strings.
        abs2sums (object): pointer to a host array of abs2 sums for each
            sub state vector, or null. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.

    .. seealso:: `custatevecExSample`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _bit_string_ordering_
    _cyb_get_resource_ptr[int32_t](_bit_string_ordering_, bit_string_ordering, <int32_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[double] ] _randnums_
    _cyb_get_resource_ptr[double](_randnums_, randnums, <double*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[double] ] _abs2sums_
    _cyb_get_resource_ptr[double](_abs2sums_, abs2sums, <double*>NULL)
    with nogil:
        __status__ = custatevecExSample(<StateVectorDescriptor>state_vector, <custatevecIndex_t*>bit_strings, <const int32_t*>(_bit_string_ordering_.data()), bit_string_ordering_len, <const double*>(_randnums_.data()), num_shots, <custatevecSamplerOutput_t>output, <const double*>(_abs2sums_.data()))
    check_status(__status__)


cpdef apply_matrix(intptr_t state_vector, intptr_t matrix, int matrix_data_type, int ex_matrix_type, int layout, int32_t adjoint, targets, int32_t num_targets, controls, control_bit_values, int32_t num_controls):
    """Apply gate matrix.

    Args:
        state_vector (intptr_t): state vector instance.
        matrix (intptr_t): pointer to a buffer that holds matrix elements.
        matrix_data_type (int): data type of matrix.
        ex_matrix_type (MatrixType): enumerator specifying the matrix type
            and layout.
        layout (int): enumerator specifying the matrix layout.
        adjoint (int32_t): apply adjoint of matrix.
        targets (object): pointer to a host array of target wires. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_targets (int32_t): the number of target wires.
        controls (object): pointer to a host array of control wires. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        control_bit_values (object): pointer to a host array of control
            bit values. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_controls (int32_t): the number of control wires.

    .. seealso:: `custatevecExApplyMatrix`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _targets_
    _cyb_get_resource_ptr[int32_t](_targets_, targets, <int32_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _controls_
    _cyb_get_resource_ptr[int32_t](_controls_, controls, <int32_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _control_bit_values_
    _cyb_get_resource_ptr[int32_t](_control_bit_values_, control_bit_values, <int32_t*>NULL)
    with nogil:
        __status__ = custatevecExApplyMatrix(<StateVectorDescriptor>state_vector, <const void*>matrix, <DataType>matrix_data_type, <_MatrixType>ex_matrix_type, <custatevecMatrixLayout_t>layout, adjoint, <const int32_t*>(_targets_.data()), num_targets, <const int32_t*>(_controls_.data()), <const int32_t*>(_control_bit_values_.data()), num_controls)
    check_status(__status__)


cpdef apply_pauli_rotation(intptr_t state_vector, double theta, paulis, targets, int32_t num_targets, controls, control_bit_values, int32_t num_controls):
    """Apply the exponential of a multi-qubit Pauli operator.

    Args:
        state_vector (intptr_t): state vector instance.
        theta (double): theta.
        paulis (object): host pointer to ``custatevecPauli_t`` array. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``custatevecPauli_t``.
        targets (object): pointer to a host array of target wires. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_targets (int32_t): the number of target wires.
        controls (object): pointer to a host array of control wires. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        control_bit_values (object): pointer to a host array of control
            bit values. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_controls (int32_t): the number of control wires.

    .. seealso:: `custatevecExApplyPauliRotation`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[custatevecPauli_t] ] _paulis_
    _cyb_get_resource_ptr[custatevecPauli_t](_paulis_, paulis, <custatevecPauli_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _targets_
    _cyb_get_resource_ptr[int32_t](_targets_, targets, <int32_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _controls_
    _cyb_get_resource_ptr[int32_t](_controls_, controls, <int32_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _control_bit_values_
    _cyb_get_resource_ptr[int32_t](_control_bit_values_, control_bit_values, <int32_t*>NULL)
    with nogil:
        __status__ = custatevecExApplyPauliRotation(<StateVectorDescriptor>state_vector, theta, <const custatevecPauli_t*>(_paulis_.data()), <const int32_t*>(_targets_.data()), num_targets, <const int32_t*>(_controls_.data()), <const int32_t*>(_control_bit_values_.data()), num_controls)
    check_status(__status__)


cpdef compute_expectation_on_pauli_basis(intptr_t state_vector, intptr_t expectation_values, pauli_operator_arrays, int32_t num_pauli_operator_arrays, basis_wires_array, num_basis_wires_array):
    """Compute expectation values for a batch of (multi-qubit) Pauli operators.

    Args:
        state_vector (intptr_t): state vector instance.
        expectation_values (intptr_t): pointer to a host array to store
            expectation values.
        pauli_operator_arrays (object): pointer to a host array of Pauli
            operator arrays. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'custatevecPauli_t', or
            - a nested Python sequence of ``custatevecPauli_t``.
        num_pauli_operator_arrays (int32_t): the number of Pauli operator
            arrays.
        basis_wires_array (object): host array of basis wire arrays. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int32_t', or
            - a nested Python sequence of ``int32_t``.
        num_basis_wires_array (object): host array of the number of basis
            wires. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

    .. seealso:: `custatevecExComputeExpectationOnPauliBasis`
    """
    cdef _cyb_nested_resource[ custatevecPauli_t ] _pauli_operator_arrays_
    _cyb_get_nested_resource_ptr[custatevecPauli_t](_pauli_operator_arrays_, pauli_operator_arrays, <custatevecPauli_t*>NULL)
    cdef _cyb_nested_resource[ int32_t ] _basis_wires_array_
    _cyb_get_nested_resource_ptr[int32_t](_basis_wires_array_, basis_wires_array, <int32_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _num_basis_wires_array_
    _cyb_get_resource_ptr[int32_t](_num_basis_wires_array_, num_basis_wires_array, <int32_t*>NULL)
    with nogil:
        __status__ = custatevecExComputeExpectationOnPauliBasis(<StateVectorDescriptor>state_vector, <double*>expectation_values, <const custatevecPauli_t**>(_pauli_operator_arrays_.ptrs.data()), num_pauli_operator_arrays, <const int32_t**>(_basis_wires_array_.ptrs.data()), <const int32_t*>(_num_basis_wires_array_.data()))
    check_status(__status__)


cpdef compute_expectation(intptr_t state_vector, intptr_t expectation_values, intptr_t matrices, int matrix_data_type, int layout, int32_t num_matrices, basis_wires, int32_t num_basis_wires):
    """Compute expectation values for a batch of matrix observables.

    Args:
        state_vector (intptr_t): state vector instance.
        expectation_values (intptr_t): pointer to a host array to store
            expectation values.
        matrices (intptr_t): pointer to a buffer that holds matrix
            observables.
        matrix_data_type (int): data type of matrices.
        layout (int): enumerator specifying the memory layout of matrices.
        num_matrices (int32_t): the number of matrix observables.
        basis_wires (object): pointer to a host array of basis wires. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_basis_wires (int32_t): the number of basis wires.

    .. seealso:: `custatevecExComputeExpectation`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _basis_wires_
    _cyb_get_resource_ptr[int32_t](_basis_wires_, basis_wires, <int32_t*>NULL)
    with nogil:
        __status__ = custatevecExComputeExpectation(<StateVectorDescriptor>state_vector, <double2*>expectation_values, <const void*>matrices, <DataType>matrix_data_type, <custatevecMatrixLayout_t>layout, num_matrices, <const int32_t*>(_basis_wires_.data()), num_basis_wires)
    check_status(__status__)


cpdef intptr_t sv_updater_create(intptr_t sv_updater_config, intptr_t resource_manager) except? 0:
    """Create SVUpdater.

    Args:
        sv_updater_config (intptr_t): SVUpdater configuration.
        resource_manager (intptr_t): resource manager.

    Returns:
        intptr_t: created SVUpdater.

    .. seealso:: `custatevecExSVUpdaterCreate`
    """
    cdef SVUpdaterDescriptor sv_updater
    with nogil:
        __status__ = custatevecExSVUpdaterCreate(&sv_updater, <const DictionaryDescriptor>sv_updater_config, <ResourceManagerDescriptor>resource_manager)
    check_status(__status__)
    return <intptr_t>sv_updater


cpdef sv_updater_destroy(intptr_t sv_updater):
    """Destroy SVUpdater instance.

    Args:
        sv_updater (intptr_t): SVUpdater instance.

    .. seealso:: `custatevecExSVUpdaterDestroy`
    """
    with nogil:
        __status__ = custatevecExSVUpdaterDestroy(<SVUpdaterDescriptor>sv_updater)
    check_status(__status__)


cpdef sv_updater_clear(intptr_t sv_updater):
    """Clear operations queued in SVUpdater.

    Args:
        sv_updater (intptr_t): SVUpdater instance.

    .. seealso:: `custatevecExSVUpdaterClear`
    """
    with nogil:
        __status__ = custatevecExSVUpdaterClear(<SVUpdaterDescriptor>sv_updater)
    check_status(__status__)


cpdef sv_updater_enqueue_matrix(intptr_t sv_updater, intptr_t matrix, int matrix_data_type, int ex_matrix_type, int layout, int32_t adjoint, targets, int32_t num_targets, controls, control_bit_values, int32_t num_controls):
    """Enqueue matrix to SVUpdater.

    Args:
        sv_updater (intptr_t): SVUpdater instance.
        matrix (intptr_t): pointer to a host buffer that holds matrix
            elements.
        matrix_data_type (int): data type of matrix.
        ex_matrix_type (MatrixType): enumerator specifying the matrix
            type.
        layout (int): enumerator specifying the matrix layout.
        adjoint (int32_t): apply adjoint of matrix.
        targets (object): pointer to a host array of target wires. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_targets (int32_t): the number of target wires.
        controls (object): pointer to a host array of control wires. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        control_bit_values (object): pointer to a host array of control
            bit values. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_controls (int32_t): the number of control wires.

    .. seealso:: `custatevecExSVUpdaterEnqueueMatrix`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _targets_
    _cyb_get_resource_ptr[int32_t](_targets_, targets, <int32_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _controls_
    _cyb_get_resource_ptr[int32_t](_controls_, controls, <int32_t*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _control_bit_values_
    _cyb_get_resource_ptr[int32_t](_control_bit_values_, control_bit_values, <int32_t*>NULL)
    with nogil:
        __status__ = custatevecExSVUpdaterEnqueueMatrix(<SVUpdaterDescriptor>sv_updater, <const void*>matrix, <DataType>matrix_data_type, <_MatrixType>ex_matrix_type, <custatevecMatrixLayout_t>layout, adjoint, <const int32_t*>(_targets_.data()), num_targets, <const int32_t*>(_controls_.data()), <const int32_t*>(_control_bit_values_.data()), num_controls)
    check_status(__status__)


cpdef sv_updater_enqueue_unitary_channel(intptr_t sv_updater, unitaries, int unitaries_data_type, ex_matrix_types, int32_t num_unitaries, int layout, probabilities, channel_wires, int32_t num_channel_wires):
    """Enqueue mixed unitary channel.

    Args:
        sv_updater (intptr_t): SVUpdater instance.
        unitaries (object): host pointer to an array of unitary matrix
            elements. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).
        unitaries_data_type (int): dataType of the specified unitary
            matrices.
        ex_matrix_types (object): matrix types of the specified unitary
            matrices. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``_MatrixType``.
        num_unitaries (int32_t): the number of matrices.
        layout (int): layout of the specified unitary matrices.
        probabilities (object): host array that holds the probabilities. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.
        channel_wires (object): wires that sampled unitary channel is
            applied. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_channel_wires (int32_t): the number of wires.

    .. seealso:: `custatevecExSVUpdaterEnqueueUnitaryChannel`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[void*] ] _unitaries_
    _cyb_get_resource_ptrs[void](_unitaries_, unitaries, <void*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int] ] _ex_matrix_types_
    _cyb_get_resource_ptr[int](_ex_matrix_types_, ex_matrix_types, <int*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[double] ] _probabilities_
    _cyb_get_resource_ptr[double](_probabilities_, probabilities, <double*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _channel_wires_
    _cyb_get_resource_ptr[int32_t](_channel_wires_, channel_wires, <int32_t*>NULL)
    with nogil:
        __status__ = custatevecExSVUpdaterEnqueueUnitaryChannel(<SVUpdaterDescriptor>sv_updater, <const void* const*>(_unitaries_.data()), <DataType>unitaries_data_type, <const _MatrixType*>(_ex_matrix_types_.data()), num_unitaries, <custatevecMatrixLayout_t>layout, <const double*>(_probabilities_.data()), <const int32_t*>(_channel_wires_.data()), num_channel_wires)
    check_status(__status__)


cpdef sv_updater_enqueue_general_channel(intptr_t sv_updater, matrices, int matrix_data_type, ex_matrix_types, int32_t num_matrices, int layout, channel_wires, int32_t num_channel_wires):
    """Enqueue general channel.

    Args:
        sv_updater (intptr_t): SVUpdater instance.
        matrices (object): host pointer to an array of matrix elements. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).
        matrix_data_type (int): dataType of the matrices.
        ex_matrix_types (object): matrix types of the specified matrices. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``_MatrixType``.
        num_matrices (int32_t): the number of matrices.
        layout (int): layout of the matrices.
        channel_wires (object): wires that the general channel is applied. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.
        num_channel_wires (int32_t): the number of wires.

    .. seealso:: `custatevecExSVUpdaterEnqueueGeneralChannel`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[void*] ] _matrices_
    _cyb_get_resource_ptrs[void](_matrices_, matrices, <void*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int] ] _ex_matrix_types_
    _cyb_get_resource_ptr[int](_ex_matrix_types_, ex_matrix_types, <int*>NULL)
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[int32_t] ] _channel_wires_
    _cyb_get_resource_ptr[int32_t](_channel_wires_, channel_wires, <int32_t*>NULL)
    with nogil:
        __status__ = custatevecExSVUpdaterEnqueueGeneralChannel(<SVUpdaterDescriptor>sv_updater, <const void* const*>(_matrices_.data()), <DataType>matrix_data_type, <const _MatrixType*>(_ex_matrix_types_.data()), num_matrices, <custatevecMatrixLayout_t>layout, <const int32_t*>(_channel_wires_.data()), num_channel_wires)
    check_status(__status__)


cpdef int32_t sv_updater_get_max_num_required_randnums(intptr_t sv_updater) except? -1:
    """Get the max number of required random numbers.

    Args:
        sv_updater (intptr_t): SVUpdater instance.

    Returns:
        int32_t: the max required number of random numbers.

    .. seealso:: `custatevecExSVUpdaterGetMaxNumRequiredRandnums`
    """
    cdef int32_t max_num_required_randnums
    with nogil:
        __status__ = custatevecExSVUpdaterGetMaxNumRequiredRandnums(<SVUpdaterDescriptor>sv_updater, &max_num_required_randnums)
    check_status(__status__)
    return max_num_required_randnums


cpdef sv_updater_apply(intptr_t sv_updater, intptr_t state_vector, randnums, int32_t num_randnums):
    """Apply queued operations.

    Args:
        sv_updater (intptr_t): SVUpdater instance.
        state_vector (intptr_t): state vector instance.
        randnums (object): a host pointer to an array of random numbers in
            the range [0, 1). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.
        num_randnums (int32_t): the number of random numbers.

    .. seealso:: `custatevecExSVUpdaterApply`
    """
    cdef _cyb_nullable_unique_ptr[ _cyb_vector[double] ] _randnums_
    _cyb_get_resource_ptr[double](_randnums_, randnums, <double*>NULL)
    with nogil:
        __status__ = custatevecExSVUpdaterApply(<SVUpdaterDescriptor>sv_updater, <StateVectorDescriptor>state_vector, <const double*>(_randnums_.data()), num_randnums)
    check_status(__status__)


cpdef intptr_t configure_sv_updater(
        int data_type, object config_items=()) except? 0:
    """Configure an SVUpdater and return its configuration dictionary.

    Args:
        data_type (int): data type used internally by the SVUpdater.
        config_items (object): a Python sequence of
            :class:`SVUpdaterConfigItem`; an empty sequence selects the
            system defaults.

    Returns:
        intptr_t: configuration dictionary to pass to
            :func:`sv_updater_create`. The caller must destroy it with
            :func:`dictionary_destroy`.

    .. seealso:: `custatevecExConfigureSVUpdater`
    """
    cdef Py_ssize_t num_config_items = len(config_items)
    cdef vector[_SVUpdaterConfigItem] _config_items_
    cdef SVUpdaterConfigItem _item_
    cdef const _SVUpdaterConfigItem* config_items_ptr = NULL
    cdef DictionaryDescriptor sv_updater_config
    cdef object item
    cdef Py_ssize_t i

    # Repack Python configuration items into an ABI-aligned contiguous C array.
    _config_items_.resize(num_config_items)
    for i in range(num_config_items):
        item = config_items[i]
        if not isinstance(item, SVUpdaterConfigItem):
            raise TypeError(f"config_items[{i}] must be an SVUpdaterConfigItem")
        _item_ = item
        _config_items_[i].name = <custatevecExSVUpdaterConfigName_t>_item_.name
        _config_items_[i].value.int32 = _item_.value

    if num_config_items:
        config_items_ptr = &_config_items_[0]

    with nogil:
        __status__ = custatevecExConfigureSVUpdater(
            &sv_updater_config,
            <DataType>data_type,
            config_items_ptr,
            <int32_t>num_config_items,
        )
    check_status(__status__)
    return <intptr_t>sv_updater_config


cpdef int communicator_initialize(
        int communicator_type, str library_path=None) except? -1:
    """Initialize inter-process communication.

    ``argc`` and ``argv`` are always passed as NULL to the underlying C API.
    Default MPI initialization works under modern launchers without them; for
    custom setups, initialize MPI beforehand (e.g. via mpi4py or a prior
    ``MPI_Init``) and then call this function.

    Args:
        communicator_type (CommunicatorType): The communicator type.
        library_path (str): Path to the inter-process communication library.
            If None, the C API selects a default based on ``communicator_type``.

    Returns:
        int: The communicator status from the underlying init() method.

    .. seealso:: `custatevecExCommunicatorInitialize`
    """
    cdef bytes _library_path_
    cdef const char* library_path_ptr = NULL
    if library_path is not None:
        _library_path_ = library_path.encode()
        library_path_ptr = _library_path_
    cdef custatevecExCommunicatorStatus_t ex_comm_status

    with nogil:
        status = custatevecExCommunicatorInitialize(
            <custatevecCommunicatorType_t>communicator_type,
            library_path_ptr,
            NULL, NULL,
            &ex_comm_status)
    check_status(status)
    return <int>ex_comm_status


cpdef tuple state_vector_get_resources_from_device_sub_sv(
        intptr_t state_vector, int32_t sub_sv_index):
    """Get resources from device sub state vector.

    Args:
        state_vector (intptr_t): The state vector descriptor.
        sub_sv_index (int32_t): Sub state vector index.

    Returns:
        A 4-tuple containing:

        - int32_t: device id of the sub state vector.
        - intptr_t: device memory pointer to the sub state vector.
        - intptr_t: CUDA stream on which custatevecEx calls for this sub state
          vector are serialized.
        - intptr_t: cuStateVec handle that can be passed to cuStateVec APIs to
          operate on the returned device pointer.

    The handle output is optional in the underlying C API (a null pointer may
    be passed to omit it); this binding always returns it, since reading the
    eagerly-created handle adds no overhead and keeps the API uniform.

    .. seealso:: `custatevecExStateVectorGetResourcesFromDeviceSubSV`
    """
    cdef int32_t device_id
    cdef void* d_sub_sv
    cdef cudaStream_t stream
    cdef custatevecHandle_t handle

    with nogil:
        status = custatevecExStateVectorGetResourcesFromDeviceSubSV(
            <StateVectorDescriptor>state_vector,
            sub_sv_index,
            &device_id, &d_sub_sv, &stream, &handle)
    check_status(status)
    return (device_id, <intptr_t>d_sub_sv, <intptr_t>stream, <intptr_t>handle)


cpdef tuple state_vector_get_resources_from_device_sub_sv_view(
        intptr_t state_vector, int32_t sub_sv_index):
    """Get resources from device sub state vector (read-only view).

    Args:
        state_vector (intptr_t): The state vector descriptor.
        sub_sv_index (int32_t): Sub state vector index.

    Returns:
        A 4-tuple containing:

        - int32_t: device id of the sub state vector.
        - intptr_t: device memory pointer to the sub state vector (read-only
          view; do not write through it).
        - intptr_t: CUDA stream on which custatevecEx calls for this sub state
          vector are serialized.
        - intptr_t: cuStateVec handle that can be passed to cuStateVec APIs to
          operate on the returned device pointer.

    The handle output is optional in the underlying C API (a null pointer may
    be passed to omit it); this binding always returns it, since reading the
    eagerly-created handle adds no overhead and keeps the API uniform.

    .. seealso:: `custatevecExStateVectorGetResourcesFromDeviceSubSVView`
    """
    cdef int32_t device_id
    cdef const void* d_sub_sv
    cdef cudaStream_t stream
    cdef custatevecHandle_t handle

    with nogil:
        status = custatevecExStateVectorGetResourcesFromDeviceSubSVView(
            <const StateVectorDescriptor>state_vector,
            sub_sv_index,
            &device_id, &d_sub_sv, &stream, &handle)
    check_status(status)
    return (device_id, <intptr_t>d_sub_sv, <intptr_t>stream, <intptr_t>handle)


cpdef tuple state_vector_get_resources_from_unstaged_sub_sv_slice(
        intptr_t state_vector, int32_t sub_sv_index, int32_t slice_index):
    """Get resources from unstaged sub state vector slice.

    Args:
        state_vector (intptr_t): The state vector descriptor.
        sub_sv_index (int32_t): Sub state vector index.
        slice_index (int32_t): Slice index.

    Returns:
        A 4-tuple containing:

        - intptr_t: host memory pointer to the slice data (accessible from the
          GPU).
        - int: memory placement of the slice (a `MemoryPlacement` value).
        - int32_t: device id associated with the slice.
        - intptr_t: CUDA stream associated with the slice.

    .. seealso:: `custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice`
    """
    cdef void* sub_sv_slice
    cdef custatevecExMemoryPlacement_t placement
    cdef int32_t device_id
    cdef cudaStream_t stream

    with nogil:
        status = custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice(
            <StateVectorDescriptor>state_vector,
            sub_sv_index, slice_index,
            &sub_sv_slice, &placement, &device_id, &stream)
    check_status(status)
    return (<intptr_t>sub_sv_slice, <int>placement, device_id, <intptr_t>stream)


cpdef tuple state_vector_get_resources_from_unstaged_sub_sv_slice_view(
        intptr_t state_vector, int32_t sub_sv_index, int32_t slice_index):
    """Get resources from unstaged sub state vector slice (read-only view).

    Args:
        state_vector (intptr_t): The state vector descriptor.
        sub_sv_index (int32_t): Sub state vector index.
        slice_index (int32_t): Slice index.

    Returns:
        A 4-tuple containing:

        - intptr_t: host memory pointer to the slice data (read-only view;
          accessible from the GPU).
        - int: memory placement of the slice (a `MemoryPlacement` value).
        - int32_t: device id associated with the slice.
        - intptr_t: CUDA stream associated with the slice.

    .. seealso:: `custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView`
    """
    cdef const void* sub_sv_slice
    cdef custatevecExMemoryPlacement_t placement
    cdef int32_t device_id
    cdef cudaStream_t stream

    with nogil:
        status = custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView(
            <const StateVectorDescriptor>state_vector,
            sub_sv_index, slice_index,
            &sub_sv_slice, &placement, &device_id, &stream)
    check_status(status)
    return (<intptr_t>sub_sv_slice, <int>placement, device_id, <intptr_t>stream)
del _cyb_IntEnum
