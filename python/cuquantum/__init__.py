# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cuquantum import custatevec
from cuquantum import cutensornet
from cuquantum.cutensornet import (
    contract, contract_path, einsum, einsum_path, tensor, tensor_qualifiers_dtype, BaseCUDAMemoryManager, CircuitToEinsum, MemoryPointer, 
    Network, NetworkOptions, OptimizerInfo, OptimizerOptions, PathFinderOptions, 
    ReconfigOptions, SlicerOptions)
from cuquantum._utils import ComputeType, cudaDataType, libraryPropertyType
from cuquantum._version import __version__


# We patch all enum values so that they have the correct docstrings
for enum in (
        custatevec.Pauli,
        custatevec.MatrixLayout,
        custatevec.MatrixType,
        custatevec.MatrixMapType,
        custatevec.Collapse,
        custatevec.SamplerOutput,
        custatevec.DeviceNetworkType,
        cutensornet.NetworkAttribute,
        custatevec.CommunicatorType,
        custatevec.DataTransferType,
        custatevec.StateVectorType,
        cutensornet.ContractionOptimizerInfoAttribute,
        cutensornet.ContractionOptimizerConfigAttribute,
        cutensornet.ContractionAutotunePreferenceAttribute,
        cutensornet.WorksizePref,
        cutensornet.Memspace,
        cutensornet.GraphAlgo,
        cutensornet.MemoryModel,
        cutensornet.OptimizerCost,
        cutensornet.TensorSVDConfigAttribute,
        cutensornet.TensorSVDNormalization,
        cutensornet.TensorSVDPartition,
        cutensornet.TensorSVDInfoAttribute,
        cutensornet.GateSplitAlgo,
        cutensornet.StatePurity,
        cutensornet.MarginalAttribute,
        cutensornet.SamplerAttribute,
        ):
    cutensornet._internal.enum_utils.add_enum_class_doc(enum, chomp="_ATTRIBUTE|_PREFERENCE_ATTRIBUTE")

del enum
