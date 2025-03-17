# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cuquantum import bindings
from cuquantum import custatevec
from cuquantum import cutensornet
from cuquantum import densitymat
from cuquantum import tensornet
from cuquantum.cutensornet import (
    contract, contract_path, einsum, einsum_path, tensor, tensor_qualifiers_dtype, BaseCUDAMemoryManager, CircuitToEinsum, MemoryPointer, 
    Network, NetworkOptions, OptimizerInfo, OptimizerOptions, PathFinderOptions, 
    ReconfigOptions, SlicerOptions, MemoryLimitExceeded)
from cuquantum.bindings._utils import ComputeType, cudaDataType, libraryPropertyType
from cuquantum._internal import enum_utils
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
        cutensornet.Status,
        cutensornet.ContractionOptimizerInfoAttribute,
        cutensornet.ContractionOptimizerConfigAttribute,
        cutensornet.ContractionAutotunePreferenceAttribute,
        cutensornet.NetworkAttribute,
        cutensornet.WorksizePref,
        cutensornet.WorkspaceKind,
        cutensornet.Memspace,
        cutensornet.SmartOption,
        cutensornet.GraphAlgo,
        cutensornet.MemoryModel,
        cutensornet.OptimizerCost,
        cutensornet.TensorSVDConfigAttribute,
        cutensornet.TensorSVDAlgo,
        cutensornet.TensorSVDNormalization,
        cutensornet.TensorSVDPartition,
        cutensornet.TensorSVDInfoAttribute,
        cutensornet.GateSplitAlgo,
        cutensornet.BoundaryCondition,
        cutensornet.StatePurity,
        cutensornet.StateAttribute,
        cutensornet.MarginalAttribute,
        cutensornet.SamplerAttribute,
        cutensornet.AccessorAttribute,
        cutensornet.ExpectationAttribute,
        cutensornet.StateMPOApplication,
        cutensornet.StateMPSGaugeOption,
        ):
    enum_utils.add_enum_class_doc(enum, chomp="_ATTRIBUTE|_PREFERENCE_ATTRIBUTE")

del enum
del enum_utils
