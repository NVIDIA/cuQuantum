# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cuquantum import custatevec
from cuquantum import cutensornet
from cuquantum.cutensornet import (
    contract, contract_path, einsum, einsum_path, Network, BaseCUDAMemoryManager, MemoryPointer,
    NetworkOptions, OptimizerInfo, OptimizerOptions, PathFinderOptions, ReconfigOptions, SlicerOptions)
from cuquantum.utils import ComputeType, cudaDataType, libraryPropertyType
from cuquantum._version import __version__


# We patch all enum values so that they have the correct docstrings
for enum in (
        custatevec.Pauli,
        custatevec.MatrixLayout,
        custatevec.MatrixType,
        custatevec.Collapse,
        custatevec.SamplerOutput,
        cutensornet.ContractionOptimizerInfoAttribute,
        cutensornet.ContractionOptimizerConfigAttribute,
        cutensornet.ContractionAutotunePreferenceAttribute,
        cutensornet.WorksizePref,
        cutensornet.Memspace,
        cutensornet.GraphAlgo,
        cutensornet.MemoryModel,
        ):
    cutensornet._internal.enum_utils.add_enum_class_doc(enum, chomp="_ATTRIBUTE|_PREFERENCE_ATTRIBUTE")

del enum, utils
