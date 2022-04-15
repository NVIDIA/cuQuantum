# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from enum import IntEnum


# The (subset of) compute types below are shared by cuStateVec and cuTensorNet
class ComputeType(IntEnum):
    """An enumeration of CUDA compute types."""
    COMPUTE_DEFAULT = 0
    COMPUTE_16F     = 1 << 0
    COMPUTE_32F     = 1 << 2
    COMPUTE_64F     = 1 << 4
    COMPUTE_8U      = 1 << 6
    COMPUTE_8I      = 1 << 8
    COMPUTE_32U     = 1 << 7
    COMPUTE_32I     = 1 << 9
    COMPUTE_16BF    = 1 << 10
    COMPUTE_TF32    = 1 << 12


# TODO: use those exposed by CUDA Python instead, but before removing these
# duplicates, check if they are fixed to inherit IntEnum instead of Enum.
class cudaDataType(IntEnum):
    """An enumeration of `cudaDataType_t`."""
    CUDA_R_16F  =  2
    CUDA_C_16F  =  6
    CUDA_R_16BF = 14
    CUDA_C_16BF = 15
    CUDA_R_32F  =  0
    CUDA_C_32F  =  4
    CUDA_R_64F  =  1
    CUDA_C_64F  =  5
    CUDA_R_4I   = 16
    CUDA_C_4I   = 17
    CUDA_R_4U   = 18
    CUDA_C_4U   = 19
    CUDA_R_8I   =  3
    CUDA_C_8I   =  7
    CUDA_R_8U   =  8
    CUDA_C_8U   =  9
    CUDA_R_16I  = 20
    CUDA_C_16I  = 21
    CUDA_R_16U  = 22
    CUDA_C_16U  = 23
    CUDA_R_32I  = 10
    CUDA_C_32I  = 11
    CUDA_R_32U  = 12
    CUDA_C_32U  = 13
    CUDA_R_64I  = 24
    CUDA_C_64I  = 25
    CUDA_R_64U  = 26
    CUDA_C_64U  = 27

class libraryPropertyType(IntEnum):
    """An enumeration of library version information."""
    MAJOR_VERSION = 0
    MINOR_VERSION = 1
    PATCH_LEVEL = 2

del IntEnum
