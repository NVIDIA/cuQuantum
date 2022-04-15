# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Functions to link type names with CUDA data and compute types.
"""

__all__ = ['NAME_TO_DATA_TYPE', 'NAME_TO_COMPUTE_TYPE']

import re

# hack to break circular import
from cuquantum.utils import ComputeType, cudaDataType


def create_cuda_data_type_map(cuda_data_type_enum_class):
    """
    Map the data type name to the corresponding CUDA data type.
    """
    cuda_data_type_pattern = re.compile("CUDA_(?P<cr>C|R)_(?P<width>\d+)(?P<type>F|I|U|BF)")

    type_code_map = { 'i' : 'int', 'u' : 'uint', 'f' : 'float', 'bf' : 'bfloat' }

    cuda_data_type_map = dict()
    for d in cuda_data_type_enum_class:
        m = cuda_data_type_pattern.match(d.name)

        is_complex = m.group('cr').lower() == 'c'
        type_code = type_code_map[m.group('type').lower()]

        if is_complex and type_code != 'float':
            continue

        width = int(m.group('width'))
        if is_complex:
            width *= 2
            type_code = 'complex'

        name = type_code + str(width)
        cuda_data_type_map[name] = d

    return cuda_data_type_map


def create_cuda_compute_type_map(cuda_compute_type_enum_class):
    """
    Map the data type name to the corresponding CUDA compute type.
    """
    cuda_compute_type_pattern = re.compile("COMPUTE_(?:(?P<width>\d+)(?P<type>F|I|U|BF)|(?P<tf32>TF32))")

    type_code_map = { 'i' : 'int', 'u' : 'uint', 'f' : 'float', 'bf' : 'bfloat' }

    cuda_compute_type_map = dict()
    for c in cuda_compute_type_enum_class:
        if c.name == 'COMPUTE_DEFAULT':
            continue

        m = cuda_compute_type_pattern.match(c.name)

        if not m:
            raise ValueError("Internal error - unexpected enum entry")

        if m.group('tf32'): 
            continue

        name = type_code_map[m.group('type').lower()] + m.group('width')
        cuda_compute_type_map[name] = c

    # Treat complex types as special case.
    cuda_compute_type_map['complex64'] = cuda_compute_type_enum_class.COMPUTE_32F
    cuda_compute_type_map['complex128'] = cuda_compute_type_enum_class.COMPUTE_64F

    return cuda_compute_type_map


NAME_TO_DATA_TYPE = create_cuda_data_type_map(cudaDataType)
NAME_TO_COMPUTE_TYPE = create_cuda_compute_type_map(ComputeType)
