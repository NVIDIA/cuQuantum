# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Path finding without device memory allocation
"""

import cupy as cp

from cuquantum import contract_path

def search(inputs, output, size_dict, options=None, optimize=None):
    """
    Path searching without device memory allocation.

    Args:
        inputs: A sequence of string specifying the modes for all inputs.
        output: A string specifying the output modes.
        size_dict: A dictionary mapping all modes to corresponding extent.
        options : Options for the tensor network as a :class:`~cuquantum.NetworkOptions` object. 
            Alternatively, a `dict` containing the parameters for the ``NetworkOptions`` constructor can also be provided. 
        optimize : Options for path optimization as an :class:`OptimizerOptions` object. 
            Alternatively, a dictionary containing the parameters for the ``OptimizerOptions`` constructor can also be provided.
    
    Returns:
        path: A list of pairs of operand ordinals representing the best contraction order in the :func:`numpy.einsum_path` format.
    """
    expr = ','.join(inputs) + f'->{output}'
    # create dummy cp.ndarray, note this is only meant for path finding, and should not be used 
    # for operations that requires physical operands, e.g, autotuning and contraction.
    operands = []
    m = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(0, 0, [], 0), 0)
    for modes in inputs:
        shape = [size_dict[m] for m in modes]
        operands.append(cp.ndarray(shape, memptr=m))
    path, info = contract_path(expr, *operands, options=options, optimize=optimize)
    return path

inputs = ['dgf', 'bfe', 'cdbe', 'g', 'ca']
output = 'a'
size_dict = {'a': 4, 'b': 8, 'c': 3, 'd': 2, 'e': 7, 'f': 4, 'g': 5}

path = search(inputs, output, size_dict)
print(path)
