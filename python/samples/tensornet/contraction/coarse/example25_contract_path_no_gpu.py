# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Path finding without a GPU device
"""

import os
import numpy as np

from cuquantum.tensornet import contract_path, NetworkOptions, OptimizerOptions

# simulate no GPU available by hiding visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def search(inputs, output, size_dict, options=None, optimize=None):
    """
    Path searching.

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
    for modes in inputs:
        shape = [size_dict[m] for m in modes]
        operands.append(np.random.rand(*shape))
    path, info = contract_path(expr, *operands, options=options, optimize=optimize)
    return path

inputs = ['dgf', 'bfe', 'cdbe', 'g', 'ca']
output = 'a'
size_dict = {'a': 4, 'b': 8, 'c': 3, 'd': 2, 'e': 7, 'f': 4, 'g': 5}

# need to specify the upper memory limit allowed for the path scratch memory
net_opt = NetworkOptions(memory_limit=1800000000)
# set which gpu architecture to optimize the path for, eg Hopper = 9
opt_opt = OptimizerOptions(gpu_arch=9)

path = search(inputs, output, size_dict, options=net_opt, optimize=opt_opt)
print(path)
