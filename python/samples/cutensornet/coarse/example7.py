# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example using NumPy ndarrays. Specify logging options.

The contraction result is also a NumPy ndarray.
"""
import logging

import numpy as np

from cuquantum import contract, NetworkOptions


a = np.ones((3,2))
b = np.ones((2,3))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
r = contract("ij,jk", a, b)       
print(r)

