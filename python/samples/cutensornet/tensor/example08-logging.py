# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
QR Example using NumPy ndarray. Specify the logging options

The decomposition results are also NumPy ndarrays.
"""
import logging

import numpy as np

from cuquantum import tensor


a = np.ones((3,2,4,5))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
q, r = tensor.decompose("ijab->ixa,xbj", a)
print(q)
print(r)

