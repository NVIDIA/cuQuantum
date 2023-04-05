# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
truncated SVD Example using NumPy ndarray. Return truncation information.

The decomposition results are also NumPy ndarrays.
"""
import numpy as np

from cuquantum import tensor
import cuquantum.cutensornet as cutn

a = np.ones((3,2,4,5))


handle = cutn.create()
options = {'device_id': 0,
          'handle': handle}
q, r = tensor.decompose("ijab->ixa,xbj", a, options=options)

cutn.destroy(handle)

print(q)
print(r)

