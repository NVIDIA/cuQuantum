# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import cupyx as cpx
import numpy as np

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType

dtype = np.complex128
sv_data_type = cudaDataType.CUDA_C_64F;

n_local_index_bits = 3;

sub_sv_size = 2 ** n_local_index_bits

# allocate host sub state vectors
n_sub_svs = 2
sub_svs = [None] * n_sub_svs
sub_svs[0] = cpx.empty_pinned(sub_sv_size, dtype=dtype)
sub_svs[0][:] = 0.25 + 0.j
sub_svs[1] = cpx.zeros_pinned(sub_sv_size, dtype=dtype)

# allocate device slots

n_device_slots = 1;
device_slots_size = sub_sv_size * n_device_slots
device_slots = cp.zeros([device_slots_size], dtype=dtype)

# initialize custatevec handle
handle = cusv.create()

# create migrator
migrator = cusv.sub_sv_migrator_create(handle, device_slots.data.ptr, sv_data_type,
                                       n_device_slots, n_local_index_bits)

device_slot_index = 0
src_sub_sv = sub_svs[0]
dst_sub_sv = sub_svs[1]

# migrate sub_svs[0] into device_slots
cusv.sub_sv_migrator_migrate(handle, migrator, device_slot_index,
                             src_sub_sv.ctypes.data, 0, 0, sub_sv_size)

# migrate device_slots into sub_svs[1]
cusv.sub_sv_migrator_migrate(handle, migrator, device_slot_index,
                             0, dst_sub_sv.ctypes.data, 0, sub_sv_size)

# destroy migrator
cusv.sub_sv_migrator_destroy(handle, migrator)

# destroy custatevec handle
cusv.destroy(handle)

# check if sub_svs[1] has expected values
correct = np.all(sub_svs[1] == 0.25 + 0.j)

if correct:
    print('subsv_migration example PASSED')
else:
    raise RuntimeError('subsv_migration example FAILED: wrong result')
