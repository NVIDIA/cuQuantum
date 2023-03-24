# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import sys

import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv
from cuquantum import cudaDataType


nGlobalBits = 2
nLocalBits  = 2
nSubSvs     = (1 << nGlobalBits)
subSvSize   = (1 << nLocalBits)

nMaxShots  = 5
nShots     = 5

bitStringLen  = 4
bitOrdering = [0, 1, 2, 3]

bitStrings = np.empty(nShots, dtype=np.int64)
bitStrings_result = np.asarray([0b0011, 0b0011, 0b0111, 0b1011, 0b1110], dtype=np.int64)

# In real appliction, random numbers in range [0, 1) will be used.
randnums = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8], dtype=np.float64)

h_sv = np.asarray([[ 0.000+0.000j,  0.000+0.125j,  0.000+0.250j,  0.000+0.375j],
                   [ 0.000+0.000j,  0.000-0.125j,  0.000-0.250j,  0.000-0.375j],
                   [ 0.125+0.000j,  0.125-0.125j,  0.125-0.250j,  0.125-0.375j],
                   [-0.125+0.000j, -0.125-0.125j, -0.125-0.250j, -0.125-0.375j]],
                  dtype=np.complex128)

# device allocation
if len(sys.argv) == 1:
    numDevices = cp.cuda.runtime.getDeviceCount()
    devices = [i % numDevices for i in range(nSubSvs)]
else:
    numDevices = min(len(sys.argv) - 1, nSubSvs)
    devices = [int(sys.argv[i+1]) for i in range(numDevices)]
    for i in range(numDevices, nSubSvs):
        devices.append(devices[i % numDevices])

print("The following devices will be used in this sample:")
for iSv in range(nSubSvs):
    print(f"  sub-SV {iSv} : device id {devices[iSv]}")

d_sv = []
for iSv in range(nSubSvs):
    with cp.cuda.Device(devices[iSv]):
        d_sv.append(cp.asarray(h_sv[iSv]))

# custatevec handle initialization
handle = []
for iSv in range(nSubSvs):
    with cp.cuda.Device(devices[iSv]):
        handle.append(cusv.create())

# create sampler and check the size of external workspace
sampler = []
extraWorkspaceSizeInBytes = []
for iSv in range(nSubSvs):
    with cp.cuda.Device(devices[iSv]) as dev:
        s, size = cusv.sampler_create(
            handle[iSv], d_sv[iSv].data.ptr, cudaDataType.CUDA_C_64F, nLocalBits,
            nMaxShots)
        sampler.append(s)
        extraWorkspaceSizeInBytes.append(size)

# allocate external workspace if necessary
extraWorkspace = []
for iSv in range(nSubSvs):
    if extraWorkspaceSizeInBytes[iSv] > 0:
        with cp.cuda.Device(devices[iSv]) as dev:
            extraWorkspace.append(cp.cuda.alloc(extraWorkspaceSizeInBytes[iSv]))

# sample preprocess
for iSv in range(nSubSvs):
    with cp.cuda.Device(devices[iSv]) as dev:
        cusv.sampler_preprocess(
            handle[iSv], sampler[iSv], extraWorkspace[iSv].ptr,
            extraWorkspaceSizeInBytes[iSv])

# get norm of the sub state vectors
subNorms = []
for iSv in range(nSubSvs):
    with cp.cuda.Device(devices[iSv]) as dev:
        subNorms.append(cusv.sampler_get_squared_norm(handle[iSv], sampler[iSv]))
        dev.synchronize()

# get cumulative array & norm
cumulativeArray = np.zeros(nSubSvs + 1, dtype=np.float64)
cumulativeArray[1:] = np.cumsum(subNorms)
norm = cumulativeArray[nSubSvs]

# apply offset and norm
for iSv in range(nSubSvs):
    with cp.cuda.Device(devices[iSv]) as dev:
        cusv.sampler_apply_sub_sv_offset(
            handle[iSv], sampler[iSv], iSv, nSubSvs, cumulativeArray[iSv], norm)

# divide randnum array
shotOffsets = np.zeros(nSubSvs+1, dtype=np.int32)
pos = np.searchsorted(randnums, cumulativeArray[1:]/norm)
pos[nSubSvs-1] = nShots
shotOffsets[1:] = pos

# sample bit strings
for iSv in range(nSubSvs):
    shotOffset = int(shotOffsets[iSv])
    nSubShots = shotOffsets[iSv + 1] - shotOffsets[iSv]
    if nSubShots > 0:
        with cp.cuda.Device(devices[iSv]) as dev:
            cusv.sampler_sample(
                handle[iSv], sampler[iSv],
                # when sliced into a 0D array, NumPy returns a scalar, so we can't do
                # bitStrings[shotOffset].ctypes.data and need this workaround
                bitStrings.ctypes.data + shotOffset * bitStrings.dtype.itemsize,
                bitOrdering, bitStringLen,
                randnums.ctypes.data + shotOffset * randnums.dtype.itemsize,
                nSubShots, cusv.SamplerOutput.RANDNUM_ORDER)

# destroy sampler descriptor and custatevec handle
for iSv in range(nSubSvs):
    cp.cuda.Device(devices[iSv]).synchronize()
    cusv.sampler_destroy(sampler[iSv])
    cusv.destroy(handle[iSv])

correct = np.allclose(bitStrings, bitStrings_result)
if correct:
    print("mgpu_sampler example PASSED")
else:
    raise RuntimeError("mgpu_sampler example FAILED: wrong result")
