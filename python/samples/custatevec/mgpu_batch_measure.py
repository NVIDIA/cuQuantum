# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import sys

import cupy as cp
import numpy as np

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType, ComputeType


nGlobalBits = 2
nLocalBits = 2
nSubSvs = (1 << nGlobalBits)
subSvSize = (1 << nLocalBits)
bitStringLen = 2
bitOrdering = (1, 0)

bitString = np.empty(bitStringLen, dtype=np.int32)
bitString_result = np.asarray((0, 0), dtype=np.int32)

# In real appliction, random number in range [0, 1) will be used.
randnum = 0.72; 

h_sv = np.asarray([[ 0.000+0.000j,  0.000+0.125j,  0.000+0.250j,  0.000+0.375j],
                   [ 0.000+0.000j,  0.000-0.125j,  0.000-0.250j,  0.000-0.375j],
                   [ 0.125+0.000j,  0.125-0.125j,  0.125-0.250j,  0.125-0.375j],
                   [-0.125+0.000j, -0.125-0.125j, -0.125-0.250j, -0.125-0.375j]],
                  dtype=np.complex128)
h_sv_result = np.asarray([[ 0.0     +0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
                          [ 0.0     +0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
                          [ 0.707107+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
                          [-0.707107+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j]],
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

# get abs2sum for each sub state vector
abs2SumArray = np.empty((nSubSvs,), dtype=np.float64)
for iSv in range(nSubSvs):
    with cp.cuda.Device(devices[iSv]) as dev:
        cusv.abs2sum_array(
            handle[iSv], d_sv[iSv].data.ptr, cudaDataType.CUDA_C_64F, nLocalBits,
            # when sliced into a 0D array, NumPy returns a scalar, so we can't do
            # abs2SumArray[iSv].ctypes.data and need this workaround
            abs2SumArray.ctypes.data + iSv * abs2SumArray.dtype.itemsize,
            0, 0, 0, 0, 0)
        dev.synchronize()

# get cumulative array
cumulativeArray = np.zeros((nSubSvs + 1,), dtype=np.float64)
cumulativeArray[1:] = np.cumsum(abs2SumArray)

# measurement
for iSv in range(nSubSvs):
    if (cumulativeArray[iSv] <= randnum and randnum < cumulativeArray[iSv + 1]):
        norm = cumulativeArray[nSubSvs]
        offset = cumulativeArray[iSv]
        with cp.cuda.Device(devices[iSv]) as dev:
            cusv.batch_measure_with_offset(
                handle[iSv], d_sv[iSv].data.ptr, cudaDataType.CUDA_C_64F, nLocalBits,
                bitString.ctypes.data, bitOrdering, bitStringLen, randnum,
                cusv.Collapse.NONE, offset, norm)
            dev.synchronize()

# get abs2Sum after collapse
for iSv in range(nSubSvs):
    with cp.cuda.Device(devices[iSv]) as dev:
        cusv.abs2sum_array(
            handle[iSv], d_sv[iSv].data.ptr, cudaDataType.CUDA_C_64F, nLocalBits,
            abs2SumArray.ctypes.data + iSv * abs2SumArray.dtype.itemsize, 0, 0,
            bitString.ctypes.data, bitOrdering, bitStringLen)
        dev.synchronize()

# get norm after collapse
norm = np.sum(abs2SumArray, dtype=np.float64)

# collapse sub state vectors
for iSv in range(nSubSvs):
    with cp.cuda.Device(devices[iSv]) as dev:
        cusv.collapse_by_bitstring(
            handle[iSv], d_sv[iSv].data.ptr, cudaDataType.CUDA_C_64F, nLocalBits,
            bitString.ctypes.data, bitOrdering, bitStringLen, norm)
        dev.synchronize()

        # destroy handle when done
        cusv.destroy(handle[iSv])

        h_sv[iSv] = cp.asnumpy(d_sv[iSv])

correct = np.allclose(h_sv, h_sv_result)
correct &= np.allclose(bitString, bitString_result) 
  
if correct:
    print("mgpu_batch_measure example PASSED");
else:
    raise RuntimeError("mgpu_batch_measure example FAILED: wrong result")
