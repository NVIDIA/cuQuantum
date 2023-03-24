# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType, ComputeType


if cp.cuda.runtime.runtimeGetVersion() < 11020:
    raise RuntimeError("memory_handler example WAIVED : This example uses CUDA's "
                       "built-in stream-ordered memory allocator, which requires "
                       "CUDA 11.2+.")

nIndexBits   = 3
nSvSize      = (1 << nIndexBits)

sv = cp.asarray([0.48+1j*0.0, 0.36+1j*0.0, 0.64+1j*0.0, 0.48+1j*0.0, 
                 0.0+1j*0.0, 0.0+1j*0.0, 0.0+1j*0.0, 0.0+1j*0.0],
                dtype=cp.complex128)

# gates
adjoint = 0
layout = cusv.MatrixLayout.ROW

# Hadamard gate
hTargets = (2,)
hNTargets = 1
hGate = np.asarray([1/np.sqrt(2)+1j*0.0, 1/np.sqrt(2)+1j*0.0,
                    1/np.sqrt(2)+1j*0.0, -1/np.sqrt(2)+1j*0.0],
                   dtype=np.complex128)

# control-SWAP gate
swapTargets = (0, 1)
swapNTargets = 2
swapControls = (2,)
swapNControls = 1
swapGate = np.asarray([1.0+1j*0.0, 0.0+1j*0.0, 0.0+1j*0.0, 0.0+1j*0.0,
                       0.0+1j*0.0, 0.0+1j*0.0, 1.0+1j*0.0, 0.0+1j*0.0,
                       0.0+1j*0.0, 1.0+1j*0.0, 0.0+1j*0.0, 0.0+1j*0.0,
                       0.0+1j*0.0, 0.0+1j*0.0, 0.0+1j*0.0, 1.0+1j*0.0],
                      dtype=np.complex128)

# observable
basisBits = (2,)
nBasisBits = 1
observable = np.asarray([1.0+1j*0.0, 0.0+1j*0.0,
                         0.0+1j*0.0, 0.0+1j*0.0], dtype=np.complex128)

# check device config
dev = cp.cuda.Device()
if not dev.attributes['MemoryPoolsSupported']:
    raise RuntimeError("memory handler example WAIVED: device does not support CUDA Memory pools")

# avoid shrinking the pool
mempool = cp.cuda.runtime.deviceGetDefaultMemPool(dev.id)
if int(cp.__version__.split('.')[0]) >= 10:
    # this API is exposed since CuPy v10
    cp.cuda.runtime.memPoolSetAttribute(
        mempool, cp.cuda.runtime.cudaMemPoolAttrReleaseThreshold, 0xffffffffffffffff)  # = UINT64_MAX

# custatevec handle initialization
handle = cusv.create()
stream = cp.cuda.Stream()
cusv.set_stream(handle, stream.ptr)

# device memory handler
# In Python we support 3 kinds of calling conventions as of v22.03, this example
# involves using Python callables. Please refer to the documentation of
# set_device_mem_handler() for further detail.
def malloc(size, stream):
    return cp.cuda.runtime.mallocAsync(size, stream)

def free(ptr, size, stream):
    cp.cuda.runtime.freeAsync(ptr, stream)

handler = (malloc, free, "memory_handler python example")
cusv.set_device_mem_handler(handle, handler)

# apply Hadamard gate
cusv.apply_matrix(
    handle, sv.data.ptr, cudaDataType.CUDA_C_64F, nIndexBits,
    hGate.ctypes.data, cudaDataType.CUDA_C_64F, layout, adjoint,
    hTargets, hNTargets, 0, 0, 0, ComputeType.COMPUTE_DEFAULT,
    0, 0)  # last two 0s indicate we're using our own mempool

# apply Hadamard gate
cusv.apply_matrix(
    handle, sv.data.ptr, cudaDataType.CUDA_C_64F, nIndexBits,
    swapGate.ctypes.data, cudaDataType.CUDA_C_64F, layout, adjoint,
    swapTargets, swapNTargets, swapControls, 0, swapNControls, ComputeType.COMPUTE_DEFAULT,
    0, 0)  # last two 0s indicate we're using our own mempool

# apply Hadamard gate
cusv.apply_matrix(
    handle, sv.data.ptr, cudaDataType.CUDA_C_64F, nIndexBits,
    hGate.ctypes.data, cudaDataType.CUDA_C_64F, layout, adjoint,
    hTargets, hNTargets, 0, 0, 0, ComputeType.COMPUTE_DEFAULT,
    0, 0)  # last two 0s indicate we're using our own mempool

# compute expectation
expect = np.empty((1,), dtype=np.float64)
cusv.compute_expectation(
    handle, sv.data.ptr, cudaDataType.CUDA_C_64F, nIndexBits, 
    expect.ctypes.data, cudaDataType.CUDA_R_64F, 
    observable.ctypes.data, cudaDataType.CUDA_C_64F, layout,
    basisBits, nBasisBits, ComputeType.COMPUTE_DEFAULT,
    0, 0)  # last two 0s indicate we're using our own mempool

stream.synchronize()

# destroy handle
cusv.destroy(handle)

expectationValueResult = 0.9608
if np.isclose(expect, expectationValueResult):
    print("memory_handler example PASSED")
else:
    raise RuntimeError("memory_handler example FAILED: wrong result")
