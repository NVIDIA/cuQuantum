# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause


# In this example, all the available devices (up to 4 devices) will be used by default:
# $ python mgpu_swap_index_bits.py
#
# When device ids are given as additional inputs, the specified devices will be used:
# $ python mgpu_swap_index_bits.py 0 1


import sys

import cupy as cp
import numpy as np

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType, ComputeType


nGlobalIndexBits = 2
nLocalIndexBits  = 1

nSubSvs   = (1 << nGlobalIndexBits)
subSvSize = (1 << nLocalIndexBits)

nMaxDevices = nSubSvs

# specify the type of device network topology to optimize the data transfer sequence.
# SWITCH provides better performance for devices connected via NVLink with an NVSwitch
# or PCIe device network with a single PCIe switch. FULLMESH provides better performance
# for devices connected by full mesh connection.
deviceNetworkType = cusv.DeviceNetworkType.SWITCH

# swap 0th and 2nd qubits
nBitSwaps = 1
bitSwaps = [(0, 2)]

# swap the state vector elements only if 1st qubit is 1
maskLen = 1
maskBitString = [1]
maskOrdering = [1]

# input: 0.2|001> + 0.4|011> - 0.4|101> - 0.8|111>
sv = np.asarray([0.0+0.0j,  0.2+0.0j, 0.0+0.0j,  0.4+0.0j, 
                 0.0+0.0j, -0.4+0.0j, 0.0+0.0j, -0.8+0.0j],
                dtype=np.complex128).reshape(nSubSvs, subSvSize)

# expected: 0.2|001> + 0.4|110> - 0.4|101> - 0.8|111>
sv_result = np.asarray([0.0+0.0j,  0.2+0.0j, 0.0+0.0j,  0.0+0.0j, 
                        0.0+0.0j, -0.4+0.0j, 0.4+0.0j, -0.8+0.0j],
                       dtype=np.complex128).reshape(nSubSvs, subSvSize)

# device allocation
if len(sys.argv) == 1:
    nDevices = min(cp.cuda.runtime.getDeviceCount(), nMaxDevices)
    devices = [i for i in range(nDevices)]
else:
    nDevices = min(len(sys.argv) - 1, nMaxDevices)
    devices = [int(sys.argv[i+1]) for i in range(nDevices)]

# check if device ids do not duplicate
duplicatedDevices = [id for id in set(devices) if devices.count(id) > 1]
if len(duplicatedDevices) != 0:
    raise ValueError(f"device id {duplicatedDevices[0]} is defined more than once")

# enable P2P access
for i in range(nDevices):
    with cp.cuda.Device(devices[i]):
        for j in range(nDevices):
            if i == j: continue
            if cp.cuda.runtime.deviceCanAccessPeer(devices[i], devices[j]) != 1:
                raise RuntimeError(f"P2P access between device id {devices[i]} and {devices[j]} is unsupported")
            cp.cuda.runtime.deviceEnablePeerAccess(devices[j])

# define which device stores each sub state vector
subSvLayout = [devices[iSv % nDevices] for iSv in range(nSubSvs)]

print("The following devices will be used in this sample:")
d_sv = []
for iSv in range(nSubSvs):
    print(f"  sub-SV #{iSv} : device id {subSvLayout[iSv]}")
    with cp.cuda.Device(subSvLayout[iSv]):
        d_sv.append(cp.asarray(sv[iSv]))
d_sv_ptrs = [arr.data.ptr for arr in d_sv]

# custatevec handle initialization
handles = []
for i in range(nDevices):
    with cp.cuda.Device(devices[i]):
        handles.append(cusv.create())

# bit swap
# Note: when this API is called, the current device must be one of the participating devices,
# see the documentation of custatevecMultiDeviceSwapIndexBits()
with cp.cuda.Device(devices[0]):
    cusv.multi_device_swap_index_bits(
        handles, nDevices, d_sv_ptrs, cudaDataType.CUDA_C_64F,
        nGlobalIndexBits, nLocalIndexBits,
        bitSwaps, nBitSwaps, maskBitString, maskOrdering, maskLen,
        deviceNetworkType)

# destroy handles
for i in range(nDevices):
    with cp.cuda.Device(devices[i]):
        cusv.destroy(handles[i])

# check results
correct = True
for iSv in range(nSubSvs):
    with cp.cuda.Device(subSvLayout[iSv]):
        correct = correct and cp.allclose(sv_result[iSv], d_sv[iSv])
if correct:
    print("mgpu_swap_index_bits example PASSED")
else:
    raise RuntimeError("mgpu_swap_index_bits example FAILED: wrong result")
