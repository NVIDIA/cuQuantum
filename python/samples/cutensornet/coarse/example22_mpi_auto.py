# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating automatically parallelizing slice-based tensor network contraction with cuQuantum using MPI.
Here we use:

 - the buffer interface APIs offered by mpi4py v3.1.0+ for communicating ndarray-like objects 
 - CUDA-aware MPI (note: as of cuTensorNet v2.0.0 using non-CUDA-aware MPI is not supported
   and would cause segfault).
 - cuQuantum 22.11+ (cuTensorNet v2.0.0+) for the new distributed contraction feature

$ mpiexec -n 4 python example22_mpi_auto.py
"""
import os

import cupy as cp
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI  # this line initializes MPI

import cuquantum
from cuquantum import cutensornet as cutn


root = 0
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# Check if the env var is set
if not "CUTENSORNET_COMM_LIB" in os.environ:
    raise RuntimeError("need to set CUTENSORNET_COMM_LIB to the path of the MPI wrapper library")

if not os.path.isfile(os.environ["CUTENSORNET_COMM_LIB"]):
    raise RuntimeError("CUTENSORNET_COMM_LIB does not point to the path of the MPI wrapper library")

# Assign the device for each process.
device_id = rank % getDeviceCount()
cp.cuda.Device(device_id).use()

expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]

# Set the operand data on root. Since we use the buffer interface APIs offered by mpi4py for communicating array
#  objects, we can directly use device arrays (cupy.ndarray, for example) as we assume mpi4py is built against
#  a CUDA-aware MPI.
if rank == root:
    operands = [cp.random.rand(*shape) for shape in shapes]
else:
    operands = [cp.empty(shape) for shape in shapes]

# Broadcast the operand data. Throughout this sample we take advantage of the upper-case mpi4py APIs
# that support communicating CPU & GPU buffers (without staging) to reduce serialization overhead for
# array-like objects. This capability requires mpi4py v3.10+.
for operand in operands:
   comm.Bcast(operand, root)

# Bind the communicator to the library handle
handle = cutn.create()
cutn.distributed_reset_configuration(
    handle, *cutn.get_mpi_comm_pointer(comm)
)

# Compute the contraction (with distributed path finding & contraction execution)
result = cuquantum.contract(expr, *operands, options={'device_id' : device_id, 'handle': handle})

# Check correctness.
if rank == root:
   result_cp = cp.einsum(expr, *operands, optimize=True)
   print("Does the cuQuantum parallel contraction result match the cupy.einsum result?", cp.allclose(result, result_cp))
