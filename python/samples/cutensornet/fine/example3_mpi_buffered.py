# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating slice-based parallel tensor network contraction with cuQuantum using MPI. Here we use
the buffer interface APIs offered by mpi4py for communicating ndarray-like objects.

$ mpiexec -n 4 python example3_mpi_buffered.py
"""

from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI
import numpy as np

from cuquantum import Network

root = 0
comm = MPI.COMM_WORLD

rank, size = comm.Get_rank(), comm.Get_size()

expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]

# Set the operand data on root. Since we use the buffer interface APIs offered by mpi4py for communicating array
#  objects, we can directly use device arrays (cupy.ndarray, for example) here if mpi4py is built against a
#  CUDA-aware MPI implementation.
if rank == root:
    operands = [np.random.rand(*shape) for shape in shapes]
else:
    operands = [np.empty(shape) for shape in shapes]

# Broadcast the operand data. Here and elsewhere in this sample we take advantage of the single-segment buffer
#  interface APIs provided by mpi4py to reduce serialization overhead for array-like objects.
for operand in operands:
   comm.Bcast(operand, root)

# Assign the device for each process.
device_id = rank % getDeviceCount()

# Create network object.
network = Network(expr, *operands, options={'device_id' : device_id})

# Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
path, info = network.contract_path(optimize={'samples': 8, 'slicing': {'min_slices': max(16, size)}})

# Select the best path from all ranks.
opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)
if rank == root:
    print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

# Broadcast info from the sender to all other ranks.
info = comm.bcast(info, sender)

# Set path and slices.
path, info = network.contract_path(optimize={'path': info.path, 'slicing': info.slices})

# Calculate this process's share of the slices.
num_slices = info.num_slices
chunk, extra = num_slices // size, num_slices % size
slice_begin = rank * chunk + min(rank, extra)
slice_end = num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
slices = range(slice_begin, slice_end)

print(f"Process {rank} is processing slice range: {slices}.")

# Contract the group of slices the process is responsible for.
result = network.contract(slices=slices)

# Sum the partial contribution from each process on root.
if rank == root:
    comm.Reduce(sendbuf=MPI.IN_PLACE, recvbuf=result, op=MPI.SUM, root=root)
else:
    comm.Reduce(sendbuf=result, recvbuf=None, op=MPI.SUM, root=root)

# Check correctness.
if rank == root:
   result_np = np.einsum(expr, *operands, optimize=True)
   print("Does the cuQuantum parallel contraction result match the numpy.einsum result?", np.allclose(result, result_np))
