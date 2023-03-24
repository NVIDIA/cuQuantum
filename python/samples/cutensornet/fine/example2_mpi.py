# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating slice-based parallel tensor network contraction with cuQuantum using MPI.

$ mpiexec -n 4 python example2_mpi.py
"""
# Sphinx
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI
import numpy as np

from cuquantum import Network

root = 0
comm = MPI.COMM_WORLD

rank, size = comm.Get_rank(), comm.Get_size()

expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]

# Set the operand data on root.
operands = [np.random.rand(*shape) for shape in shapes] if rank == root else None

# Broadcast the operand data.
operands = comm.bcast(operands, root)
    
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
result = comm.reduce(sendobj=result, op=MPI.SUM, root=root)

# Check correctness.
if rank == root:
   result_np = np.einsum(expr, *operands, optimize=True)
   print("Does the cuQuantum parallel contraction result match the numpy.einsum result?", np.allclose(result, result_np))
