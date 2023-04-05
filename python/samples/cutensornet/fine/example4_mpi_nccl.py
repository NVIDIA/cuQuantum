# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating slice-based parallel tensor network contraction with cuQuantum using NCCL and MPI. Here
we create the input tensors directly on the GPU using CuPy since NCCL only supports GPU buffers.

The low-level Python wrapper for NCCL is provided by CuPy. MPI (through mpi4py) is only needed to bootstrap
the multiple processes, set up the NCCL communicator, and to communicate data on the CPU. NCCL can be used
without MPI for a "single process multiple GPU" model.

For users who do not have NCCL installed already, CuPy provides detailed instructions on how to install
it for both pip and conda users when "import cupy.cuda.nccl" fails.

We recommend that those using CuPy v10+ use CuPy's high-level "cupyx.distributed" module to avoid having to
manipulate GPU pointers in Python.

Note that with recent NCCL, GPUs cannot be oversubscribed (not more than one process per GPU). Users will
see an NCCL error if the number of processes on a node exceeds the number of GPUs on that node.

$ mpiexec -n 4 python example4_mpi_nccl.py
"""

import cupy as cp
from cupy.cuda import nccl
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

from cuquantum import Network

# Set up the MPI environment.
root = 0
comm_mpi = MPI.COMM_WORLD
rank, size = comm_mpi.Get_rank(), comm_mpi.Get_size()

# Assign the device for each process.
device_id = rank % getDeviceCount()

# Define the tensor network topology.
expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]

# Note that all NCCL operations must be performed in the correct device context.
cp.cuda.Device(device_id).use()

# Set up the NCCL communicator.
nccl_id = nccl.get_unique_id() if rank == root else None
nccl_id = comm_mpi.bcast(nccl_id, root)
comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)

# Set the operand data on root.
if rank == root:
    operands = [cp.random.rand(*shape) for shape in shapes]
else:
    operands = [cp.empty(shape) for shape in shapes]

# Broadcast the operand data. We pass in the CuPy ndarray data pointers to the NCCL APIs.
stream_ptr = cp.cuda.get_current_stream().ptr
for operand in operands:
    comm_nccl.broadcast(operand.data.ptr, operand.data.ptr, operand.size, nccl.NCCL_FLOAT64, root, stream_ptr)

# Create network object.
network = Network(expr, *operands)

# Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
path, info = network.contract_path(optimize={'samples': 8, 'slicing': {'min_slices': max(16, size)}})

# Select the best path from all ranks. Note that we still use the MPI communicator here for simplicity.
opt_cost, sender = comm_mpi.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)
if rank == root:
    print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

# Broadcast info from the sender to all other ranks.
info = comm_mpi.bcast(info, sender)

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
stream_ptr = cp.cuda.get_current_stream().ptr
comm_nccl.reduce(result.data.ptr, result.data.ptr, result.size, nccl.NCCL_FLOAT64, nccl.NCCL_SUM, root, stream_ptr)

# Check correctness.
if rank == root:
    result_cp = cp.einsum(expr, *operands, optimize=True)
    print("Does the cuQuantum parallel contraction result match the cupy.einsum result?", cp.allclose(result, result_cp))
