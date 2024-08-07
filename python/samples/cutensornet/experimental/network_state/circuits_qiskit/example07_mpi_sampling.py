# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
from mpi4py import MPI

import qiskit

from cuquantum import cutensornet as cutn
from cuquantum.cutensornet.experimental import NetworkState, TNConfig

root = 0
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
if rank == root:
    print("*** Printing is done only from the root process to prevent jumbled messages ***")
    print(f"The number of processes is {size}")

num_devices = cp.cuda.runtime.getDeviceCount()
device_id = rank % num_devices
dev = cp.cuda.Device(device_id)
dev.use()

props = cp.cuda.runtime.getDeviceProperties(dev.id)
if rank == root:
    print("cuTensorNet-vers:", cutn.get_version())
    print("===== root process device info ======")
    print("GPU-name:", props["name"].decode())
    print("GPU-clock:", props["clockRate"])
    print("GPU-memoryClock:", props["memoryClockRate"])
    print("GPU-nSM:", props["multiProcessorCount"])
    print("GPU-major:", props["major"])
    print("GPU-minor:", props["minor"])
    print("========================")

handle = cutn.create()
cutn_comm = comm.Dup()
cutn.distributed_reset_configuration(handle, MPI._addressof(cutn_comm), MPI._sizeof(cutn_comm))
if rank == root:
    print("Reset distributed MPI configuration")

free_mem = dev.mem_info[0]
free_mem = comm.allreduce(free_mem, MPI.MIN)
workspace_limit = int(free_mem * 0.5) 

# device id must be explicitly set on each process
options = {'handle': handle, 
           'device_id': device_id, 
           'memory_limit': workspace_limit}

# create a QFT circuit
n_qubits = 12
qubits = list(range(n_qubits))
circuit = qiskit.QuantumCircuit(n_qubits)
qft = qiskit.circuit.library.QFT(num_qubits=n_qubits)
circuit.append(qft, qubits)
print(circuit)

# select tensor network contraction as the simulation method
config = TNConfig(num_hyper_samples=4)

# create a NetworkState object
with NetworkState.from_circuit(circuit, dtype='complex128', backend='cupy', config=config, options=options) as state:
    # draw samples from the state object
    nshots = 1000
    samples = state.compute_sampling(nshots)
    if rank == root:
        print("Sampling results:")
        print(samples)

cutn.destroy(handle)