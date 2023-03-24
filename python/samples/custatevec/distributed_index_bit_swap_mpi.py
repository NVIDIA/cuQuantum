# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import time

import cupy as cp
import numpy as np
from mpi4py import MPI

import cuquantum
from cuquantum import cudaDataType
from cuquantum import custatevec as cusv


N_LOCAL_INDEX_BITS = 26


def run_distributed_index_bit_swaps(
        rank, size, n_global_index_bits, n_local_index_bits,
        index_bit_swaps, mask_bit_string, mask_ordering):
    if rank == 0:
        print(f"index bit swaps: {index_bit_swaps}")
        print(f"mask bit string: {mask_bit_string}")
        print(f"mask ordering: {mask_ordering}")

    # data type of the state vector, acceptable values are CUDA_C_32F and CUDA_C_64F.
    sv_data_type = cudaDataType.CUDA_C_64F
    sv_dtype = cp.complex128 if sv_data_type == cudaDataType.CUDA_C_64F else cp.complex64

    # the number of index bits corresponding to sub state vectors accessible via GPUDirect P2P,
    # and it should be adjusted based on the number of GPUs/node, N, participating in the distributed
    # state vector (N=2^n_p2p_device_bits) that supports P2P data transfer
    n_p2p_device_bits = 0
    n_sub_svs_p2p = 1 << n_p2p_device_bits

    # use rank and size to map sub state vectors
    # this sample assigns one device to one rank and allocates one sub state vector on the assigned device
    # use the rank as the index of the sub state vector locally allocated in this process
    org_sub_sv_index = rank

    # the number of sub state vectors is identical to the number of processes
    n_sub_svs = size

    # transfer workspace size
    transfer_workspace_size = 1 << N_LOCAL_INDEX_BITS

    # bind the device to the process
    # this is based on the assumption of the global rank placement that the
    # processes are mapped to nodes in contiguous chunks (see the comment below)
    num_devices = cp.cuda.runtime.getDeviceCount()
    assert num_devices > 0
    if n_p2p_device_bits > 0:
        assert num_devices >= n_sub_svs_p2p
        cp.cuda.runtime.setDevice(rank % n_sub_svs_p2p)
    else:
        cp.cuda.runtime.setDevice(rank % num_devices)

    # allocate local sub state vector, stream and event
    d_org_sub_sv = cp.zeros((1 << n_local_index_bits,), dtype=sv_dtype)
    local_stream = cp.cuda.Stream()
    # event should be created with the cudaEventInterprocess flag
    local_event = cp.cuda.Event(disable_timing=True, interprocess=True)

    # create cuStateVec handle
    handle = cusv.create()

    # create communicator
    #
    # cuStateVec provides built-in communicators for Open MPI and MPICH.
    #
    # Built-in communicators dynamically resolve required MPI functions by using dlopen().
    # This Python sample relies on mpi4py loading libmpi.so and initializing MPI for us.
    # The deviation of the treatment for Open MPI and MPICH stems from the fact that the
    # scope of the loaded MPI symbols (by mpi4py) are different due to a Python limitation
    # (NVIDIA/cuQuantum#31), so we have to work around it.
    #
    # An external communicator can be used for MPI libraries that are not ABI compatible
    # with Open MPI or MPICH. It uses a shared library that wraps the MPI library of choice.
    # The soname should be set to the full path to the shared library. If you need this
    # capability, please refer to the "mpicomm.c" file that comes with the C/C++ sample
    # (which is a counterpart of this sample).

    name, _ = MPI.get_vendor()
    if name == "Open MPI":
        # use built-in OpenMPI communicator
        communicator_type = cusv.CommunicatorType.OPENMPI
        soname = ""
    elif name == "MPICH":
        # use built-in MPICH communicator
        communicator_type = cusv.CommunicatorType.MPICH
        # work around a Python limitation as discussed in NVIDIA/cuQuantum#31
        soname = "libmpi.so"
    else:
        # use external communicator
        communicator_type = cusv.CommunicatorType.EXTERNAL
        # please compile mpicomm.c to generate the shared library and place its path here
        soname = ""
        if not soname:
            raise ValueError("please supply the soname to the shared library providing "
                             "an external communicator for cuStateVec")

    # create communicator
    communicator = cusv.communicator_create(handle, communicator_type, soname)
    comm = MPI.COMM_WORLD

    # create sv segment swap worker
    sv_seg_swap_worker, extra_workspace_size, min_transfer_workspace_size = cusv.sv_swap_worker_create(
        handle, communicator,
        d_org_sub_sv.data.ptr, org_sub_sv_index, local_event.ptr, sv_data_type,
        local_stream.ptr)

    # set extra workspace
    d_extra_workspace = cp.cuda.alloc(extra_workspace_size)
    cusv.sv_swap_worker_set_extra_workspace(
        handle, sv_seg_swap_worker, d_extra_workspace.ptr, extra_workspace_size)

    # set transfer workspace
    # The size should be equal to or larger than min_transfer_workspace_size
    # Depending on the systems, larger transfer workspace can improve the performance
    transfer_workspace_size = max(min_transfer_workspace_size, transfer_workspace_size)
    d_transfer_workspace = cp.cuda.alloc(transfer_workspace_size)
    cusv.sv_swap_worker_set_transfer_workspace(
        handle, sv_seg_swap_worker, d_transfer_workspace.ptr, transfer_workspace_size)

    # set remote sub state vectors accessible via GPUDirect P2P
    # events should be also set for synchronization
    sub_sv_indices_p2p = []
    d_sub_svs_p2p = []
    remote_events = []
    if n_p2p_device_bits > 0:

        # distribute device memory handles
        # under the hood the handle is stored as a Python bytes object
        ipc_mem_handle = cp.cuda.runtime.ipcGetMemHandle(d_org_sub_sv.data.ptr)
        ipc_mem_handles = comm.allgather(ipc_mem_handle)

        # distribute event handles
        ipc_event_handle = cp.cuda.runtime.ipcGetEventHandle(local_event.ptr)
        ipc_event_handles = comm.allgather(ipc_event_handle)

        # get remote device pointers and events
        # this calculation assumes that the global rank placement is done in a round-robin fashion
        # across nodes, so for example if n_p2p_device_bits=2 there are 2^2=4 processes/node (and
        # 1 GPU/progress) and we expect the global MPI ranks to be assigned as
        #   0  1  2  3 -> node 0
        #   4  5  6  7 -> node 1
        #   8  9 10 11 -> node 2
        #             ...
        # if the rank placement scheme is different, you will need to calculate based on local MPI
        # rank/size, as CUDA IPC is only for intra-node, not inter-node, communication.
        p2p_sub_sv_index_begin = (org_sub_sv_index // n_sub_svs_p2p) * n_sub_svs_p2p
        p2p_sub_sv_index_end = p2p_sub_sv_index_begin + n_sub_svs_p2p
        for p2p_sub_sv_index in range(p2p_sub_sv_index_begin, p2p_sub_sv_index_end):
            if org_sub_sv_index == p2p_sub_sv_index:
                continue  # don't need local sub state vector pointer
            sub_sv_indices_p2p.append(p2p_sub_sv_index)

            dst_mem_handle = ipc_mem_handles[p2p_sub_sv_index]
            # default is to use cudaIpcMemLazyEnablePeerAccess
            d_sub_sv_p2p = cp.cuda.runtime.ipcOpenMemHandle(dst_mem_handle)
            d_sub_svs_p2p.append(d_sub_sv_p2p)

            event_p2p = cp.cuda.runtime.ipcOpenEventHandle(ipc_event_handles[p2p_sub_sv_index])
            remote_events.append(event_p2p)

        # set p2p sub state vectors
        assert len(d_sub_svs_p2p) == len(sub_sv_indices_p2p) == len(remote_events)
        cusv.sv_swap_worker_set_sub_svs_p2p(
            handle, sv_seg_swap_worker,
            d_sub_svs_p2p, sub_sv_indices_p2p, remote_events, len(d_sub_svs_p2p))

    # create distributed index bit swap scheduler
    scheduler = cusv.dist_index_bit_swap_scheduler_create(
        handle, n_global_index_bits, n_local_index_bits)

    # set the index bit swaps to the scheduler
    # n_swap_batches is obtained by the call.  This value specifies the number of loops
    assert len(mask_bit_string) == len(mask_ordering)
    n_swap_batches = cusv.dist_index_bit_swap_scheduler_set_index_bit_swaps(
        handle, scheduler,
        index_bit_swaps, len(index_bit_swaps),
        mask_bit_string, mask_ordering, len(mask_bit_string))

    # the main loop of index bit swaps
    n_loops = 2
    for loop in range(n_loops):
        start = time.perf_counter()

        for swap_batch_index in range(n_swap_batches):
            # get parameters
            parameters = cusv.dist_index_bit_swap_scheduler_get_parameters(
                handle, scheduler, swap_batch_index, org_sub_sv_index)

            # the rank of the communication endpoint is parameters.dst_sub_sv_index
            # as "rank == sub_sv_index" is assumed in the present sample.
            rank = parameters.dst_sub_sv_index

            # set parameters to the worker
            cusv.sv_swap_worker_set_parameters(
                handle, sv_seg_swap_worker, parameters, rank)

            # execute swap
            cusv.sv_swap_worker_execute(
                handle, sv_seg_swap_worker, 0, parameters.transfer_size)

            # all internal CUDA calls are serialized on local_stream

        # synchronize all operations on device
        local_stream.synchronize()

        # barrier here for time measurement
        comm.barrier()
        elapsed = time.perf_counter() - start
        if (loop == n_loops - 1) and (org_sub_sv_index == 0):
            # output benchmark result
            elm_size = 16 if sv_data_type == cudaDataType.CUDA_C_64F else 8
            fraction = 1. - 0.5 ** len(index_bit_swaps)
            transferred = 2 ** n_local_index_bits * fraction * elm_size
            bw = transferred / elapsed * 1E-9
            print(f"BW {bw} [GB/s]")

    # free all resources
    cusv.dist_index_bit_swap_scheduler_destroy(handle, scheduler)
    cusv.sv_swap_worker_destroy(handle, sv_seg_swap_worker)
    cusv.communicator_destroy(handle, communicator)
    cusv.destroy(handle)

    # free IPC pointers and events
    for d_sub_sv in d_sub_svs_p2p:
        cp.cuda.runtime.ipcCloseMemHandle(d_sub_sv)
    for event in remote_events:
        cp.cuda.runtime.eventDestroy(event)


if __name__ == "__main__":
    # get rank and size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # size should be a power of two number
    assert (size & (size - 1)) == 0

    # compute n_global_index_bits from the size 
    # n_global_index_bits = log2(size)
    n_global_index_bits = 0
    while (1 << n_global_index_bits) < size:
        n_global_index_bits += 1
    # the size of local sub state vectors
    n_local_index_bits = N_LOCAL_INDEX_BITS

    # create index bit swap
    index_bit_swaps = []
    n_index_bit_swaps = 1
    n_index_bits = n_local_index_bits + n_global_index_bits
    for idx in range(n_index_bit_swaps):
        index_bit_swaps.append((n_local_index_bits-1-idx, n_index_bits-idx-1))
    # empty mask
    mask_bit_string = mask_ordering = []

    run_distributed_index_bit_swaps(
        rank, size, n_global_index_bits, n_local_index_bits,
        index_bit_swaps, mask_bit_string, mask_ordering)
