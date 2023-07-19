# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

import numpy as np
import cupy as cp
from cupyx.profiler import benchmark

from cuquantum import custatevec as cusv

from .._utils import (check_sequence, dtype_to_cuda_type, precision_str_to_dtype,
                      wrap_with_nvtx)


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


def test_cusv_sampler(
        n_qubits, dtype_sv, bit_ordering, n_shots, output_order, n_warmup, n_repeat, *,
        benchmark_data=None):
    logger.debug(f"{n_qubits=}")
    logger.debug(f"{dtype_sv=}")
    logger.debug(f"{bit_ordering=}")
    logger.debug(f"{n_shots=}")
    logger.debug(f"{output_order}")
    logger.debug(f"{n_warmup=}")
    logger.debug(f"{n_repeat=}")

    check_sequence(bit_ordering, max_size=n_qubits, name="bit_ordering")
    dtype_sv = precision_str_to_dtype(dtype_sv)
    size_sv = (1 << n_qubits)

    # the statevector must reside on device
    sv = cp.ones((size_sv,), dtype=dtype_sv)
    sv /= np.sqrt(size_sv)
    # assert cp.allclose(cp.sum(cp.abs(sv)**2), 1)
    data_type_sv = dtype_to_cuda_type(dtype_sv)

    # the output bitstrings must reside on host
    bit_strings = np.empty((n_shots,), dtype=np.int64)

    # the random seeds must be a host array
    randnums = np.random.random((n_shots,)).astype(np.float64)

    cp.cuda.Device().synchronize()  # ensure data prep is done before switching stream

    ####################################################################################
    
    # cuStateVec handle initialization
    handle = cusv.create()
    stream = cp.cuda.Stream()
    cusv.set_stream(handle, stream.ptr)

    # create sampler and check the size of external workspace
    sampler, workspace_size = cusv.sampler_create(
        handle, sv.data.ptr, data_type_sv, n_qubits, n_shots)
    
    with stream:
        # manage the workspace
        if workspace_size > 0:
            workspace = cp.cuda.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        # sample preprocess
        sampler_preprocess = wrap_with_nvtx(
            cusv.sampler_preprocess, "sampler_preprocess")
        args = (handle, sampler, workspace_ptr, workspace_size)

        result1 = benchmark(
            sampler_preprocess,
            args,
            n_warmup=n_warmup, n_repeat=n_repeat)
        logger.debug(str(result1))

        # sample bit strings
        sampler_sample = wrap_with_nvtx(
            cusv.sampler_sample, "sampler_sample")
        args = (
            handle, sampler, bit_strings.ctypes.data, bit_ordering, len(bit_ordering),
            randnums.ctypes.data, n_shots,
            cusv.SamplerOutput.RANDNUM_ORDER if output_order == "random" else cusv.SamplerOutput.ASCENDING_ORDER)

        result2 = benchmark(
            sampler_sample,
            args,
            n_warmup=n_warmup, n_repeat=n_repeat)
        logger.debug(str(result2))
    
    # clean up
    cusv.sampler_destroy(sampler)
    cusv.destroy(handle)

    cpu_time = np.average(result1.cpu_times) + np.average(result2.cpu_times)
    gpu_time = np.average(result1.gpu_times[0]) + np.average(result2.gpu_times[0])

    return cpu_time, gpu_time
