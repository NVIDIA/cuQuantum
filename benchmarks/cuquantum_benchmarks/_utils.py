# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import ctypes
from dataclasses import dataclass
import functools
import math
import json
import hashlib
import logging
import os
import platform
import random
import re
import time
from typing import Iterable, Optional, Union
import warnings

import cupy as cp
import numpy as np
import nvtx
from cuquantum import cudaDataType, ComputeType
from cuquantum.cutensornet._internal.einsum_parser import create_size_dict
import psutil


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


def wrap_with_nvtx(func, msg):
    """Add NVTX makers to a function with a message."""
    @functools.wraps(func)
    def inner(*args, **kwargs):
        with nvtx.annotate(msg):
            return func(*args, **kwargs)
    return inner


def reseed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    # Q: How about CuPy?


def precision_str_to_dtype(precision, is_complex=True):
    if precision == "single":
        if is_complex:
            return np.complex64
        else:
            return np.float32
    elif precision == "double":
        if is_complex:
            return np.complex128
        else:
            return np.float64
    else:
        raise ValueError


def dtype_to_cuda_type(dtype):
    if dtype == np.complex64:
        return cudaDataType.CUDA_C_32F
    elif dtype == np.complex128:
        return cudaDataType.CUDA_C_64F
    else:
        raise ValueError


def dtype_to_compute_type(dtype):
    if dtype == np.complex64:
        return ComputeType.COMPUTE_32F
    elif dtype == np.complex128:
        return ComputeType.COMPUTE_64F
    else:
        raise ValueError


def generate_size_dict_from_operands(einsum, operands):
    inputs = einsum.split("->")[0]
    inputs = inputs.split(",")
    assert len(inputs) == len(operands)
    return create_size_dict(inputs, operands)


# TODO: clean up, this is copied from internal utils
def convert_einsum_to_txt(einsum, size_dict, filename):

    def _gen_txt(content, idx_map, idx_counter, tn, size_dict, dump_extents=True):
        # TODO: refactor this with the contraction_*.py utilities
        for i in tn:
            # dump indices
            if i == '': continue
            idx = idx_map.get(i, idx_counter)
            assert idx is not None, f"got {idx} for {i} from {o}"
            content += f"{idx} "
            if idx == idx_counter:
                idx_map[i] = idx_counter
                idx_counter += 1

        if dump_extents:
            content += ' | '
            for i in tn:
                content += f"{size_dict[i]} "
            content += '\n'
        return content, idx_map, idx_counter

    # TODO: refactor this with the contraction_*.py utilities
    content = ''
    idx_map = {}
    idx_counter = 0

    inputs, output = re.split("->", einsum)
    inputs = re.split(",", inputs.strip())
    for tn in inputs:
        content, idx_map, idx_counter = _gen_txt(content, idx_map, idx_counter, tn, size_dict)
    content += '---\n'
    content, _, _ = _gen_txt(content, idx_map, None, output.strip(), size_dict, dump_extents=False)

    assert filename.endswith('.txt')
    def dump():
        with open(filename, 'w') as f:
            f.write(content)
    call_by_root(dump)


def random_unitary(size, rng=None, dtype=np.float64, check=False):
    # the same functionality can be done with scipy.stats.unitary_group.rvs(),
    # but this is so simple that we just re-implement it here
    rng = np.random.default_rng(1234) if rng is None else rng  # TODO: honor a global seed?
    m = rng.standard_normal(size=(size, size), dtype=dtype) \
        + 1j*rng.standard_normal(size=(size, size), dtype=dtype)
    q, r = np.linalg.qr(m)
    d = np.diag(r)
    q *= d/abs(d)
    if check:
        is_unitary = np.allclose(
            np.abs(np.dot(q, q.T.conj()) - np.eye(size, dtype=q.dtype)),
            0,
        )
        if not is_unitary:
            warnings.warn("generated random matrix might not be unitary")
    return q


def is_running_mpiexec():
    # This is not 100% robust but should cover MPICH, Open MPI and Slurm
    # PMI_SIZE, Hydra(MPICH), OMPI_COMM_WORLD_SIZE(OpenMPI)
    if 'PMI_SIZE' in os.environ or \
       'OMPI_COMM_WORLD_SIZE' in os.environ:
        return True
    # SLURM_NPROCS is defined by Slurm
    if 'SLURM_NPROCS' in os.environ:
        nprocs = os.environ['SLURM_NPROCS']
        return nprocs != '1'
    # no environmental variable found
    return False


def is_running_mpi():
    if is_running_mpiexec():
        try:
            from mpi4py import MPI  # init!
        except ImportError as e:
            raise RuntimeError(
                'it seems you are running mpiexec/mpirun but mpi4py cannot be '
                'imported, maybe you forgot to install it?') from e
    else:
        MPI = None
    return MPI


def get_mpi_size():
    MPI = is_running_mpi()
    return MPI.COMM_WORLD.Get_size() if MPI else 1


def get_mpi_rank():
    MPI = is_running_mpi()
    return MPI.COMM_WORLD.Get_rank() if MPI else 0


def call_by_root(f, root=0):
    """ Call the callable f only by the root process. """
    MPI = is_running_mpi()
    if MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == root:
            return f()
    else:
        return f()


class MPHandler(logging.StreamHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MPI = is_running_mpi()
        if MPI:
            self.rank = MPI.COMM_WORLD.Get_rank()
        else:
            self.rank = 0

    def emit(self, record):
        # don't log unless I am the root process
        if self.rank == 0:
            super().emit(record)


def str_to_seq(data):
    data = data.split(',')
    out = []
    for i in data:
        if i:
            out.append(int(i))
    return out


def get_cpu_name():
    # This helper avoids the need of installing py-cpuinfo. This works
    # because we only support Linux.
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    m = re.search(r"model name.*\n", cpuinfo)
    if m:
        return m.group(0).split(':')[-1].strip()
    else:
        return f"unknown"


def get_gpu_driver_version():
    # this function will not raise

    inited = False

    try:
        # nvml comes with the driver, so it'd be always available
        from ctypes.util import find_library
        lib_name = find_library('nvidia-ml')
        lib = ctypes.CDLL(lib_name)
        init = lib.nvmlInit_v2
        func = lib.nvmlSystemGetDriverVersion
        shutdown = lib.nvmlShutdown
        out = ctypes.create_string_buffer(80)

        status = init()
        if status != 0:
            raise RuntimeError('cannot init nvml')
        inited = True

        status = func(ctypes.byref(out), 80)
        if status != 0:
            raise RuntimeError('cannot get driver version')
    except:
        ver = "N/A"
    else:
        ver = out.value.decode()
    finally:
        if inited:
            shutdown()

    return ver


class RawTextAndDefaultArgFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


class HashableDict(dict):

    def get_hash(self):
        # 1. we want a stable hash scheme, the built-in hash() is not
        # 2. but hash() requires __hash__() returning an int, while this is a str
        return hashlib.sha256(str(tuple(self.items())).encode()).hexdigest()


@dataclass
class Gate:
    """A data class for holding all gate-related information.

    Attributes:
        id: The gate identity.
        targets: The target qubit(s).
        controls: The control qubit(s), if any.
        matrix: The gate matrix.
        params: The gate parameter(s).
        name: The gate name.
    """
    id: str = ''
    targets: Union[int, Iterable[int]] = None
    controls: Optional[Union[int, Iterable[int]]] = None
    matrix: Optional[Union[Iterable[float], Iterable[complex]]] = None
    params: Optional[Union[float, Iterable[float]]] = None
    name: Optional[str] = ''

    def __post_init__(self):
        if not self.id:
            raise ValueError("gate id must be specified")
        if self.targets is None:
            raise ValueError("targets must be specified")
        if self.matrix is not None:
            if self.params is not None:
                raise ValueError("gate matrix and gate parameters cannot coexist")
            try:
                n_targets = len(self.targets)
            except TypeError:
                n_targets = 1  # targets is int
            try:
                # 1D/2D ndarray-like objects
                assert self.matrix.size == (2**n_targets)**2
            except AttributeError:
                # plain Python objects, must be 2D/nested (otherwise we'd have to
                # assume there's a certain memory layout); we're being sloppy
                # here and do not check the inner sequence lengths...
                try:
                    assert len(self.matrix) == 2**n_targets
                except Exception as e:
                    raise ValueError("gate matrix size must match targets") from e

    def __repr__(self):
        s = f"Gate(id={self.id}, targets={self.targets}"
        if self.controls is not None:
            s += f", controls={self.controls}"
        if self.matrix is not None:
            s += f", matrix={self.matrix}"
        elif self.params is not None:
            s += f", params={self.params}"
        if self.name:
            s += f", name={self.name}"
        s += ")"
        return s


def gen_run_env(gpu_device_properties):
    run_env = HashableDict({
        'hostname': platform.node(),
        'cpu_name': get_cpu_name(),
        'gpu_name': gpu_device_properties['name'].decode('utf-8'),
        'gpu_driver_ver': cp.cuda.runtime.driverGetVersion(),
        'gpu_runtime_ver': cp.cuda.runtime.runtimeGetVersion(),
        'nvml_driver_ver': get_gpu_driver_version(),
    })
    return run_env


def report(perf_time, cuda_time, post_time, ngpus, run_env, gpu_device_properties, benchmark_data):
    hostname = run_env['hostname']
    cpu_name = run_env['cpu_name']
    cpu_phy_mem = round(psutil.virtual_memory().total/1000000000, 2)
    cpu_used_mem = round(psutil.virtual_memory().used/1000000000, 2)
    cpu_phy_cores = psutil.cpu_count(logical=False)
    cpu_log_cores = psutil.cpu_count(logical=True)
    cpu_curr_freq = round(psutil.cpu_freq().current, 2)
    cpu_min_freq = psutil.cpu_freq().min
    cpu_max_freq = psutil.cpu_freq().max

    gpu_name = run_env['gpu_name']
    gpu_total_mem = round(gpu_device_properties['totalGlobalMem']/1000000000, 2)
    gpu_clock_rate = round(gpu_device_properties['clockRate']/1000, 2)
    gpu_multiprocessor_num = gpu_device_properties['multiProcessorCount']
    gpu_driver_ver = run_env['gpu_driver_ver']
    gpu_runtime_ver = run_env['gpu_runtime_ver']
    nvml_driver_ver = run_env['nvml_driver_ver']

    logger.debug(f' - hostname: {hostname}')
    logger.info(f' - [CPU] Averaged elapsed time: {perf_time:.9f} s')
    if post_time is not None:
        logger.info(f' - [CPU] Averaged postprocessing Time: {post_time:.6f} s')
        benchmark_data['cpu_post_time'] = post_time
    logger.info(f' - [CPU] Processor type: {cpu_name}')
    logger.debug(f' - [CPU] Total physical memory: {cpu_phy_mem} GB')
    logger.debug(f' - [CPU] Total used memory: {cpu_used_mem} GB')
    logger.debug(f' - [CPU] Number of physical cores: {cpu_phy_cores}, and logical cores: {cpu_log_cores}')
    logger.debug(f' - [CPU] Frequency current (Mhz): {cpu_curr_freq}, min: {cpu_min_freq}, and max: {cpu_max_freq}')
    logger.info(' -')
    logger.info(f' - [GPU] Averaged elapsed time: {cuda_time:.9f} s {"(unused)" if ngpus == 0 else ""}')
    logger.info(f' - [GPU] GPU device name: {gpu_name}')
    logger.debug(f' - [GPU] Total global memory: {gpu_total_mem} GB')
    logger.debug(f' - [GPU] Clock frequency (Mhz): {gpu_clock_rate}')
    logger.debug(f' - [GPU] Multi processor count: {gpu_multiprocessor_num}')
    logger.debug(f' - [GPU] CUDA driver version: {gpu_driver_ver} ({nvml_driver_ver})')
    logger.debug(f' - [GPU] CUDA runtime version: {gpu_runtime_ver}')
    logger.info('')

    benchmark_data['cpu_time'] = perf_time
    benchmark_data['cpu_phy_mem'] = cpu_phy_mem
    benchmark_data['cpu_used_mem'] = cpu_used_mem
    benchmark_data['cpu_phy_cores'] = cpu_phy_cores
    benchmark_data['cpu_log_cores'] = cpu_log_cores
    benchmark_data['cpu_current_freq'] = cpu_curr_freq

    benchmark_data['gpu_time'] = cuda_time
    benchmark_data['gpu_total_mem'] = gpu_total_mem
    benchmark_data['gpu_clock_freq'] = gpu_clock_rate
    benchmark_data['gpu_multiprocessor_num'] = gpu_multiprocessor_num

    return benchmark_data


def save_benchmark_data(
        num_qubits, sim_config_hash, benchmark_data, full_data, filepath, save=True):
    try:
        full_data[num_qubits][sim_config_hash] = benchmark_data
    except KeyError:
        if num_qubits not in full_data:
            full_data[num_qubits] = {}

        if sim_config_hash not in full_data[num_qubits]:
            full_data[num_qubits][sim_config_hash] = {}

        full_data[num_qubits][sim_config_hash] = benchmark_data

    if save:
        def dump():
            with open(filepath, 'w') as f:
                json.dump(full_data, f, indent=2)
        call_by_root(dump)
        logger.debug(f'Saved {filepath} as JSON')

    return full_data


def load_benchmark_data(filepath):
    try:
        with open(filepath, 'r') as f:
            full_data = json.load(f)
            logger.debug(f'Loaded {filepath} as JSON')
    # If the data file does not exist, we'll create it later
    except FileNotFoundError:
        full_data = {}
        logger.debug(f'{filepath} not found')

    return full_data


def create_cache(cache_dir, required_subdirs):
    for subdir in required_subdirs:
        path = os.path.join(cache_dir, subdir)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)


# TODO: upstream this to cupyx.profiler.benchmark
class L2flush:
    """ Handly utility for flushing the current device's L2 cache.

    This instance must be created and used on the same (CuPy's) current device/stream
    as those used by the target workload.

    Reimplementation of the l2flush class from NVBench, see
    https://github.com/NVIDIA/nvbench/blob/main/nvbench/detail/l2flush.cuh.
    """
    def __init__(self):
        self.l2_size = cp.cuda.Device().attributes['L2CacheSize']
        self.mem = cp.cuda.alloc(self.l2_size) if self.l2_size > 0 else None

    def flush(self):
        if self.mem:
            self.mem.memset_async(0, self.l2_size)


# TODO: we should convince upstream to allow this use case.
def benchmark_with_prerun(
        func, args=(), kwargs={}, *,
        n_warmup=10, n_repeat=10000, pre_run=None):
    """A simplified version of cupyx.profiler.benchmark(), with the additional
    support for a user-supplied function ("pre_run") that's run every time
    before the target "func" is run.

    This is simplifed to only permit using on single GPUs.
    """
    e1 = cp.cuda.Event()
    e2 = cp.cuda.Event()
    try:
        from cupyx.profiler._time import _PerfCaseResult
    except ImportError:
        _PerfCaseResult = None
        class _Result: pass
    cpu_times = []
    gpu_times = [[]]

    for _ in range(n_warmup):
        func(*args, **kwargs)

    for _ in range(n_repeat):
        if pre_run:
            pre_run(*args, **kwargs)
        e1.record()
        t1 = time.perf_counter()
        func(*args, **kwargs)
        t2 = time.perf_counter()
        e2.record()
        e2.synchronize()

        cpu_times.append(t2-t1)
        gpu_times[0].append(cp.cuda.get_elapsed_time(e1, e2)*1E-3)

    if _PerfCaseResult:
        result = _PerfCaseResult(
            func.__name__,
            np.asarray([cpu_times] + gpu_times, dtype=np.float64),
            (cp.cuda.Device().id,))
    else:
        result = _Result()
        result.cpu_times = cpu_times
        result.gpu_times = gpu_times

    return result


class EarlyReturnError(RuntimeError): pass


is_unique = lambda a: len(set(a)) == len(a)
is_disjoint = lambda a, b: not bool(set(a) & set(b))


def check_targets_controls(targets, controls, n_qubits):
    # simple checks for targets and controls
    assert len(targets) >= 1, "must have at least 1 target qubit"
    assert is_unique(targets), "qubit indices in targets must be unique"
    assert is_unique(controls), "qubit indices in controls must be unique"
    assert is_disjoint(targets, controls), "qubit indices in targets and controls must be disjoint"
    assert all(0 <= q and q < n_qubits for q in targets + controls), f"target and control qubit indices must be in range [0, {n_qubits})"


def check_sequence(seq, expected_size=None, max_size=None, name=''):
    if expected_size is not None:
        assert len(seq) == expected_size, f"the provided {name} must be of length {expected_size}"
        size = expected_size
    elif max_size is not None:
        assert len(seq) <= max_size, f"the provided {name} must have length <= {max_size}"
        size = max_size
    else:
        assert False
    assert is_unique(seq), f"the provided {name} must have non-repetitve entries"
    assert all(0 <= i and i < size for i in seq), f"entries in the {name} must be in [0, {size})"
