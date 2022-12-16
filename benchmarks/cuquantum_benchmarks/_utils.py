# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import ctypes
import hashlib
import logging
import os
import re

import numpy as np
from cuquantum.cutensornet._internal.einsum_parser import create_size_dict


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
    write_on_rank_0(dump)


def random_unitary(size, rng, dtype=np.float64):
    # the same functionality can be done with scipy.stats.unitary_group.rvs(),
    # but this is too simple that we just re-implement it here
    m = rng.standard_normal(size=(size, size), dtype=dtype) \
        + 1j*rng.standard_normal(size=(size, size), dtype=dtype)
    q, r = np.linalg.qr(m)
    d = np.diag(r)
    q *= d/abs(d)
    return q


def is_running_mpiexec():
    # This is not 100% robust but should cover MPICH & Open MPI
    for key in os.environ.keys():
        if key.startswith('PMI_') or key.startswith('OMPI_COMM_WORLD_'):
            return True
    else:
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


def write_on_rank_0(f):
    MPI = is_running_mpi()
    if MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            f()
    else:
        f()


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
        assert False, f"getting cpu info failed"


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
