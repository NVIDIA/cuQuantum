# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import time
import warnings

import numpy as np
import cupy as cp
from cuquantum import contract, contract_path, CircuitToEinsum
from cuquantum import cutensornet as cutn

from .backend import Backend
from .._utils import convert_einsum_to_txt, generate_size_dict_from_operands, is_running_mpiexec


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


class cuTensorNet(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, **kwargs):
        if ngpus != 1:
            raise ValueError("the cutn backend must be run with --ngpus 1 (regardless if MPI is in use)")

        self.ncpu_threads = ncpu_threads
        self.precision = precision
        self.nqubits = kwargs.pop('nqubits')
        self.rank = 0
        self.handle = cutn.create()
        try:
            # cuQuantum Python 22.11+ supports nonblocking & auto-MPI
            opts = cutn.NetworkOptions(handle=self.handle, blocking="auto")
            if is_running_mpiexec():
                from mpi4py import MPI  # init should be already done earlier
                comm = MPI.COMM_WORLD
                rank, size = comm.Get_rank(), comm.Get_size()
                device_id = rank % cp.cuda.runtime.getDeviceCount()
                cp.cuda.Device(device_id).use()
                cutn.distributed_reset_configuration(
                    self.handle, *cutn.get_mpi_comm_pointer(comm)
                )
                logger.debug("enable MPI support for cuTensorNet")
                self.rank = rank
        except (TypeError, AttributeError):
            # cuQuantum Python 22.07 or below
            opts = cutn.NetworkOptions(handle=self.handle)
        self.network_opts = opts
        self.n_samples = kwargs.pop('nhypersamples')
        self.version = cutn.get_version()

    def __del__(self):
        cutn.destroy(self.handle)

    def preprocess_circuit(self, circuit, *args, **kwargs):
        circuit_filename = kwargs.pop('circuit_filename')
        target = kwargs.pop('target')
        pauli = kwargs.pop('pauli')
        preprocess_data = {}

        t1 = time.perf_counter()

        if self.precision == 'single':
            circuit_converter = CircuitToEinsum(circuit, dtype='complex64', backend=cp)
        else:
            circuit_converter = CircuitToEinsum(circuit, dtype='complex128', backend=cp)

        t2 = time.perf_counter()
        time_circ2einsum = t2 - t1
        logger.info(f'CircuitToEinsum took {time_circ2einsum} s')

        t1 = time.perf_counter()
        if target == 'amplitude':
            # any bitstring would give same TN topology, so let's just pick "000...0"
            self.expression, self.operands = circuit_converter.amplitude('0'*self.nqubits)
        elif target == 'state_vector':
            self.expression, self.operands = circuit_converter.state_vector()
        elif target == 'expectation':
            # new in cuQuantum Python 22.11
            assert pauli is not None
            logger.info(f"compute expectation value for Pauli string: {pauli}")
            self.expression, self.operands = circuit_converter.expectation(pauli)
        else:
            # TODO: add other CircuitToEinsum methods?
            raise NotImplementedError(f"the target {target} is not supported")
        t2 = time.perf_counter()
        time_tn = t2 - t1
        logger.info(f'{target}() took {time_tn} s')

        tn_format = os.environ.get('CUTENSORNET_DUMP_TN')
        if tn_format == 'txt':
            size_dict = generate_size_dict_from_operands(
                self.expression, self.operands)
            convert_einsum_to_txt(
                self.expression, size_dict, circuit_filename + '.txt')
        elif tn_format is not None:
            # TODO: dump expression & size_dict as plain unicode?
            raise NotImplementedError(f"the TN format {tn_format} is not supported")
        self.network = cutn.Network(
            self.expression, *self.operands, options=self.network_opts)

        t1 = time.perf_counter()
        path, opt_info = self.network.contract_path(
            # TODO: samples may be too large for small circuits
            optimize={'samples': self.n_samples, 'threads': self.ncpu_threads})
        t2 = time.perf_counter()
        time_path = t2 - t1
        logger.info(f'contract_path() took {time_path} s')
        logger.debug(f'# samples: {self.n_samples}')
        logger.debug(opt_info)

        self.path = path
        self.opt_info = opt_info
        preprocess_data = {
            'CircuitToEinsum': time_circ2einsum,
            target:            time_tn,
            'contract_path':   time_path,
        }

        return preprocess_data

    def run(self, circuit, nshots=0):
        if self.rank == 0 and nshots > 0:
            warnings.warn("the cutn backend does not support sampling")

        self.network.contract()

        # TODO: support these return values?
        return {'results': None, 'post_results': None, 'run_data': {}}
