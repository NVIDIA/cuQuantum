# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import time
import warnings
from math import log10, log2
import numpy as np
import cupy as cp
try:
    import cuquantum
except ImportError:
    cuquantum = None

from .backend import Backend
from .._utils import convert_einsum_to_txt, generate_size_dict_from_operands, is_running_mpiexec
from ..constants import LOGGER_NAME


# set up a logger
logger = logging.getLogger(LOGGER_NAME)


if cuquantum is not None:
    bindings = getattr(cuquantum, 'bindings', None)
    if bindings is not None:
       # cuquantum >= 25.03
       cutn = bindings.cutensornet
       tensornet = cuquantum.tensornet
    else:
       # cuquantum < 25.03
       cutn = cuquantum.cutensornet
       tensornet = cutn
       logger.warning("Warning:support of cuquantum-python<25.3 is deprecated and will be removed in future release")
else:
    cutn = None
    tensornet = None


class cuTensorNet(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, **kwargs):
        if cutn is None:
            raise RuntimeError("cuquantum-python is not installed")
        if ngpus != 1:
            raise ValueError("the cutn backend must be run with --ngpus 1 (regardless if MPI is in use)")

        self.ncpu_threads = ncpu_threads
        self.precision = precision
        self.nqubits = kwargs.pop('nqubits')
        self.rank = 0
        self.handle = cutn.create()
        self.meta = {}
        try:
            # cuQuantum Python 22.11+ supports nonblocking & auto-MPI
            opts = tensornet.NetworkOptions(handle=self.handle, blocking="auto")
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
            opts = tensornet.NetworkOptions(handle=self.handle)
        self.network_opts = opts
        self.n_hyper_samples = kwargs.pop('nhypersamples')
        self.version = cutn.get_version()

        self.meta['ncputhreads'] = self.ncpu_threads
        self.meta['nhypersamples'] = self.n_hyper_samples

    def __del__(self):
        cutn.destroy(self.handle)

    def preprocess_circuit(self, circuit, *args, **kwargs):
        circuit_filename = kwargs.pop('circuit_filename')
        self.compute_mode = kwargs.pop('compute_mode')
        self.pauli = kwargs.pop('pauli')
        valid_choices = ['amplitude', 'expectation', 'statevector']
        if self.compute_mode not in valid_choices:
            raise ValueError(f"The '{self.compute_mode}' computation mode is not supported for this backend. Supported modes are: {valid_choices}")
        
        t1 = time.perf_counter()
        if self.precision == 'single':
            circuit_converter = tensornet.CircuitToEinsum(circuit, dtype='complex64', backend=cp)
        else:
            circuit_converter = tensornet.CircuitToEinsum(circuit, dtype='complex128', backend=cp)
        t2 = time.perf_counter()
        time_circ2einsum = t2 - t1
        
        t1 = time.perf_counter()
        if self.compute_mode == 'amplitude':
            # any bitstring would give same TN topology, so let's just pick "000...0"
            self.expression, self.operands = circuit_converter.amplitude('0'*self.nqubits)
        elif self.compute_mode == 'statevector':
            self.expression, self.operands = circuit_converter.state_vector()
        elif self.compute_mode == 'expectation':
            # new in cuQuantum Python 22.11
            assert self.pauli is not None
            logger.info(f"compute expectation value for Pauli string: {self.pauli}")
            self.expression, self.operands = circuit_converter.expectation(self.pauli)
        else:
            # TODO: add other CircuitToEinsum methods?
            raise NotImplementedError(f"the target {self.compute_mode} is not supported")
        t2 = time.perf_counter()
        time_tn = t2 - t1
        
        tn_format = os.environ.get('CUTENSORNET_DUMP_TN')
        if tn_format == 'txt':
            size_dict = generate_size_dict_from_operands(
                self.expression, self.operands)
            convert_einsum_to_txt(
                self.expression, size_dict, circuit_filename + '.txt')
        elif tn_format is not None:
            # TODO: dump expression & size_dict as plain unicode?
            raise NotImplementedError(f"the TN format {tn_format} is not supported")
        self.network = tensornet.Network(
            self.expression, *self.operands, options=self.network_opts)

        t1 = time.perf_counter()
        path, opt_info = self.network.contract_path(
            # TODO: samples may be too large for small circuits
            optimize={'samples': self.n_hyper_samples, 'threads': self.ncpu_threads})
        t2 = time.perf_counter()
        time_path = t2 - t1

        self.opt_info = opt_info # cuTensorNet returns "real-number" Flops. To get the true FLOP count, multiply it by 4
        
        self.meta['compute-mode'] = f'{self.compute_mode}()'
        self.meta['circuitToEinsum time'] = f'{time_circ2einsum + time_tn} s'
        self.meta['contract-path time'] = f'{time_path} s'
        self.meta['FLOPs'] = f"{self.opt_info.opt_cost * 4:.2e}"
        self.meta['largest-intermediate size'] = f"{opt_info.largest_intermediate:.2e}"
        logger.info(f'data: {self.meta}')

        pre_data = self.meta
        return pre_data
    
    def run(self, circuit, nshots=0):
        results = self.network.contract()
        return {'results': None, 'post_results': None, 'run_data': {}}
