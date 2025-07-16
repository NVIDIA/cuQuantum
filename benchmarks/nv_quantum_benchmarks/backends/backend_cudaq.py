# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import functools
import warnings
import os
import numpy as np
try:
    import cudaq
except ImportError:
    cudaq = None

from .backend import Backend
from ..constants import LOGGER_NAME


# set up a logger
logger = logging.getLogger(LOGGER_NAME)


class Cudaq(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, identifier=None, **kwargs):
        if cudaq is None:
            raise RuntimeError("cudaq is not installed")
        # if ncpu_threads > 1:
        #     warnings.warn("cannot set the number of CPU threads for the cudaq backend")
        self.precision = precision
        self.identifier = identifier
        self.nqubits = kwargs.pop('nqubits')
        self.is_mpi_used = False
        self.version = cudaq.__version__
        self.set_cudaq_backend(identifier, ngpus, ncpu_threads, *args, **kwargs)
        self.meta = {}
        self.meta['backend'] = f"{self.identifier}-{self.version} precision={self.precision}"

    def preprocess_circuit(self, circuit, *args, **kwargs):
        self.compute_mode = kwargs.pop('compute_mode')
        self.pauli = kwargs.pop('pauli')
        valid_choices = ['expectation', 'sampling', 'amplitude']
        if self.compute_mode not in valid_choices:
            raise ValueError(f"The string '{self.compute_mode}' is not a valid option for --compute-mode argument. Valid options are: {valid_choices}")

        self.meta['compute-mode'] = f'{self.compute_mode}()'
        logger.info(f'data: {self.meta}')
        if self.compute_mode == 'expectation':
            logger.info(f"compute expectation value for Pauli string: {self.pauli}")
        return {}

    def set_cudaq_backend(self, identifier, ngpus, ncpu_threads, *args, **kwargs):
        if identifier.endswith("cusv"):
            if ngpus != 1:
                raise ValueError("the cusv backend requires 1 GPU per process (--ngpus 1)")
            nfused = kwargs.pop('nfused')
            os.environ['CUDAQ_FUSION_MAX_QUBITS'] = str(nfused)
            os.environ['CUDAQ_FUSION_NUM_HOST_THREADS'] = str(ncpu_threads)
            if self.precision == 'single':
                cudaq.set_target('nvidia', option='fp32')
            else:
                cudaq.set_target('nvidia', option='fp64')
        elif identifier.endswith("mgpu"):
            if ngpus < 1:
                raise ValueError(f"need to specify --ngpus for the backend {identifier}")
            elif ngpus > 1: # Only set MPI for GPUS > 1, no need to use MPI for 1 GPU
                # check for proper environment variables
                if not (os.getenv('CUDAQ_MGPU_P2P_DEVICE_BITS') or os.getenv('CUDAQ_GPU_FABRIC')):
                    os.environ['CUDAQ_GPU_FABRIC'] = "NVL"
                cudaq.mpi.initialize()
                self.is_mpi_used = True
            nfused = kwargs.pop('nfused')
            os.environ['CUDAQ_FUSION_MAX_QUBITS'] = str(nfused)
            os.environ['CUDAQ_FUSION_NUM_HOST_THREADS'] = str(ncpu_threads)
            if self.precision == 'single':
                cudaq.set_target('nvidia', option='mgpu,fp32')
            else:
                cudaq.set_target('nvidia', option='mgpu,fp64')
        elif identifier.endswith("cpu"):
            cudaq.set_target('qpp-cpu')
        else:
            raise ValueError(f"the backend {identifier} is not recognized")
        cudaq.set_random_seed(123) # do after set_target https://github.com/NVIDIA/cuda-quantum/issues/2760

    def run(self, circuit, nshots=1024):
        run_data = {}
        if self.compute_mode == 'sampling':
            samples = cudaq.sample(circuit, shots_count=nshots)
        elif self.compute_mode == 'expectation':
            pauli_str = ''.join(self.pauli)
            spin_operator = cudaq.SpinOperator.from_word(pauli_str)
            res = cudaq.observe(circuit, spin_operator)
            exp = res.expectation()
        elif self.compute_mode == 'amplitude':
            bitstring = '0' * self.nqubits
            state = cudaq.get_state(circuit)
            amp = state.amplitude(bitstring)
        if self.is_mpi_used:
            cudaq.mpi.finalize()

        return {'results': None, 'post_results': None, 'run_data': run_data}


CudaqCusv = functools.partial(Cudaq, identifier="cudaq-cusv")
CudaqMgpu = functools.partial(Cudaq, identifier="cudaq-mgpu")
CudaqCpu = functools.partial(Cudaq, identifier="cudaq-cpu")
