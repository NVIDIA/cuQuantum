# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import os
import time
import warnings
import numpy as np
try:
    import pennylane
except ImportError:
    pennylane = None

try:
    from .. import _internal_utils
except ImportError:
    _internal_utils = None
from .backend import Backend
from ..constants import LOGGER_NAME


# set up a logger
logger = logging.getLogger(LOGGER_NAME)


class Pennylane(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, identifier=None, **kwargs):
        if pennylane is None:
            raise RuntimeError("pennylane is not installed")
        self.dtype = np.complex64 if precision == "single" else np.complex128
        self.identifier = identifier
        self.ngpus = ngpus
        self.ncpu_threads = ncpu_threads
        self.nqubits = kwargs.pop('nqubits')
        self.version = self.find_version(identifier) 
        self.meta = {}
        self.meta['ncputhreads'] = ncpu_threads

    def preprocess_circuit(self, circuit, *args, **kwargs):
        if _internal_utils is not None:
            _internal_utils.preprocess_circuit(self.identifier, circuit, *args, **kwargs)
        
        self.circuit = circuit
        self.compute_mode = kwargs.pop('compute_mode')
        valid_choices = ['statevector', 'sampling']
        if self.compute_mode not in valid_choices:
            raise ValueError(f"The '{self.compute_mode}' computation mode is not supported for this backend. Supported modes are: {valid_choices}")
        
        nshots = kwargs.get('nshots', 1024)
        t1 = time.perf_counter()
        if self.compute_mode == 'statevector':
            self.device = self._make_device(nshots=None, **kwargs)
        elif self.compute_mode == 'sampling':
            self.device = self._make_device(nshots=nshots, **kwargs)
        t2 = time.perf_counter()
        time_make_device = t2 - t1
        
        self.meta['compute-mode'] = f'{self.compute_mode}()'
        self.meta['make-device time:'] = f'{time_make_device} s'
        logger.info(f'data: {self.meta}')

        pre_data = self.meta
        return pre_data

    def find_version(self, identifier):
        if identifier == "pennylane-lightning-gpu":
            if self.ngpus == 1:
                try:
                    from pennylane_lightning.lightning_gpu import LightningGPU
                    return LightningGPU.version
                except ImportError:
                    try: # pre pennylane_lightning 0.33.0 version
                        import pennylane_lightning_gpu
                        return pennylane_lightning_gpu.__version__
                    except ImportError:
                        raise RuntimeError("PennyLane-Lightning-GPU plugin is not installed")
            else:
                raise ValueError(f"cannot specify --ngpus > 1 for the backend {identifier}")
        elif identifier == "pennylane-lightning-kokkos":
            try:
                from pennylane_lightning.lightning_kokkos import LightningKokkos
                return LightningKokkos.version
            except ImportError:
                try: # pre pennylane_lightning 0.33.0 version
                    import pennylane_lightning_kokkos
                    return pennylane_lightning_kokkos.__version__
                except ImportError:
                    raise RuntimeError("PennyLane-Lightning-Kokkos plugin is not installed")
        elif identifier == "pennylane-lightning-qubit":
            try:
                from pennylane_lightning import lightning_qubit
                return lightning_qubit.__version__
            except ImportError as e:
                raise RuntimeError("PennyLane-Lightning plugin is not installed") from e
        else: # identifier == "pennylane"
            return pennylane.__version__
        
    def _make_device(self, nshots=None, **kwargs):
        if self.identifier == "pennylane-lightning-gpu":
            dev = pennylane.device("lightning.gpu", wires=self.nqubits, shots=nshots, c_dtype=self.dtype)
        elif self.identifier == "pennylane-lightning-kokkos":
            # there's no way for us to query what execution space (=backend) that kokkos supports at runtime,
            # so let's just set up Kokkos::InitArguments and hope kokkos to do the right thing...
            dev = None
            try:
                if self.ncpu_threads > 1 :
                    warnings.warn(f"--ncputhreads is ignored for {self.identifier}", stacklevel=2)
                dev = pennylane.device(
                "lightning.kokkos", wires=self.nqubits, shots=nshots, c_dtype=self.dtype,
                sync=False)
            except ImportError:
                try: # pre pennylane_lightning 0.33.0 version
                    from pennylane_lightning_kokkos.lightning_kokkos import InitArguments
                    args = InitArguments()
                    args.num_threads = self.ncpu_threads
                    args.disable_warnings = int(logger.getEffectiveLevel() != logging.DEBUG)
                    ## Disable MPI because it's unclear if pennylane actually supports it (at least it's untested)
                    # # if we're running MPI, we want to know now and get it init'd before kokkos is
                    # MPI = is_running_mpi()
                    # if MPI:
                    #     comm = MPI.COMM_WORLD
                    #     args.ndevices = min(comm.Get_size(), self.ngpus)  # note: kokkos uses 1 GPU per process
                    dev = pennylane.device(
                        "lightning.kokkos", wires=self.nqubits, shots=nshots, c_dtype=self.dtype,
                        sync=False,
                        kokkos_args=args)
                except ImportError:
                    raise RuntimeError("Could not load PennyLane-Lightning-Kokkos plugin. Is it installed?")
        elif self.identifier == "pennylane-lightning-qubit":
            if self.ngpus != 0:
                raise ValueError(f"cannot specify --ngpus for the backend {self.identifier}")
            if self.ncpu_threads > 1 and self.ncpu_threads != int(os.environ.get("OMP_NUM_THREADS", "-1")):
                warnings.warn(f"--ncputhreads is ignored, for {self.identifier} please set the env var OMP_NUM_THREADS instead",
                              stacklevel=2)
            dev = pennylane.device("lightning.qubit", wires=self.nqubits, shots=nshots, c_dtype=self.dtype)
        elif self.identifier == "pennylane":
            if self.ngpus != 0:
                raise ValueError(f"cannot specify --ngpus for the backend {self.identifier}")
            if self.dtype == np.complex64:
                raise ValueError("As of version 0.33.0, Pennylane's default.qubit device only supports double precision.")
            dev = pennylane.device("default.qubit", wires=self.nqubits, shots=nshots)
        else:
            raise ValueError(f"the backend {self.identifier} is not recognized")
        
        return dev

    def state_vector_qnode(self):
        @pennylane.qnode(self.device)
        def circuit():
            self.circuit()
            return pennylane.state()
        return circuit()

    def sampling_qnode(self):
        @pennylane.qnode(self.device)
        def circuit():
            self.circuit()
            return pennylane.counts(wires=range(self.nqubits))
        return circuit()

    def run(self, circuit, nshots=1024):
        if self.compute_mode == 'sampling':
            samples = self.sampling_qnode() 
        elif self.compute_mode == 'statevector':
            sv = self.state_vector_qnode() 

        return {'results': None, 'post_results': None, 'run_data': {}}


PnyLightningGpu = functools.partial(Pennylane, identifier='pennylane-lightning-gpu')
PnyLightningCpu = functools.partial(Pennylane, identifier='pennylane-lightning-qubit')
PnyLightningKokkos = functools.partial(Pennylane, identifier='pennylane-lightning-kokkos')
Pny = functools.partial(Pennylane, identifier='pennylane')
