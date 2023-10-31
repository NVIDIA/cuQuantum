# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
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


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


class Pennylane(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, identifier=None, **kwargs):
        if pennylane is None:
            raise RuntimeError("pennylane is not installed")
        self.dtype = np.complex64 if precision == "single" else np.complex128
        self.identifier = identifier
        self.ngpus = ngpus
        self.ncpu_threads = ncpu_threads
        self.nqubits = kwargs.pop('nqubits')
        self.circuit = None
        self.version = self.find_version(identifier) 

    def find_version(self, identifier):
        if identifier == "pennylane-lightning-gpu":
            if self.ngpus == 1:
                try:
                    import pennylane_lightning_gpu
                except ImportError as e:
                    raise RuntimeError("PennyLane-Lightning-GPU plugin is not installed") from e
            else:
                raise ValueError(f"cannot specify --ngpus > 1 for the backend {identifier}")
            ver = pennylane_lightning_gpu.__version__
        elif identifier == "pennylane-lightning-kokkos":
            try:
                import pennylane_lightning_kokkos
            except ImportError as e:
                raise RuntimeError("PennyLane-Lightning-Kokkos plugin is not installed") from e
            ver = pennylane_lightning_kokkos.__version__
        elif identifier == "pennylane-lightning-qubit":
            try:
                from pennylane_lightning import lightning_qubit
            except ImportError as e:
                raise RuntimeError("PennyLane-Lightning plugin is not installed") from e
            ver = lightning_qubit.__version__
        else: # identifier == "pennylane"
            ver = pennylane.__version__
        return ver

    def _make_qnode(self, circuit, nshots=1024, **kwargs):
        if self.identifier == "pennylane-lightning-gpu":
            dev = pennylane.device("lightning.gpu", wires=self.nqubits, shots=nshots, c_dtype=self.dtype)
        elif self.identifier == "pennylane-lightning-kokkos":
            # there's no way for us to query what execution space (=backend) that kokkos supports at runtime,
            # so let's just set up Kokkos::InitArguments and hope kokkos to do the right thing...
            try:
                import pennylane_lightning_kokkos
            except ImportError as e:
                raise RuntimeError("PennyLane-Lightning-Kokkos plugin is not installed") from e
            args = pennylane_lightning_kokkos.lightning_kokkos.InitArguments()
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
            dev = pennylane.device("default.qubit", wires=self.nqubits, shots=nshots, c_dtype=self.dtype)
        else:
            raise ValueError(f"the backend {self.identifier} is not recognized")

        qnode = pennylane.QNode(circuit, device=dev)
        return qnode

    def preprocess_circuit(self, circuit, *args, **kwargs):
        if _internal_utils is not None:
            _internal_utils.preprocess_circuit(self.identifier, circuit, *args, **kwargs)
        
        nshots = kwargs.get('nshots', 1024)
        t1 = time.perf_counter()
        self.circuit = self._make_qnode(circuit, nshots, **kwargs)
        t2 = time.perf_counter()
        time_make_qnode = t2 - t1
        logger.info(f'make qnode took {time_make_qnode} s')
        return {'make_qnode': time_make_qnode}

    def run(self, circuit, nshots=1024):
        # both circuit & nshots are set in preprocess_circuit()
        results = self.circuit()
        post_res = None # TODO
        run_data = {}
        return {'results': results, 'post_results': post_res, 'run_data': run_data}


PnyLightningGpu = functools.partial(Pennylane, identifier='pennylane-lightning-gpu')
PnyLightningCpu = functools.partial(Pennylane, identifier='pennylane-lightning-qubit')
PnyLightningKokkos = functools.partial(Pennylane, identifier='pennylane-lightning-kokkos')
Pny = functools.partial(Pennylane, identifier='pennylane')
