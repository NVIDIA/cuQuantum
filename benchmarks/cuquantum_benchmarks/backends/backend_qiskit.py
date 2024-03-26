# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import functools
import logging
import time
from importlib.metadata import version

import numpy as np
import cupy as cp
try:
    import qiskit
except ImportError:
    qiskit = None

try:
    from .. import _internal_utils
except ImportError:
    _internal_utils = None
from .backend import Backend
from .._utils import get_mpi_size, get_mpi_rank
from .._utils import call_by_root, EarlyReturnError


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


class Qiskit(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, identifier=None, **kwargs):
        if qiskit is None:
            raise RuntimeError("qiskit is not installed")
        self.precision = precision
        self.identifier = identifier
        self.nqubits = kwargs.pop('nqubits')
        self.version = self.find_version(identifier)
        self.backend = self.create_aer_backend(self.identifier, ngpus, ncpu_threads, *args, **kwargs)

    def find_version(self, identifier):
        if identifier == 'cusvaer':
            return version('cusvaer')

        if hasattr(qiskit_aer, "__version__"):
            return qiskit_aer.__version__
        else:
            return qiskit.__qiskit_version__['qiskit-aer']
    
    def preprocess_circuit(self, circuit, *args, **kwargs):
        if _internal_utils is not None:
            _internal_utils.preprocess_circuit(self.identifier, circuit, *args, **kwargs)
        
        t0 = time.perf_counter()
        self.transpiled_qc = qiskit.transpile(circuit, self.backend) # (circuit, basis_gates=['u3', 'cx'], backend=self.backend)
        t1 = time.perf_counter()
        time_transpile = t1 - t0
        logger.info(f'transpile took {time_transpile} s')
        return {'transpile': time_transpile}

    def run(self, circuit, nshots=1024):
        run_data = {}
        transpiled_qc = self.transpiled_qc
        if nshots > 0:
            results = self.backend.run(transpiled_qc, shots=nshots, memory=True)
        else:
            results = self.backend.run(transpiled_qc, shots=0, memory=True)
        # workaround for memory allocation failure for cusvaer 22.11/23.03
        if self.identifier == 'cusvaer' and self._need_sync():
            self._synchronize()

        post_res_list = results.result().get_memory()
        post_res_list = [list(i) for i in post_res_list]
        post_res = np.array(post_res_list)
        return {'results': results, 'post_results': post_res, 'run_data': run_data}

    def create_aer_backend(self, identifier, ngpus, ncpu_threads, *args, **kwargs):
        nfused = kwargs.pop('nfused')
        try:
            # we defer importing Aer as late as possible, due to a bug it has that
            # could init all GPUs prematurely
            if hasattr(qiskit, "__version__") and qiskit.__version__ >= "1.0.0":
                from qiskit_aer import AerSimulator
            else:
                from qiskit.providers.aer import AerSimulator
        except ImportError as e:
            raise RuntimeError("qiskit-aer (or qiskit-aer-gpu) is not installed") from e

        if identifier == 'cusvaer':
            import cusvaer

            cusvaer_global_index_bits = kwargs.pop('cusvaer_global_index_bits')
            cusvaer_p2p_device_bits = kwargs.pop('cusvaer_p2p_device_bits')
            cusvaer_comm_plugin_type = kwargs.pop('cusvaer_comm_plugin_type')
            cusvaer_comm_plugin_soname = kwargs.pop('cusvaer_comm_plugin_soname')
            # convert comm plugin type to enum
            if not cusvaer_comm_plugin_type:
                cusvaer_comm_plugin_type = cusvaer.CommPluginType.MPI_AUTO
            elif cusvaer_comm_plugin_type == 'self':
                cusvaer_comm_plugin_type = cusvaer.CommPluginType.SELF
            elif cusvaer_comm_plugin_type == 'mpi_auto':
                cusvaer_comm_plugin_type = cusvaer.CommPluginType.MPI_AUTO
            elif cusvaer_comm_plugin_type == 'mpi_openmpi':
                cusvaer_comm_plugin_type = cusvaer.CommPluginType.MPI_OPENMPI
            elif cusvaer_comm_plugin_type == 'mpi_mpich':
                cusvaer_comm_plugin_type = cusvaer.CommPluginType.MPI_MPICH
            elif cusvaer_comm_plugin_type == 'external':
                cusvaer_comm_plugin_type = cusvaer.CommPluginType.EXTERNAL
            else:
                raise ValueError(f"Unknown cusvaer_comm_plugin_type, {cusvaer_comm_plugin_type}")
            if not cusvaer_comm_plugin_soname:  # empty string
                if cusvaer_comm_plugin_type == cusvaer.CommPluginType.EXTERNAL:
                    raise ValueError("cusvaer_comm_plugin_soname should be specified "
                                     "if cusvaer_comm_plugin_type=external is specified")
                cusvaer_comm_plugin_soname = None
            cusvaer_data_transfer_buffer_bits = kwargs.pop('cusvaer_data_transfer_buffer_bits')

            if ngpus != 1:
                raise ValueError("the cusvaer requires 1 GPU per process (--ngpus 1)")
            try:
                backend = AerSimulator(
                    method='statevector', device="GPU", cusvaer_enable=True, noise_model=None,
                    fusion_max_qubit=nfused,
                    cusvaer_global_index_bits=cusvaer_global_index_bits,
                    cusvaer_p2p_device_bits=cusvaer_p2p_device_bits,
                    precision=self.precision,
                    cusvaer_data_transfer_buffer_bits=cusvaer_data_transfer_buffer_bits,
                    cusvaer_comm_plugin_type=cusvaer_comm_plugin_type,
                    cusvaer_comm_plugin_soname=cusvaer_comm_plugin_soname
                )
            except:  # AerError
                raise RuntimeError(
                    "the cusvaer backend is only available in cuQuantum Appliance "
                    "container 22.11+")
        elif identifier == "aer-cuda":
            if ngpus >= 1:
                blocking_enable, blocking_qubits = self.get_aer_blocking_setup(ngpus)
                try:
                    # use cuQuantum Appliance interface
                    backend = AerSimulator(
                        method='statevector', device="GPU", cusvaer_enable=False, cuStateVec_enable=False,
                        blocking_enable=blocking_enable, blocking_qubits=blocking_qubits,
                        fusion_max_qubit=nfused, precision=self.precision)
                except:  # AerError
                    # use public interface
                    backend = AerSimulator(
                        method='statevector', device="GPU", cuStateVec_enable=False,
                        blocking_enable=blocking_enable, blocking_qubits=blocking_qubits,
                        fusion_max_qubit=nfused, precision=self.precision)
            else:
                raise ValueError(f"need to specify --ngpus for the backend {identifier}")
        elif identifier == "aer-cusv":
            if ngpus >= 1:
                blocking_enable, blocking_qubits = self.get_aer_blocking_setup(ngpus)
                try:
                    # use cuQuantum Appliance interface
                    backend = AerSimulator(
                        method='statevector', device="GPU", cusvaer_enable=False, cuStateVec_enable=True,
                        blocking_enable=blocking_enable, blocking_qubits=blocking_qubits,
                        fusion_max_qubit=nfused, precision=self.precision)
                except:  # AerError
                    # use public interface
                    backend = AerSimulator(
                        method='statevector', device="GPU", cuStateVec_enable=True,
                        blocking_enable=blocking_enable, blocking_qubits=blocking_qubits,
                        fusion_max_qubit=nfused, precision=self.precision)
            else:
                raise ValueError(f"need to specify --ngpus for the backend {identifier}")
        elif identifier == 'aer':
            if ngpus != 0:
                raise ValueError(f"cannot specify --ngpus for the backend {identifier}")
            blocking_enable, blocking_qubits = self.get_aer_blocking_setup()
            try:
                # use cuQuantum Appliance interface
                backend = AerSimulator(
                    method='statevector', device="CPU", max_parallel_threads=ncpu_threads,
                    cusvaer_enable=False, cuStateVec_enable=False,
                    blocking_enable=blocking_enable, blocking_qubits=blocking_qubits,
                    fusion_max_qubit=nfused, precision=self.precision)
            except:  # AerError
                # use public interface
                backend = AerSimulator(
                    method='statevector', device="CPU", max_parallel_threads=ncpu_threads,
                    blocking_enable=blocking_enable, blocking_qubits=blocking_qubits,
                    fusion_max_qubit=nfused, precision=self.precision)
        else:
            backend = None
        return backend

    def get_aer_blocking_setup(self, ngpus=None):
        size = get_mpi_size()  # check if running MPI
        if size > 1:
            blocking_enable = True
            if self.identifier == 'aer':
                blocking_qubits = self.nqubits - int(math.log2(size))
            else:
                blocking_qubits = self.nqubits - int(math.log2(ngpus))
        else:
            # use default
            blocking_enable = False
            blocking_qubits = None
        return blocking_enable, blocking_qubits

    def _need_sync(self):
        ver_str = version('cusvaer')
        ver = [int(num) for num in ver_str.split('.')]
        return ver[0] == 0 and ver[1] <= 2

    def _synchronize(self):
        my_rank = get_mpi_rank()
        ndevices_in_node = cp.cuda.runtime.getDeviceCount()
        # GPU selected in this process
        device_id = my_rank % ndevices_in_node
        cp.cuda.Device(device_id).synchronize()


CusvAer = functools.partial(Qiskit, identifier="cusvaer")
AerCuda = functools.partial(Qiskit, identifier="aer-cuda")
AerCusv = functools.partial(Qiskit, identifier="aer-cusv")
Aer = functools.partial(Qiskit, identifier="aer")
