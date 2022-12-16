# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import functools
import time

import numpy as np
try:
    from qiskit.providers.aer import AerSimulator
    from qiskit import transpile
except ImportError:
    AerSimulator = transpile = None

from .backend import Backend
from .._utils import is_running_mpi


class Qiskit(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, logger, *args, identifier=None, **kwargs):
        self.precision = precision
        self.logger = logger
        self.identifier = identifier
        self.nqubits = kwargs.pop('nqubits')
        self.backend = self.create_aer_backend(identifier, ngpus, ncpu_threads, *args, **kwargs)

    def preprocess_circuit(self, circuit, *args, **kwargs):
        t0 = time.perf_counter()
        self.transpiled_qc = transpile(circuit, self.backend) # (circuit, basis_gates=['u3', 'cx'], backend=self.backend)
        t1 = time.perf_counter()
        time_transpile = t1 - t0
        self.logger.info(f'transpile took {time_transpile} s')
        return {'transpile': time_transpile}

    def run(self, circuit, nshots=1024):
        run_data = {}
        transpiled_qc = self.transpiled_qc
        if nshots > 0:
            results = self.backend.run(transpiled_qc, shots=nshots, memory=True)
        else:
            results = self.backend.run(transpiled_qc, shots=0, memory=True)

        post_res_list = results.result().get_memory()
        post_res_list = [list(i) for i in post_res_list]
        post_res = np.array(post_res_list)
        return {'results': results, 'post_results': post_res, 'run_data': run_data}

    def create_aer_backend(self, identifier, ngpus, ncpu_threads, *args, **kwargs):
        nfused = kwargs.pop('nfused')
        if identifier == 'cusvaer':
            cusvaer_global_index_bits = kwargs.pop('cusvaer_global_index_bits')
            cusvaer_p2p_device_bits = kwargs.pop('cusvaer_p2p_device_bits')

            if ngpus != 1:
                raise ValueError("the cusvaer requires 1 GPU per process (--ngpus 1)")
            try:
                backend = AerSimulator(
                    method='statevector', device="GPU", cusvaer_enable=True, noise_model=None,
                    fusion_max_qubit=nfused,
                    cusvaer_global_index_bits=cusvaer_global_index_bits,
                    cusvaer_p2p_device_bits=cusvaer_p2p_device_bits,
                    precision=self.precision
                    #cusvaer_data_transfer_buffer_bits=26,  # default
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
            raise ValueError(f"the backend {identifier} is not recognized")

        return backend

    def get_aer_blocking_setup(self, ngpus=None):
        MPI = is_running_mpi()
        if MPI:
            blocking_enable = True
            if self.identifier == 'aer':
                comm = MPI.COMM_WORLD
                size = comm.Get_size()
                blocking_qubits = self.nqubits - int(math.log2(size))
            else:
                blocking_qubits = self.nqubits - int(math.log2(ngpus))
        else:
            # use default
            blocking_enable = False
            blocking_qubits = None
        return blocking_enable, blocking_qubits


CusvAer = functools.partial(Qiskit, identifier="cusvaer")
AerCuda = functools.partial(Qiskit, identifier="aer-cuda")
AerCusv = functools.partial(Qiskit, identifier="aer-cusv")
Aer = functools.partial(Qiskit, identifier="aer")
