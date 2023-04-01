# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import os
import pickle
import random
import time

import cupy as cp

from .backends import createBackend
from .frontends import createFrontend
from ._utils import (call_by_root, gen_run_env, HashableDict, is_running_mpiexec,
                     load_benchmark_data, report, save_benchmark_data, reseed,
                     is_running_mpi)


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


def run_interface(
        benchmarks, nqubits_interface, ngpus_interface, ncpu_threads_interface, frontend, backend, nwarmups, nrepeats, nshots_interface,
        nfused_interface, precision_interface, new_circ, save, cache_dir,
        cusvaer_global_index_bits, cusvaer_p2p_device_bits, cusvaer_data_transfer_buffer_bits, cusvaer_comm_plugin_type, cusvaer_comm_plugin_soname):

    reseed(1234)  # TODO: use a global seed?
    backend, backend_config = backend  # unpack
    ngpus = ngpus_interface if ngpus_interface is not None else backend_config['config']['ngpus']
    ncpu_threads = ncpu_threads_interface if ncpu_threads_interface is not None else backend_config['config']['ncputhreads']
    nshots = nshots_interface if nshots_interface is not None else backend_config['config']['nshots']
    nfused = nfused_interface if nfused_interface is not None else backend_config['config']['nfused']
    precision = precision_interface if precision_interface is not None else backend_config['config']['precision']

    general_interface = GeneralInterface(frontend=frontend,
                                         backend=backend,
                                         nshots=nshots,
                                         nfused=nfused,
                                         precision=precision,
                                         #append=append,
                                         new_circ=new_circ,
                                         save=save)

    for benchmark_name in benchmarks.keys(): # Iterate over diferent benchmarks
        benchmark = benchmarks[benchmark_name]

        gpu_device_properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        gpu_name = gpu_device_properties['name'].decode('utf-8').split(' ')[-1]
        if gpu_name not in benchmark['nqubits']:
            # Use the default config for this benchmark if there is no GPU-specific config
            gpu_name = 'default'
        nqubits_list = [nqubits_interface] if nqubits_interface else benchmark['nqubits'][gpu_name]

        benchmark_object = benchmark['benchmark']
        config = benchmark['config']
        config['precision'] = precision  # WAR

        for nqubits in nqubits_list: # Iterate over diferent number of qubits
            run_specific = RunSpecific(benchmark_name=benchmark_name,
                                       benchmark_object=benchmark_object,
                                       nqubits=nqubits,
                                       ngpus=ngpus,
                                       ncpu_threads=ncpu_threads,
                                       nwarmups=nwarmups,
                                       nrepeats=nrepeats,
                                       config=config,
                                       general_interface=general_interface,
                                       cache_dir=cache_dir,
                                       cusvaer_global_index_bits=cusvaer_global_index_bits,
                                       cusvaer_p2p_device_bits=cusvaer_p2p_device_bits,
                                       cusvaer_data_transfer_buffer_bits=cusvaer_data_transfer_buffer_bits,
                                       cusvaer_comm_plugin_type=cusvaer_comm_plugin_type,
                                       cusvaer_comm_plugin_soname=cusvaer_comm_plugin_soname)
            run_specific.run()


class GeneralInterface:

    def __init__(self, frontend, backend, nshots, nfused, precision, new_circ, save):
        self.frontend = frontend
        self.backend = backend
        self.nshots = nshots
        self.nfused = nfused
        self.precision = precision
        #self.append = append
        self.new_circ = new_circ
        self.save = save
        self.full_data = {}


class RunSpecific:

    def __init__(
            self, benchmark_name, benchmark_object, nqubits, ngpus, ncpu_threads, nwarmups, nrepeats, config,
            general_interface, cache_dir,
            cusvaer_global_index_bits, cusvaer_p2p_device_bits, cusvaer_data_transfer_buffer_bits,
            cusvaer_comm_plugin_type, cusvaer_comm_plugin_soname):
        self.benchmark_name = benchmark_name
        self.benchmark_object = benchmark_object
        self.nqubits = nqubits
        self.ngpus = ngpus
        self.ncpu_threads = ncpu_threads
        self.nwarmups=nwarmups
        self.nrepeats=nrepeats
        self.config = config
        self.general_interface = general_interface
        self.benchmark_data = {}
        self.cache_dir = cache_dir
        # cusvaer options
        self.cusvaer_global_index_bits = cusvaer_global_index_bits
        self.cusvaer_p2p_device_bits = cusvaer_p2p_device_bits
        self.cusvaer_data_transfer_buffer_bits = cusvaer_data_transfer_buffer_bits
        self.cusvaer_comm_plugin_type = cusvaer_comm_plugin_type
        self.cusvaer_comm_plugin_soname = cusvaer_comm_plugin_soname

        # currently we assume the following subdirectories exist
        self.required_subdirs = ('circuits', 'data')

    def _load_or_generate_circuit(self, circuit_filename):
        # We need a mechanism to ensure any incompatible gate_sequence generated
        # and cached from the previous releases is invalidated. We do so by
        # assigning a version number gate_seq_ver for the gate sequence and
        # encoding it in the pickle filename.
        #
        # v0.1.0: the gate_sequence is a list of size-2 lists.
        # v0.2.0: the gate_sequence is a list of Gate objects. gate_seq_ver = 1.
        gate_seq_ver = 1

        circuit_filename += f"_v{gate_seq_ver}.pickle"
        frontend = createFrontend(self.general_interface.frontend, self.nqubits, self.config)
        try:
            if self.general_interface.new_circ:
                raise ValueError

            # If this circuit has been generated previously, load it
            with open(os.path.join(self.cache_dir, circuit_filename), 'rb') as f:
                gate_sequence = pickle.load(f)
                circuit = frontend.generateCircuit(gate_sequence)
                logger.debug(f'Circuit loaded from {circuit_filename}')

        except:  # Otherwise, generate the circuit and save it
            gate_sequence = self.benchmark_object.generateGatesSequence(self.nqubits, self.config)
            circuit = frontend.generateCircuit(gate_sequence)
            def dump():
                with open(os.path.join(self.cache_dir, circuit_filename), 'wb') as f:
                    pickle.dump(gate_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.debug(f'Circuit generated and saved to {circuit_filename}')
            call_by_root(dump)

        return circuit

    def get_circuit(self, circuit_filename):
        # This method ensures only the root process is responsible to generate/broadcast the circuit
        # so that all processes see the same circuit.
        MPI = is_running_mpi()
        circuit = call_by_root(functools.partial(self._load_or_generate_circuit, circuit_filename))
        if MPI:
            comm = MPI.COMM_WORLD
            circuit = comm.bcast(circuit)
        return circuit

    def timer(self, backend, circuit, nshots):
        perf_time = 0
        cuda_time = 0
        post_time = 0
        if self.ngpus > 0:
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()

        # warm up
        for i in range(self.nwarmups):
            backend.pre_run(circuit, nshots=nshots)
            backend.run(circuit, nshots)

        # actual timing
        for i in range(self.nrepeats):
            backend.pre_run(circuit, nshots=nshots)

            if self.ngpus > 0:
                start_gpu.record()
            pe1 = time.perf_counter()

            run_dict = backend.run(circuit, nshots)

            pe2 = time.perf_counter()
            if self.ngpus > 0:
                end_gpu.record()

            perf_time += pe2 - pe1
            if self.ngpus > 0:
                end_gpu.synchronize()
                cuda_time += cp.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000  # ms->s

            # TODO: remove results?
            results = run_dict['results']
            post_res = run_dict['post_results']
            run_data = run_dict['run_data']

            for k, v in run_data.items():
                self.benchmark_data[k] = v

            pe2 = time.perf_counter()
            post_process = self.benchmark_object.postProcess(self.nqubits, post_res)
            pe3 = time.perf_counter()
            post_time += pe3 - pe2

        return perf_time / self.nrepeats, cuda_time / self.nrepeats, post_time / self.nrepeats, post_process

    def _fix_filename_for_cutn(self, circuit_filename, nqubits):
        target = pauli = None
        if self.general_interface.backend == 'cutn':
            target = os.environ.get('CUTENSORNET_BENCHMARK_TARGET', 'amplitude')
            circuit_filename += f'_{target}'
            if target == 'expectation':
                pauli = random.choices(('I', 'X', 'Y', 'Z'), k=nqubits)
                circuit_filename += f"_{''.join(pauli)}"
        return circuit_filename, target, pauli

    def extract_backend_version(self):
        if 'aer' in self.general_interface.backend:
            import qiskit
            version = qiskit.__qiskit_version__['qiskit-aer']
        elif 'qsim' in self.general_interface.backend:
            import qsimcirq
            version = qsimcirq.__version__
        elif self.general_interface.backend == 'cutn':
            import cuquantum
            version = cuquantum.cutensornet.get_version()
        elif self.general_interface.backend == 'cirq':
            import cirq
            version = cirq.__version__
        elif self.general_interface.backend == 'naive':
            from .backends import backends
            version = backends['naive'].version
        elif self.general_interface.backend == 'pennylane':
            import pennylane
            version = pennylane.__version__
        elif self.general_interface.backend == 'pennylane-lightning-gpu':
            import pennylane_lightning_gpu
            version = pennylane_lightning_gpu.__version__
        elif self.general_interface.backend == 'pennylane-lightning-qubit':
            import pennylane_lightning
            version = pennylane_lightning.__version__
        elif self.general_interface.backend == 'pennylane-lightning-kokkos':
            import pennylane_lightning_kokkos
            version = pennylane_lightning_kokkos.__version__
        elif self.general_interface.backend in ('qulacs-gpu', 'qulacs-cpu'):
            import qulacs
            version = qulacs.__version__
        else:
            assert False
        return version

    def extract_frontend_version(self):
        if self.general_interface.frontend == 'qiskit':
            import qiskit
            version = qiskit.__qiskit_version__['qiskit-terra']
        elif self.general_interface.frontend == 'cirq':
            import cirq
            version = cirq.__version__
        elif self.general_interface.frontend == 'naive':
            from .frontends import frontends
            version = frontends['naive'].version
        elif self.general_interface.frontend == 'pennylane':
            import pennylane
            version = pennylane.__version__
        elif self.general_interface.frontend == 'qulacs':
            import qulacs
            version = qulacs.__version__
        else:
            assert False
        return version

    def extract_glue_layer_version(self):
        if self.general_interface.backend == 'cutn':
            import cuquantum
            glue_ver = f'cuquantum {cuquantum.__version__}'
        else:
            return None
        return glue_ver

    def run(self):
        measure = self.config['measure']

        # try to load existing perf data, if any
        data_filename = f'{self.benchmark_name}.json'
        filepath = f'{self.cache_dir}/data/{data_filename}'
        self.general_interface.full_data = load_benchmark_data(
            filepath, self.cache_dir, self.required_subdirs)

        gpu_device_properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        gpu_name = gpu_device_properties['name'].decode('utf-8').split(' ')[-1]
        num_qubits = str(self.nqubits)
        num_gpus = str(self.ngpus)

        # FIXME: this is buggy (no early return)
        # try:
        #     if (self.general_interface.append
        #             and num_gpus in self.general_interface.full_data[num_qubits][self.general_interface.frontend+'-v'+frontend_version][self.general_interface.backend+'-v'+backend_version][gpu_name]):
        #         self.general_interface.logger.info(
        #             f'Skipping {self.benchmark_name} with {self.nqubits} qubits and {self.ngpus} GPUs [{self.general_interface.backend}-v{backend_version}]')
        # except KeyError:
        #     # KeyError means this configuration is not currently benchmarked, so we can continue running
        #     self.general_interface.logger.debug('Benchmark configuration not found in existing data')
        #     pass

        circuit_filename = f'circuits/{self.benchmark_name}_{self.nqubits}'

        if 'unfold' in self.config.keys() and self.config['unfold']:
            circuit_filename += '_unfold'
        if 'p' in self.config.keys():
            p = self.config['p']
            circuit_filename += f'_p{p}'
        if measure:
            circuit_filename += '_measure'
        circuit_filename, target, pauli = self._fix_filename_for_cutn(circuit_filename, self.nqubits)
        self.general_interface.cutn_target = target

        # get circuit
        circuit = self.get_circuit(circuit_filename)

        # get backend
        backend = createBackend(
            self.general_interface.backend, self.ngpus, self.ncpu_threads, self.general_interface.precision,
            nqubits=self.nqubits,                                      # TODO: backend config
            cusvaer_global_index_bits=self.cusvaer_global_index_bits,  # cusvaer options
            cusvaer_p2p_device_bits=self.cusvaer_p2p_device_bits,
            cusvaer_data_transfer_buffer_bits=self.cusvaer_data_transfer_buffer_bits,
            cusvaer_comm_plugin_type=self.cusvaer_comm_plugin_type,
            cusvaer_comm_plugin_soname=self.cusvaer_comm_plugin_soname,
            nfused=self.general_interface.nfused,                      # only qiskit and qsim
        )

        # get versions; it's assumed up to this point, the existence of Python modules for
        # both frontend and backend is confirmed
        backend_version = self.extract_backend_version()
        frontend_version = self.extract_frontend_version()
        glue_layer_version = self.extract_glue_layer_version()

        if self.ngpus == 0:
            logger.info(
                f'* Running {self.benchmark_name} with {self.ncpu_threads} CPU threads, and {self.nqubits} qubits [{self.general_interface.backend}-v{backend_version}]:')
        else:
            logger.info(
                f'* Running {self.benchmark_name} with {self.ngpus} GPUs, and {self.nqubits} qubits [{self.general_interface.backend}-v{backend_version}]:')

        preprocess_data = backend.preprocess_circuit(
            circuit,
            # only cutn needs these, TODO: backend config
            circuit_filename=os.path.join(self.cache_dir, circuit_filename),
            target=target,
            pauli=pauli
        )

        for k in preprocess_data.keys():
            self.benchmark_data[k] = preprocess_data[k]

        # run benchmark
        perf_time, cuda_time, post_time, post_process = self.timer(backend, circuit, self.general_interface.nshots) # nsamples -> nshots

        # report the result
        run_env = gen_run_env(gpu_device_properties)
        report(perf_time, cuda_time, post_time if post_process else None, self.ngpus,
               run_env, gpu_device_properties, self.benchmark_data)

        # Save the new benchmark data
        out = self.canonicalize_benchmark_data(frontend_version, backend_version, run_env, glue_layer_version)
        save_benchmark_data(
            *out,
            self.general_interface.full_data, filepath, self.general_interface.save)

    def canonicalize_benchmark_data(self, frontend_version, backend_version, run_env, glue_layer_version):
        """
        json scheme: this is designed such that if any item in sim_config changes, the
        benchmark data would be appended, not overwriting.

        benchmark
         |_ num_qubits
             |_ sim_config_hash ( = hash string of sim_config )
                 |_ benchmark_data
                     |_ frontend (part of sim_config)
                         |_ name
                         |_ version
                     |_ backend (part of sim_config)
                         |_ name
                         |_ version
                         |_ ngpus
                         |_ ncputhreads
                         |_ nshots
                         |_ nfused
                         |_ precision
                         |_ ... (all backend-specific options go here)
                     |_ glue_layer (part of sim_config)
                         |_ name
                         |_ version
                     |_ run_env (part of sim_config)
                         |_ hostname
                         |_ cpu_name
                         |_ gpu_name
                         |_ gpu_driver_ver
                         |_ gpu_runtime_ver
                         |_ nvml_driver_ver
                     |_ cpu_time
                     |_ gpu_time
                     |_ ... (other timings, env info, ...)
        """
        # TODO: consider recording cuquantum-benchmarks version?
        # TODO: alternatively, version each individual benchmark and record it?

        num_qubits = str(self.nqubits)

        sim_config = HashableDict({
            'frontend': HashableDict({
                "name": self.general_interface.frontend,
                "version": frontend_version,
            }),
            'backend': HashableDict({
                "name": self.general_interface.backend,
                "version": backend_version,
                "ngpus": self.ngpus,
                "ncputhreads": self.ncpu_threads,
                "nshots": self.general_interface.nshots,
                "nfused": self.general_interface.nfused,
                "precision": self.general_interface.precision,
                "with_mpi": is_running_mpiexec(),
            }),
            'glue_layer': HashableDict({
                "name": None,
                "version": glue_layer_version,
                }),
            'run_env': run_env,
        })

        # frontend-specific options
        # TODO: record "measure"?

        # backend-specific options
        if self.general_interface.backend == "cusvaer":
            sim_config["backend"]["cusvaer_global_index_bits"] = self.cusvaer_global_index_bits
            sim_config["backend"]["cusvaer_p2p_device_bits"] = self.cusvaer_p2p_device_bits
        elif self.general_interface.backend == "cutn":
            sim_config["backend"]["target"] = self.general_interface.cutn_target

        sim_config_hash = sim_config.get_hash()
        self.benchmark_data = {**self.benchmark_data, **sim_config}

        return num_qubits, sim_config_hash, self.benchmark_data


class BenchApiRunner:

    supported_cusv_apis = ('apply_matrix',)
    supported_cutn_apis = ()
    supported_apis = supported_cusv_apis + supported_cutn_apis

    def __init__(self, **kwargs):
        self.num_qubits = kwargs.pop("nqubits")
        self.benchmark = kwargs.pop("benchmark")
        self.cache_dir = kwargs.pop("cachedir")
        kwargs.pop("verbose")  # don't care
        self.args = kwargs  # just hold the entire group of parsed cmdline args, don't unpack all

        # currently we assume the following subdirectories exist
        self.required_subdirs = ('data',)

        # load existing json, if any
        self.data_filename = f"{self.benchmark}.json"
        self.file_path = f'{self.cache_dir}/data/{self.data_filename}'
        self.full_data = load_benchmark_data(
            self.file_path, self.cache_dir, self.required_subdirs)

    def run(self):
        # prep
        if self.benchmark not in self.supported_apis:
            raise NotImplementedError(f"only {self.supported_apis} is supported for now")
        gpu_device_properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        benchmark_data = {}  # dummy

        # time the api
        perf_time, cuda_time = self._run_apply_matrix()

        # report the result
        run_env = gen_run_env(gpu_device_properties)
        report(perf_time, cuda_time, None, 1,
               run_env, gpu_device_properties, benchmark_data)

        # Save the new benchmark data
        out = self.canonicalize_benchmark_data(run_env, benchmark_data)
        save_benchmark_data(*out, self.full_data, self.file_path)

    def _run_apply_matrix(self):
        # TODO: It's better to move this method elsewhere, once we support more apis
        from .benchmarks.apply_matrix import test_apply_matrix
        args = self.args

        # create targets while keeping args clean for later use
        ntargets = args.pop("ntargets")
        targets = args.pop("targets")
        targets = tuple(range(ntargets)) if targets is None else tuple(targets)
        args["targets"] = targets

        # create controls while keeping args clean for later use
        ncontrols = args.pop("ncontrols")
        controls = args.pop("controls")
        if controls is None and ncontrols is None:
            controls = ()
        elif controls is None:
            controls = tuple(range(ncontrols))
        else:
            controls = tuple(controls)
        args["controls"] = controls

        # run
        return test_apply_matrix(
            self.num_qubits,
            targets,
            controls,
            args["precision"],
            args["precision"],  # TODO: allow different mat precision?
            args["layout"],
            int(args["adjoint"]),
            args["nwarmups"],
            args["nrepeats"],
            args["location"],
            flush_l2=args["flush_cache"],
        )

    def canonicalize_benchmark_data(self, run_env, benchmark_data):
        """
        json scheme: this is designed such that if any item in sim_config changes, the
        benchmark data would be appended, not overwriting.

        benchmark
         |_ num_qubits
             |_ sim_config_hash ( = hash string of sim_config )
                 |_ benchmark_data
                     |_ api (part of sim_config)
                         |_ name
                         |_ cuqnt_py_ver
                         |_ lib_ver
                         |_ precision
                         |_ ... (all api-specific options go here)
                     |_ run_env (part of sim_config)
                         |_ hostname
                         |_ cpu_name
                         |_ gpu_name
                         |_ gpu_driver_ver
                         |_ gpu_runtime_ver
                         |_ nvml_driver_ver
                     |_ cpu_time
                     |_ gpu_time
                     |_ ... (other timings, env info, ...)
        """
        # TODO: consider recording cuquantum-benchmarks version?
        from cuquantum import __version__ as cuqnt_py_ver
        num_qubits = str(self.num_qubits)
        benchmark = self.benchmark

        if benchmark in self.supported_cusv_apis:
            from cuquantum import custatevec as lib
        elif benchmark in self.supported_cutn_apis:
            from cuquantum import cutensornet as lib
        else:
            assert False

        # Note: be mindful that we unpack self.args here, as it's designed to be
        # sensitive to any change in the cmdline options.
        sim_config = HashableDict({
            "api": HashableDict({**{
                "name": benchmark,
                "cuqnt_py_ver": cuqnt_py_ver,
                "lib_ver": lib.get_version(),
            }, **self.args}),
            'run_env': run_env,
        })

        # TODO: remember to record cutn_target once we support it
        #elif self.args.backend == "cutn":
        #    sim_config["backend"]["target"] = self.args.cutn_target

        sim_config_hash = sim_config.get_hash()
        benchmark_data = {**benchmark_data, **sim_config}

        return num_qubits, sim_config_hash, benchmark_data
