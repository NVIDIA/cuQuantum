# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import math
import nvtx
import os
import pickle
import random
import time
import cupy as cp

from .backends import createBackend
from .frontends import createFrontend
from ._utils import (
    call_by_root, create_cache, EarlyReturnError, gen_run_env, get_mpi_rank, HashableDict,
    is_running_mpiexec, is_running_mpi, load_benchmark_data, report, reseed,
    save_benchmark_data,
)


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


class BenchCircuitRunner:

    # currently we assume the following subdirectories exist
    required_subdirs = ('circuits', 'data')

    def __init__(self, **kwargs):
        # use default backend config unless users want to overwrite it
        self.backend_config = backend_config = kwargs.pop("backend_config")
        for k in (# generic backend options
                  "ngpus", "ncputhreads", "nshots", "nfused", "precision",
                  # cusvaer options
                  'cusvaer_global_index_bits', 'cusvaer_p2p_device_bits',
                  'cusvaer_data_transfer_buffer_bits', 'cusvaer_comm_plugin_type',
                  'cusvaer_comm_plugin_soname',
                  # cutn options
                  'nhypersamples'):
            v = kwargs.pop(k)
            if k.startswith('cusvaer') or v is not None:
                setattr(self, k, v)
            else:
                setattr(self, k, backend_config['config'][k])

        # To be parsed in run()
        self._benchmarks = kwargs.pop("benchmarks")
        self._nqubits = kwargs.pop("nqubits")

        # other common benchmark args
        self.frontend = kwargs.pop("frontend")
        self.backend = kwargs.pop("backend")
        self.cache_dir = kwargs.pop("cachedir")
        self.nwarmups = kwargs.pop("nwarmups")
        self.nrepeats = kwargs.pop("nrepeats")
        self.new_circ = kwargs.pop("new")
        self.save = True
        assert len(kwargs) == 0, f"unhandled cmdline args: {kwargs}"

        self.full_data = {}
        self.benchmark_data = {}

        # it could be that the cache dirs are not created yet
        call_by_root(functools.partial(create_cache, self.cache_dir, self.required_subdirs))

    def run(self):
        if self._nqubits is None:
            gpu_prop = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            max_n_qubits = math.floor(math.log2(gpu_prop['totalGlobalMem'] / (8 if self.precision == 'single' else 16)))
            nqubits_list = list(range(16, max_n_qubits + 4, 4))
        else:
            nqubits_list = [self._nqubits]

        for benchmark_name in self._benchmarks.keys():
            b = self._benchmarks[benchmark_name]
            benchmark_object = b['benchmark']
            benchmark_config = b['config']
            benchmark_config['precision'] = self.precision  # some frontends may need it

            for nqubits in nqubits_list:
                self.benchmark_name = benchmark_name
                self.benchmark_object = benchmark_object
                self.benchmark_config = benchmark_config
                self.nqubits = nqubits
                self._run()

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
        frontend = createFrontend(self.frontend, self.nqubits, self.benchmark_config)

        dump_only = bool(os.environ.get('CUQUANTUM_BENCHMARKS_DUMP_GATES', False))
        if dump_only:
            # hijack & discard user input
            from .frontends.frontend_dumper import Dumper
            frontend = Dumper(
                self.nqubits,
                {**self.benchmark_config, 'circuit_filename': circuit_filename})
        try:
            if self.new_circ:
                raise ValueError

            # If this circuit has been generated previously, load it
            with open(os.path.join(self.cache_dir, circuit_filename), 'rb') as f:
                gate_sequence = pickle.load(f)
                circuit = frontend.generateCircuit(gate_sequence)
                logger.debug(f'Circuit loaded from {circuit_filename}')

        except:  # Otherwise, generate the circuit and save it
            gate_sequence = self.benchmark_object.generateGatesSequence(self.nqubits, self.benchmark_config)
            circuit = frontend.generateCircuit(gate_sequence)
            def dump():
                with open(os.path.join(self.cache_dir, circuit_filename), 'wb') as f:
                    pickle.dump(gate_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.debug(f'Circuit generated and saved to {circuit_filename}')
            call_by_root(dump)

        if dump_only:
            logger.info("early exiting as the dumper task is completed")
            raise EarlyReturnError

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

        annotation_string = f"p{get_mpi_rank()}_run_"
        # actual timing
        for i in range(self.nrepeats):
            backend.pre_run(circuit, nshots=nshots)

            if self.ngpus > 0:
                start_gpu.record()
            pe1 = time.perf_counter()

            with nvtx.annotate(annotation_string + str(i)):
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
        if self.backend == 'cutn':
            target = os.environ.get('CUTENSORNET_BENCHMARK_TARGET', 'amplitude')
            circuit_filename += f'_{target}'
            if target == 'expectation':
                pauli = random.choices(('I', 'X', 'Y', 'Z'), k=nqubits)
                circuit_filename += f"_{''.join(pauli)}"
        return circuit_filename, target, pauli

    def extract_frontend_version(self):
        if self.frontend == 'qiskit':
            import qiskit
            if hasattr(qiskit, "__version__") and qiskit.__version__ >= "1.0.0":
                version = qiskit.__version__
            else:
                version = qiskit.__qiskit_version__['qiskit-terra']
        elif self.frontend == 'cirq':
            import cirq
            version = cirq.__version__
        elif self.frontend == 'naive':
            from .frontends import frontends
            version = frontends['naive'].version
        elif self.frontend == 'pennylane':
            import pennylane
            version = pennylane.__version__
        elif self.frontend == 'qulacs':
            import qulacs
            version = qulacs.__version__
        else:
            assert False
        return version

    def extract_glue_layer_version(self):
        if self.backend == 'cutn':
            import cuquantum
            glue_ver = f'cuquantum {cuquantum.__version__}'
        else:
            return None
        return glue_ver

    def _run(self):
        reseed(1234)  # TODO: use a global seed?
        measure = self.benchmark_config['measure']

        # try to load existing perf data, if any
        data_filename = f'{self.benchmark_name}.json'
        filepath = f'{self.cache_dir}/data/{data_filename}'
        self.full_data = load_benchmark_data(filepath)

        gpu_device_properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        gpu_name = gpu_device_properties['name'].decode('utf-8').split(' ')[-1]
        num_qubits = str(self.nqubits)
        num_gpus = str(self.ngpus)

        circuit_filename = f'circuits/{self.benchmark_name}_{self.nqubits}'
        if 'unfold' in self.benchmark_config.keys() and self.benchmark_config['unfold']:
            circuit_filename += '_unfold'
        if 'p' in self.benchmark_config.keys():
            p = self.benchmark_config['p']
            circuit_filename += f'_p{p}'
        if measure:
            circuit_filename += '_measure'
        circuit_filename, target, pauli = self._fix_filename_for_cutn(circuit_filename, self.nqubits)
        self.cutn_target = target

        # get circuit
        circuit = self.get_circuit(circuit_filename)

        # get backend
        # TODO: use backend config to simplify this...
        backend = createBackend(
            self.backend, self.ngpus, self.ncputhreads, self.precision,
            nqubits=self.nqubits,
            # cusvaer options
            cusvaer_global_index_bits=self.cusvaer_global_index_bits,
            cusvaer_p2p_device_bits=self.cusvaer_p2p_device_bits,
            cusvaer_data_transfer_buffer_bits=self.cusvaer_data_transfer_buffer_bits,
            cusvaer_comm_plugin_type=self.cusvaer_comm_plugin_type,
            cusvaer_comm_plugin_soname=self.cusvaer_comm_plugin_soname,
            # qiskit and qsim
            nfused=self.nfused,
            # cutn
            nhypersamples=self.nhypersamples,
        )

        # get versions; it's assumed up to this point, the existence of Python modules for
        # both frontend and backend is confirmed
        backend_version = backend.version
        frontend_version = self.extract_frontend_version()
        glue_layer_version = self.extract_glue_layer_version()
        if glue_layer_version is not None:
            ver_str = f'[{self.frontend}-v{frontend_version} | (glue ver: {glue_layer_version}) | {self.backend}-v{backend_version}]'
        else:
            ver_str = f'[{self.frontend}-v{frontend_version} | {self.backend}-v{backend_version}]'

        if self.ngpus == 0:
            logger.info(
                f'* Running {self.benchmark_name} with {self.ncputhreads} CPU threads, and {self.nqubits} qubits {ver_str}:'
            )
        else:
            logger.info(
                f'* Running {self.benchmark_name} with {self.ngpus} GPUs, and {self.nqubits} qubits {ver_str}:'
            )

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
        perf_time, cuda_time, post_time, post_process = self.timer(backend, circuit, self.nshots) # nsamples -> nshots

        # report the result
        run_env = gen_run_env(gpu_device_properties)
        report(perf_time, cuda_time, post_time if post_process else None, self.ngpus,
               run_env, gpu_device_properties, self.benchmark_data)

        # Save the new benchmark data
        out = self.canonicalize_benchmark_data(frontend_version, backend_version, run_env, glue_layer_version)
        save_benchmark_data(
            *out,
            self.full_data, filepath, self.save)

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
                "name": self.frontend,
                "version": frontend_version,
            }),
            'backend': HashableDict({
                "name": self.backend,
                "version": backend_version,
                "ngpus": self.ngpus,
                "ncputhreads": self.ncputhreads,
                "nshots": self.nshots,
                "nfused": self.nfused,
                "precision": self.precision,
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
        if self.backend == "cusvaer":
            sim_config["backend"]["cusvaer_global_index_bits"] = self.cusvaer_global_index_bits
            sim_config["backend"]["cusvaer_p2p_device_bits"] = self.cusvaer_p2p_device_bits
        elif self.backend == "cutn":
            sim_config["backend"]["target"] = self.cutn_target

        sim_config_hash = sim_config.get_hash()
        self.benchmark_data = {**self.benchmark_data, **sim_config}

        return num_qubits, sim_config_hash, self.benchmark_data


class BenchApiRunner:

    supported_cusv_apis = ('apply_matrix', 'apply_generalized_permutation_matrix', 'cusv_sampler', )
    supported_cutn_apis = ('tensor_decompose',)
    supported_apis = supported_cusv_apis + supported_cutn_apis

    # currently we assume the following subdirectories exist
    required_subdirs = ('data',)

    def __init__(self, **kwargs):
        self.benchmark = kwargs.pop("benchmark")
        self.cache_dir = kwargs.pop("cachedir")
        self.args = kwargs  # just hold the entire group of parsed cmdline args, don't unpack all

        # it could be that the cache dirs are not created yet
        call_by_root(functools.partial(create_cache, self.cache_dir, self.required_subdirs))

        # load existing json, if any
        self.data_filename = f"{self.benchmark}.json"
        self.file_path = f'{self.cache_dir}/data/{self.data_filename}'
        self.full_data = load_benchmark_data(self.file_path)

    def run(self):
        # prep
        if self.benchmark not in self.supported_apis:
            raise NotImplementedError(f"only {self.supported_apis} is supported for now")
        gpu_device_properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        benchmark_data = {}

        # time the api
        bench_func = getattr(self, f"_run_{self.benchmark}")
        perf_time, cuda_time = bench_func(benchmark_data)  # update benchmark_data in-place

        # report the result
        run_env = gen_run_env(gpu_device_properties)
        report(perf_time, cuda_time, None, 1,
               run_env, gpu_device_properties, benchmark_data)

        # Save the new benchmark data
        out = self.canonicalize_benchmark_data(run_env, benchmark_data)
        save_benchmark_data(*out, self.full_data, self.file_path)

    def _run_apply_matrix(self, benchmark_data):
        # TODO: It's better to move this method elsewhere, once we support more apis
        from .benchmarks.apply_matrix import test_apply_matrix
        args = self.args
        self.num_qubits = args.pop("nqubits")

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
            benchmark_data=benchmark_data,
        )

    def _run_apply_generalized_permutation_matrix(self, benchmark_data):
        # TODO: It's better to move this method elsewhere, once we support more apis
        from .benchmarks.apply_gen_perm_matrix import test_apply_generalized_permutation_matrix
        args = self.args
        self.num_qubits = args.pop("nqubits")

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

        # create perm_table while keeping args clean for later use
        has_perm = args.pop("has_perm")
        perm_table = args.pop("perm_table")
        if has_perm is False and perm_table is None:
            perm_table = []
        elif perm_table is None:
            # used as a flag to fill perm_table randomly later
            perm_table = bool(has_perm)
        else:
            perm_table = list(perm_table)
        args["perm_table"] = perm_table

        # run
        return test_apply_generalized_permutation_matrix(
            self.num_qubits,
            args["precision"],
            targets,
            controls,
            int(args["adjoint"]),
            args["has_diag"],
            args["precision_diag"],
            args["location_diag"],
            args["perm_table"],
            args["location_perm"],
            args["nwarmups"],
            args["nrepeats"],
            benchmark_data=benchmark_data,
        )

    def _run_cusv_sampler(self, benchmark_data):
        from .benchmarks.cusv_sampler import test_cusv_sampler
        args = self.args
        self.num_qubits = args.pop("nqubits")

        # create bit_ordering while keeping args clean for later use
        nbit_ordering = args.pop("nbit_ordering")
        bit_ordering = args.pop("bit_ordering")
        bit_ordering = tuple(range(nbit_ordering)) if bit_ordering is None else tuple(bit_ordering)
        args["bit_ordering"] = bit_ordering

        # run
        return test_cusv_sampler(
            self.num_qubits,
            args["precision"],
            bit_ordering,
            args["nshots"],
            args["output_order"],
            args["nwarmups"],
            args["nrepeats"],
            benchmark_data=benchmark_data,
        )

    def _run_tensor_decompose(self, benchmark_data):
        from .benchmarks.tensor_decompose import benchmark_tensor_decompose
        args = self.args
        self.num_qubits = 0  # WAR

        # ensure the combination of method/algorithm is meaningful
        if args["method"] == "SVD":
            args["algorithm"] = "gesvd"
        elif args["algorithm"] is not None:
            # algorithm is set, must be doing SVD
            args["method"] = "SVD"

        # run
        return benchmark_tensor_decompose(
            args["expr"],
            tuple(args["shape"]),
            args["precision"],
            args["is_complex"],
            args["method"],
            args["algorithm"],
            args["nwarmups"],
            args["nrepeats"],
            args["check_reference"],
            benchmark_data=benchmark_data,
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
