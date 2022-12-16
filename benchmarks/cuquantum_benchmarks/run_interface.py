# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import logging
import os
import pickle
import random
import time
import platform
import psutil

import cupy as cp

from .backends import createBackend
from .frontends import createFrontend
from ._utils import (get_cpu_name, get_gpu_driver_version, is_running_mpiexec,
                     write_on_rank_0, HashableDict)


def run_interface(
        benchmarks, nqubits_interface, ngpus_interface, ncpu_threads_interface, frontend, backend, nwarmups, nrepeats, nshots_interface,
        nfused_interface, precision_interface, new_circ, save, logger, cache_dir, cusvaer_global_index_bits, cusvaer_p2p_device_bits):

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
                                         save=save,
                                         logger=logger)
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
                                       cusvaer_p2p_device_bits=cusvaer_p2p_device_bits)
            run_specific.run()


class GeneralInterface:

    def __init__(self, frontend, backend, nshots, nfused, precision, new_circ, save, logger):
        self.frontend = frontend
        self.backend = backend
        self.nshots = nshots
        self.nfused = nfused
        self.precision = precision
        #self.append = append
        self.new_circ = new_circ
        self.save = save
        self.logger = logger
        self.full_data = {}


class RunSpecific:

    def __init__(
            self, benchmark_name, benchmark_object, nqubits, ngpus, ncpu_threads, nwarmups, nrepeats, config,
            general_interface, cache_dir, cusvaer_global_index_bits, cusvaer_p2p_device_bits):
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
        self.cusvaer_global_index_bits = cusvaer_global_index_bits
        self.cusvaer_p2p_device_bits = cusvaer_p2p_device_bits

        # currently we assume the following subdirectories exist
        def create_cache():
            for subdir in ('circuits', 'data'):
                path = os.path.join(cache_dir, subdir)
                if not os.path.isdir(path):
                    os.makedirs(path, exist_ok=True)
        write_on_rank_0(create_cache)

    def load_or_generate_circuit(self, circuit_filename):
        try:
            if self.general_interface.new_circ:
                raise ValueError

            # If this circuit has been generated previously, load it
            with open(os.path.join(self.cache_dir, circuit_filename), 'rb') as f:
                gate_sequence = pickle.load(f)
                frontend = createFrontend(self.general_interface.frontend, self.nqubits, self.config)
                circuit = frontend.generateCircuit(gate_sequence)
                self.general_interface.logger.debug(f'Circuit loaded from {circuit_filename}')

        except: # Otherwise, generate the circuit and save it
            gate_sequence = self.benchmark_object.generateGatesSequence(self.nqubits, self.config)
            frontend = createFrontend(self.general_interface.frontend, self.nqubits, self.config)
            circuit = frontend.generateCircuit(gate_sequence)
            def dump():
                with open(os.path.join(self.cache_dir, circuit_filename), 'wb') as f:
                    pickle.dump(gate_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
                    self.general_interface.logger.debug(f'Circuit generated and saved to {circuit_filename}')
            write_on_rank_0(dump)

        return circuit

    def timer(self, backend, circuit, nshots):
        perf_time = 0
        cuda_time = 0
        post_time = 0
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()

        # warm up
        for i in range(self.nwarmups):
            backend.run(circuit, nshots)

        # actual timing
        for i in range(self.nrepeats):
            pe1 = time.perf_counter()
            if self.ngpus > 0:
                start_gpu.record()

            run_dict = backend.run(circuit, nshots)

            if self.ngpus > 0:
                end_gpu.record()
                end_gpu.synchronize()
                cuda_time += cp.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000  # ms->s

            pe2 = time.perf_counter()
            perf_time += pe2 - pe1

            results = run_dict['results']
            post_res = run_dict['post_results']
            run_data = run_dict['run_data']

            for k, v in run_data.items():
                self.benchmark_data[k] = v

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
            version = cuquantum.__version__
        elif self.general_interface.backend == 'cirq':
            import cirq
            version = cirq.__version__
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
        else:
            assert False
        return version

    def run(self):
        measure = self.config['measure']

        data_filename = f'{self.benchmark_name}.json'
        if os.path.exists(f'data/{data_filename}'):
            try:
                with open(f'{self.cache_dir}/data/{data_filename}', 'r') as f:
                    self.general_interface.full_data = json.load(f)
                    self.general_interface.logger.debug(f'Loaded data/{data_filename} as JSON')
            # If the data file does not exist, we'll create it later
            except FileNotFoundError:
                self.general_interface.logger.debug(f'data/{data_filename} not found')
                pass

        gpu_device_properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        gpu_name = gpu_device_properties['name'].decode('utf-8').split(' ')[-1]
        num_qubits = str(self.nqubits)
        num_gpus = str(self.ngpus)

        backend_version = self.extract_backend_version()
        frontend_version = self.extract_frontend_version()

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

        if self.ngpus == 0:
            self.general_interface.logger.info(
                f'* Running {self.benchmark_name} with {self.ncpu_threads} CPU threads, and {self.nqubits} qubits [{self.general_interface.backend}-v{backend_version}]:')
        else:
            self.general_interface.logger.info(
                f'* Running {self.benchmark_name} with {self.ngpus} GPUs, and {self.nqubits} qubits [{self.general_interface.backend}-v{backend_version}]:')

        # get circuit
        circuit = self.load_or_generate_circuit(circuit_filename + '.pickle')

        backend = createBackend(
            self.general_interface.backend, self.ngpus, self.ncpu_threads, self.general_interface.precision, self.general_interface.logger,
            nqubits=self.nqubits,                                      # TODO: backend config
            cusvaer_global_index_bits=self.cusvaer_global_index_bits,  # only cusvaer needs this, TODO: backend config
            cusvaer_p2p_device_bits=self.cusvaer_p2p_device_bits,      # only cusvaer needs this, TODO: backend config
            nfused=self.general_interface.nfused,                      # only qiskit and qsim
        )
        preprocess_data = backend.preprocess_circuit(
            circuit,
            circuit_filename=circuit_filename, target=target, pauli=pauli  # only cutn needs this, TODO: backend config
        )

        for k in preprocess_data.keys():
            self.benchmark_data[k] = preprocess_data[k]

        # run benchmark
        perf_time, cuda_time, post_time, post_process = self.timer(backend, circuit, self.general_interface.nshots) # nsamples -> nshots

        # report the result
        run_env = HashableDict({
            'hostname': platform.node(),
            'cpu_name': get_cpu_name(),
            'gpu_name': gpu_device_properties['name'].decode('utf-8'),
            'gpu_driver_ver': cp.cuda.runtime.driverGetVersion(),
            'gpu_runtime_ver': cp.cuda.runtime.runtimeGetVersion(),
            'nvml_driver_ver': get_gpu_driver_version(),
        })
        self.report(perf_time, cuda_time, post_time, post_process, run_env, gpu_device_properties)

        # Save the new benchmark data
        self.save_benchmark_data(data_filename, frontend_version, backend_version, run_env)

    def report(self, perf_time, cuda_time, post_time, post_process, run_env, gpu_device_properties):
        hostname = run_env['hostname']
        cpu_name = run_env['cpu_name']
        cpu_phy_mem = round(psutil.virtual_memory().total/1000000000, 2)
        cpu_used_mem = round(psutil.virtual_memory().used/1000000000, 2)
        cpu_phy_cores = psutil.cpu_count(logical=False)
        cpu_log_cores = psutil.cpu_count(logical=True)
        cpu_curr_freq = round(psutil.cpu_freq().current, 2)
        cpu_min_freq = psutil.cpu_freq().min
        cpu_max_freq = psutil.cpu_freq().max

        gpu_name = run_env['gpu_name']
        gpu_total_mem = round(gpu_device_properties['totalGlobalMem']/1000000000, 2)
        gpu_clock_rate = round(gpu_device_properties['clockRate']/1000, 2)
        gpu_multiprocessor_num = gpu_device_properties['multiProcessorCount']
        gpu_driver_ver = run_env['gpu_driver_ver']
        gpu_runtime_ver = run_env['gpu_runtime_ver']
        nvml_driver_ver = run_env['nvml_driver_ver']

        self.general_interface.logger.debug(f' - hostname: {hostname}')
        self.general_interface.logger.info(f' - [CPU] Averaged elapsed time: {perf_time:.6f} s')
        if post_process:
            self.general_interface.logger.info(f' - [CPU] Averaged postprocessing Time: {post_time:.6f} s')
            self.benchmark_data['cpu_post_time'] = post_time
        self.general_interface.logger.info(f' - [CPU] Processor type: {cpu_name}')
        self.general_interface.logger.debug(f' - [CPU] Total physical memory: {cpu_phy_mem} GB')
        self.general_interface.logger.debug(f' - [CPU] Total used memory: {cpu_used_mem} GB')
        self.general_interface.logger.debug(f' - [CPU] Number of physical cores: {cpu_phy_cores}, and logical cores: {cpu_log_cores}')
        self.general_interface.logger.debug(f' - [CPU] Frequency current (Mhz): {cpu_curr_freq}, min: {cpu_min_freq}, and max: {cpu_max_freq}')
        self.general_interface.logger.info(' -')
        self.general_interface.logger.info(f' - [GPU] Averaged elapsed time: {cuda_time:.6f} s {"(unused)" if self.ngpus == 0 else ""}')
        self.general_interface.logger.info(f' - [GPU] GPU device name: {gpu_name}')
        self.general_interface.logger.debug(f' - [GPU] Total global memory: {gpu_total_mem} GB')
        self.general_interface.logger.debug(f' - [GPU] Clock frequency (Mhz): {gpu_clock_rate}')
        self.general_interface.logger.debug(f' - [GPU] Multi processor count: {gpu_multiprocessor_num}')
        self.general_interface.logger.debug(f' - [GPU] CUDA driver version: {gpu_driver_ver} ({nvml_driver_ver})')
        self.general_interface.logger.debug(f' - [GPU] CUDA runtime version: {gpu_runtime_ver}')
        self.general_interface.logger.info('\n')

        self.benchmark_data['cpu_time'] = perf_time
        self.benchmark_data['cpu_phy_mem'] = cpu_phy_mem
        self.benchmark_data['cpu_used_mem'] = cpu_used_mem
        self.benchmark_data['cpu_phy_cores'] = cpu_phy_cores
        self.benchmark_data['cpu_log_cores'] = cpu_log_cores
        self.benchmark_data['cpu_current_freq'] = cpu_curr_freq

        self.benchmark_data['gpu_time'] = cuda_time
        self.benchmark_data['gpu_total_mem'] = gpu_total_mem
        self.benchmark_data['gpu_clock_freq'] = gpu_clock_rate
        self.benchmark_data['gpu_multiprocessor_num'] = gpu_multiprocessor_num

    def save_benchmark_data(self, data_filename, frontend_version, backend_version, run_env):
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
                         |_ gpu_name
                         |_ ngpus
                         |_ ncputhreads
                         |_ nshots
                         |_ nfused
                         |_ precision
                         |_ ... (all backend-specific options go here)
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
        try:
            self.general_interface.full_data[num_qubits][sim_config_hash] = self.benchmark_data
        except KeyError:
            if num_qubits not in self.general_interface.full_data:
                self.general_interface.full_data[num_qubits] = {}

            if sim_config_hash not in self.general_interface.full_data[num_qubits]:
                self.general_interface.full_data[num_qubits][sim_config_hash] = {}

            self.general_interface.full_data[num_qubits][sim_config_hash] = self.benchmark_data

        if self.general_interface.save:
            def dump():
                with open(f'{self.cache_dir}/data/{data_filename}', 'w') as f:
                    json.dump(self.general_interface.full_data, f, indent=2)
            write_on_rank_0(dump)
