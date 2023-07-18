# cuquantum-benchmarks

## Installing

After cloning this repository, simply run
```
pip install .
```
to install the benchmark suite.

You can install all optional dependencies via
```
pip install .[all]
```
if running outside of the [cuQuantum Appliance container](https://docs.nvidia.com/cuda/cuquantum/latest/appliance/index.html).

**Note: You may have to build `qsimcirq`, `qiskit-aer`, and `qulacs` GPU support from source if needed.**

Alternatively, you can choose to manage all (required & optional) dependencies yourself via
```
pip install --no-deps .
```
and `pip` would not install any extra package for you.

## Running

After installation, a new command `cuquantum-benchmarks` is installed to your Python environment. You can see the help message via `cuquantum-benchmarks --help`:

```
usage: cuquantum-benchmarks [-h] {circuit,api} ...

=============== NVIDIA cuQuantum Performance Benchmark Suite ===============

positional arguments:
  {circuit,api}
    circuit      benchmark different classes of quantum circuits
    api          benchmark different APIs from cuQuantum's libraries

optional arguments:
  -h, --help     show this help message and exit
```

Starting v0.2.0, we offer subcommands for performing benchmarks at different levels, as shown above. For details, please refer to the help message of each subcommand, ex: `cuquantum-benchmarks circuit --help`.

Alternatively, you can launch the benchmark program via `python -m cuquantum_benchmarks`. This is equivalent to the standalone command, and is useful when, say, `pip` installs this package to the user site-package (so that the `cuquantum-benchmarks` command may not be available without modifying `$PATH`).

For GPU backends, it is preferred that `--ngpus N` is explicitly set. On a multi-GPU system, the first `N` GPUs would be used. To limit which GPUs can be accessed by the CUDA runtime, use the environment variable `CUDA_VISIBLE_DEVICES` following the CUDA documentation.

For backends that support MPI parallelism, it is assumed that `MPI_COMM_WORLD` is the communicator, and that `mpi4py` is installed. You can run the benchmarks as you would normally do to launch MPI processes: `mpiexec -n N cuquantum-benchmarks ...`. It is preferred if you fully specify the problem (explicitly set `--benchmark` & `--nqubits`).

Examples:
- `cuquantum-benchmarks api --benchmark apply_matrix --targets 4,5 --controls 2,3 --nqubits 16`: Apply a random gate matrix controlled by qubits 2 & 3 to qubits 4 & 5 of a 16-qubit statevector using cuStateVec's `apply_matrix()` API
- `cuquantum-benchmarks circuit --frontend qiskit --backend cutn --benchmark qft --nqubits 8 --ngpus 1`: Construct a 8-qubit QFT circuit in Qiskit and run it with cuTensorNet on GPU
- `cuquantum-benchmarks circuit --frontend cirq --backend qsim-mgpu --benchmark qaoa --nqubits 16 --ngpus 2`: Construct a 16-qubit QAOA circuit in Cirq and run it with the (multi-GPU) `qsim-mgpu` backend on 2 GPUs (requires cuQuantum Appliance)
- `mpiexec -n 4 cuquantum-benchmarks circuit --frontend qiskit --backend cusvaer --benchmark quantum_volume --nqubits 32 --ngpus 1 --cusvaer-global-index-bits 1,1 --cusvaer-p2p-device-bits 1`: Construct a 32-qubit Quantum Volume circuit in Qiskit and run it with the (multi-GPU-multi-node) `cusvaer` backend on 2 nodes. Each node runs 2 MPI processes, each of which controls 1 GPU (requires cuQuantum Appliance)

## Known issues

- Due to Qiskit Aer's design, it'd initialize the CUDA contexts for all GPUs installed on the system at import time. While we can defer the import, it might have an impact to the (multi-GPU) system performance when any `aer*` backend is in use. For the time being, we recommend to work around it by limiting the visible devices. For example, `CUDA_VISIBLE_DEVICES=0,1 cuquantum-benchmarks ...` would only use GPU 0 & 1.

## Output data

All the recorded data is stored in the `data` directory as JSON files, and is separated by benchmarks. The data can be accessed by `json_data[nqubits][sim_config_hash]`, where `sim_config_hash` is a hash string for the benchmark setup as determined by `frontend`, `backend`, and `run_env`. This ensures benchmark data is properly recorded once any part of the benchmark setup changes.

It is recommended to loop over all recorded `sim_config_hash` to gather perf data for analysis.

## Environment variables

Currently all environment variables are reserved for internal use only, and are subject to change in the future without notification.

* `CUTENSORNET_DUMP_TN=txt`
* `CUTENSORNET_BENCHMARK_TARGET={amplitude,state_vector,expectation}` (pick one)
* `CUTENSORNET_APPROX_TN_UTILS_PATH`
* `CUQUANTUM_BENCHMARKS_DUMP_GATES`

## Development Overview

The benchmark suite generates a benchmark in a framework-agnostic fashion. Then, it's mapped to each framework's own circuit object. This mapping is done by picking a "frontend". Once a circuit object is generated, it is passed to a "backend" to execute, which may or may not be part of the framework. This componentized design allows for future extension of this benchmark suite to quickly support other quantum computing frameworks.

### Adding New Benchmarks

There are two components to add when adding a new benchmark, namely the benchmark file in the benchmarks directory, and
registering the benchmark in `config.py`.

To add a new benchmark, a new benchmark class that inherits from `Benchmark` needs to be defined. The only function that needs
to be implemented is `generateGatesSequence`. Some optional methods, such as `postProcess`, may be defined if necessary.

The method `generateGatesSequence` takes in an int `nqubits` and a `config` dictionary. The `config` dictionary contains the benchmark-specific configurations that are defined in `config.py`; the output of `generateGatesSequence` should be a list of gate types as tuples.

### Adding New Frontends

There are two components to add when adding a new frontend, namely the frontend file in the frontends directory, and
registering the frontend in `frontends/__init__.py`.

To add a new frontend, a new frontend object needs to inherit from `Frontend` in `frontends/frontend.py`. The only two methods that have to be implemented are `__init__`, which takes the number of qubits, and a `config` dictionary, and `generateCircuit`, which takes a sequence of gates
and the number of ancillas.

The output of `generateCircuit` is a quantum circuit using gates of corresponding frontend.

### Adding New Backends

There are three components to add when adding a new backend, namely the backend file in the backends directory, and
registering the backend in `backends/__init__.py` and `config.py`.

To add a new backend, a new backend object needs to inherit from `Backend` in `backends/backend.py`. The only two methods that have to be implemented are `__init__`, which takes the number of gpus, the number of cpu threads and a `logger`, and `run`, which takes a circuit
and the number of samples. Additionally, `preprocess_circuit` can be implemented if necessary, which takes in a circuit
and can set up configurations to later run the circuit correctly.

The output of `run` is a dictionary that contains `results`, the result of computation, and `run_data`, and performance data
that one would want to record.

The output of `preprocess_circuit` is a dictionary that similarly contains any performance data that one would like to record.
