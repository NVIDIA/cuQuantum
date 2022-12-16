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
if running outside of the [cuQuantum Appliance container](https://docs.nvidia.com/cuda/cuquantum/appliance/index.html). You may have to build `qsimcirq` and `qiskit-aer` GPU support from source if needed.

Alternatively, you can choose to manage all (required & optional) dependencies yourself via
```
pip install --no-deps .
```
and `pip` would not install any extra package for you.

## Running

After installation, a new command `cuquantum-benchmarks` is installed to your Python environment. You can see the help message via `cuquantum-benchmarks --help`:

```
usage: cuquantum-benchmarks [-h] --frontend {cirq,qiskit} --backend
                            {aer,aer-cuda,aer-cusv,cusvaer,cirq,cutn,qsim,qsim-cuda,qsim-cusv,qsim-mgpu}
                            [--benchmark {qft,iqft,ghz,simon,hidden_shift,qaoa,qpe,quantum_volume,random,all}] [--new] [--nqubits NQUBITS]
                            [--nwarmups NWARMUPS] [--nrepeats NREPEATS] [--cachedir CACHEDIR] [--verbose] [--ngpus NGPUS]
                            [--ncputhreads NCPUTHREADS] [--nshots NSHOTS] [--nfused NFUSED] [--precision {single,double}]
                            [--cusvaer-global-index-bits [CUSVAER_GLOBAL_INDEX_BITS]] [--cusvaer-p2p-device-bits [CUSVAER_P2P_DEVICE_BITS]]

============= NVIDIA cuQuantum Circuit Performance Benchmark Suite =============

Supported Backends:

  - aer: runs Qiskit Aer's CPU backend
  - aer-cuda: runs the native Qiskit Aer GPU backend
  - aer-cusv: runs Qiskit Aer's cuStateVec integration
  - cusvaer: runs the *multi-GPU, multi-node* custom Qiskit Aer GPU backend, only
    available in the cuQuantum Appliance container
  - cirq: runs Cirq's native CPU backend (cirq.Simulator)
  - cutn: runs cuTensorNet by constructing the tensor network corresponding to the
    benchmark circuit (through cuquantum.CircuitToEinsum)
  - qsim: runs qsim's CPU backend
  - qsim-cuda: runs the native qsim GPU backend
  - qsim-cusv: runs qsim's cuStateVec integration
  - qsim-mgpu: runs the *multi-GPU* (single-node) custom qsim GPU backend, only
    available in the cuQuantum Appliance container

================================================================================

optional arguments:
  -h, --help            show this help message and exit
  --frontend {cirq,qiskit}
                        set the simulator frontend (default: None)
  --backend {aer,aer-cuda,aer-cusv,cusvaer,cirq,cutn,qsim,qsim-cuda,qsim-cusv,qsim-mgpu}
                        set the simulator backend that is compatible with the frontend (default: None)
  --benchmark {qft,iqft,ghz,simon,hidden_shift,qaoa,qpe,quantum_volume,random,all}
                        pick the circuit to benchmark (default: all)
  --new                 create a new circuit rather than use existing circuit (default: False)
  --nqubits NQUBITS     set the number of qubits for each benchmark circuit (default: None)
  --nwarmups NWARMUPS   set the number of warm-up runs for each benchmark (default: 3)
  --nrepeats NREPEATS   set the number of repetitive runs for each benchmark (default: 10)
  --cachedir CACHEDIR   set the directory to cache generated data (default: .)
  --verbose             output extra information during benchmarking (default: False)

backend-specific options:
  each backend has its own default config, see cuquantum_benchmarks/config.py for detail

  --ngpus NGPUS         set the number of GPUs to use (default: None)
  --ncputhreads NCPUTHREADS
                        set the number of CPU threads to use (default: None)
  --nshots NSHOTS       set the number of shots for quantum state measurement (default: None)
  --nfused NFUSED       set the maximum number of fused qubits for gate matrix fusion (default: None)
  --precision {single,double}
                        set the floating-point precision (default: None)
  --cusvaer-global-index-bits [CUSVAER_GLOBAL_INDEX_BITS]
                        set the global index bits to represent the inter-node network structure, refer to the cusvaer backend documentation
                        for further detail. If not followed by any argument, the default (empty sequence) is used; otherwise, the argument
                        should be a comma-separated string. Setting this option is mandatory for the cusvaer backend and an error otherwise
                        (default: -1)
  --cusvaer-p2p-device-bits [CUSVAER_P2P_DEVICE_BITS]
                        set the number of p2p device bits, refer to the cusvaer backend documentation for further detail. If not followed by
                        any argument, the default (0) is used. Setting this option is mandatory for the cusvaer backend and an error
                        otherwise (default: -1)
```

Alternatively, you can launch the benchmark program via `python -m cuquantum_benchmarks`. This is equivalent to the standalone command, and is useful when, say, `pip` installs this package to the user site-package (so that the `cuquantum-benchmarks` command may not be available without modifying `$PATH`).

For GPU backends, it is preferred that `--ngpus` is explicitly set.

For backends that support MPI parallelism, it is assumed that `MPI_COMM_WORLD` is the communicator, and that `mpi4py` is installed. You can run the benchmarks as you would normally do to launch MPI processes: `mpiexec -n N cuquantum-benchmarks ...`. It is preferred if you fully specify the problem (explicitly set `--benchmark` & `--nqubits`).

Examples:
- `cuquantum-benchmarks --frontend qiskit --backend cutn --benchmark qft --nqubits 8 --ngpus 1`: Construct a 8-qubit QFT circuit in Qiskit and run it with cuTensorNet on GPU
- `cuquantum-benchmarks --frontend cirq --backend qsim-mgpu --benchmark qaoa --nqubits 16 --ngpus 2`: Construct a 16-qubit QAOA circuit in Cirq and run it with the (multi-GPU) `qsim-mgpu` backend on 2 GPUs (requires cuQuantum Appliance)
- `mpiexec -n 4 cuquantum-benchmarks --frontend qiskit --backend cusvaer --benchmark quantum_volume --nqubits 32 --ngpus 1 --cusvaer-global-index-bits 2,2 --cusvaer-p2p-device-bits 2`: Construct a 32-qubit Quantum Volume circuit in Qiskit and run it with the (multi-GPU-multi-node) `cusvaer` backend on 2 nodes, each with 2 GPUs (requires cuQuantum Appliance)

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

## Output data

All the recorded data is stored in the `data` directory as JSON files, and is separated by benchmarks. The data can be accessed by `json_data[nqubits][sim_config_hash]`, where `sim_config_hash` is a hash string for the benchmark setup as determined by `frontend`, `backend`, and `run_env`. This ensures benchmark data is properly recorded once any part of the benchmark setup changes.

It is recommended to loop over all recorded `sim_config_hash` to gather perf data for analysis.

## Environment variables

Currently all environment variables are reserved for internal use only, and are subject to change in the future without notification.

* `CUTENSORNET_DUMP_TN=txt`
* `CUTENSORNET_BENCHMARK_TARGET={amplitude,state_vector,expectation}` (pick one)
