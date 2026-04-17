# Release Notes

## nv-quantum-benchmarks v0.6.1

* New features & enhancements:

  * Enable `--reuse` flag (enabled by default) for cudaq. It will reuse the JIT-compiled circuit from warmup runs for benchmarking, so the JIT compilation time will not be measured as part of the execution time. Use the `--no-reuse` flag to disable this behavior.
  * Add a default seed for pauli string for `--compute-mode expectation`.


## nv-quantum-benchmarks v0.6.0

* New features & enhancements:

  * Set default cudaq simulator to CPU (`qpp-cpu`).
  * Explicitly set default fuse sizes for cudaq to `None`, therefore cuquantum chooses the best fuse size based on the architecture.
  * Update the other backends than cudaq to auto select fuse size to the best of their ability. The only backend that can not do that right now is Qsim and the related family.
  * Installing the benchmarking repository inside the CUDA Quantum container is not as simple as `pip install .`, documenting it in the README as a point of reference.
  * Fix the code to also run on CPU-only containers.

*Compatibility notes*:

* Support Pennylane version 0.42.0. Fix ControlledQubitUnitary syntax for the new version. Remove the support of very old versions <0.33.0. Support MPI to run with multi-GPU for lightning-kokkos and lightning-gpu.


## nv-quantum-benchmarks v0.5.0

* New features & enhancements:

  * Tool Renamed: The suite formerly known as `cuquantum-benchmarks` is now renamed to `nv-quantum-benchmarks`.

  * CUDA-Q (cudaq) Support: CUDA-Q (cudaq) is now available as both a frontend and backend. Newly supported frontend and backends include:

    - `--frontend`: cudaq
    - `--backend`: cudaq-cusv, cudaq-mgpu, cudaq-cpu

  * Expanded Compute Modes: The `--compute-mode` argument is now available for all backends. Modes include:

    - amplitude
    - statevector
    - sampling
    - expectation

    Note: Not all backends support every compute mode, and each backend has its own default.

  * Enhanced Pauli String Options: New arguments for customizing Pauli strings in your benchmarks:

    - `--pauli-string`: Directly specify the desired Pauli string.
    - `--pauli-seed`: Set a seed for generating random Pauli strings.
    - `--pauli-identity-fraction`: Control the fraction of identity operators when creating random Pauli strings.

  * QAOA Benchmark Update: The Quantum Approximate Optimization Algorithm (QAOA) benchmark now selects gammas and betas randomly for more diverse tests.

  * Internal Improvements: Extensive code refactoring and cleanup for better maintainability and extensibility.


## cuquantum-benchmarks v0.4.0

* New features & enhancements:

  * Add `--compute-target` argument for cutn backend.
  * Make dependency on cuquantum-python optional.

* Bugs fixed:

  * Fixes Pennylane kokkos > 0.33.0.
  * Fix the wrong implementation for Qiskit version number checking.


## cuquantum-benchmarks v0.3.0

* New features & enhancements:

  * Add more API-level benchmarks. See `cuquantum-benchmarks api --help` and e.g. `cuquantum-benchmarks api --benchmark tensor_decompose --help` for more details.

    - `apply_generalized_permutation_matrix`
    - `cusv_sampler`
    - `tensor_decompose`

  * Improvements to the `cutn` backend:

    - Support `--nhypersamples` CLI option to control the number of hypersamples.
    - The contraction optimizer outcome is printed if the benchmark is run with the verbose mode (`-v`).

  * All benchmarks are now covered by the NVTX ranges for ease of profiling.

    - This requires the `nvtx` Python package, installable via `pip install nvtx` or `conda install -c conda-forge nvtx`.

  * The help messages for CLI options are now clearer.

    - As the benchmark suite grows, some CLI options are dynamically generated/recognized (based on the top-level inputs, such as "api"/"circuit"). Be sure to do the `--help` query for the selected benchmark and/or frontend-backend.

  * Internal code refactoring and cleanup.

* Bugs fixed:

  * Fix a bug that the subdirectories in the cache dir might not be created correctly.
  * Fix a bug in our Qiskit circuit generator.
  * Various fixes for the `cusvaer` backend:

    - Fix an unexpected device index is set.
    - Fix unnecessary synchronization.


## cuquantum-benchmarks v0.2.0

* Breaking changes:

  * To run the circuit-level benchmarks, a `circuit` subcommand needs to be added. For example,

    ```bash
    cuquantum-benchmarks --frontend cirq --backend qsim --benchmark qft --nqubits 16  # v0.1.0
    ```

    becomes

    ```bash
    cuquantum-benchmarks circuit --frontend cirq --backend qsim --benchmark qft --nqubits 16  # v0.2.0
    ```

* New features & enhancements:

  * Introduce API-level benchmarks. Currently, `cuquantum.custatevec.apply_matrix()` is supported. See `cuquantum-benchmarks api --help` for more details.

    - The existing circuit-level benchmarks should be run with the addition of the subcommand `circuit`, as shown above.

  * Support for more simulator frontends & backends, including Pennylane and Qulacs.
  * Support single- and multi-process Slurm jobs.
  * Add more command line options to support the cusvaer backend.
  * Include more system & runtime information in the generated json logs.
  * Improve the internal timer implementation.
  * Improve the overall code quality, such as the logging system and the circuit generator.

    - The cached circuit data in `/path/to/your/cachedir/circuit/` would be automatically invalidated.

* Bugs fixed:

  * Work around a bug in cusvaer 22.11.
  * Fix potential circuit mismatches across multiple processes.
  * Fix potential random seed mismatches across multiple processes.
  * Fix the unexpected behavior of `--cachedir`.


## cuquantum-benchmarks v0.1.0

* Initial release

* New features & enhancements:

  * Specify the target benchmark with `--benchmark` (9 benchmarks are offered as of v0.1.0; to be expanded in the future).
  * Create the circuit either with Qiskit or Cirq via `--frontend`.
  * Specify the target CPU or GPU backend (that is compatible with the frontend) via `--backend`.
  * Specify the backend-specific options, such as the number of CPU threads, number of GPUs, etc.
  * Run the benchmark to collect CPU/GPU elapsed time.
  * Store the generated circuit and the benchmark result (as json files) in disk.
