cuStateVec Ex samples
=====================

This directory contains examples demonstrating various features of the
cuStateVec Ex API for quantum circuit simulation.

Samples
-------

Single Device Samples:

1. estimate_pi.cpp
   Quantum phase estimation algorithm to estimate the value of pi.
   Derived from Qiskit's pi estimation example.
   Demonstrates two circuit simulation approaches using
   custatevecExApplyMatrix() and StateVectorUpdater.

2. pauli_functions.cpp
   Demonstrates Pauli rotations and expectation value calculations
   using cuStateVecEx Pauli matrix APIs.

3. noise_channel.cpp
   Advanced SVUpdater workflow and quantum noise modeling.
   Demonstrates "build once, apply many times" pattern with
   GHZ fidelity measurements.

4. index_bit_permutation.cpp
   Bit permutation operations on quantum state vectors.
   Demonstrates qubit index reordering and state vector manipulation.

5. quantum_state_initialization.cpp
   Quantum state initialization and advanced state vector operations.
   Demonstrates setting quantum state, device resource access, and
   wire ordering management with state verification.

Multi-Device/Multi-Process Samples:

The following samples run with all state vector configurations, single-device, multi-device and multi-process.

- Without specifying options, those samples run with single device configuration.
- By specifying the number of devices to -d option, samples will run with the specified number of devices.
- If Open MPI is available, and the number of processes is a multiple of two, samples will run with multiple-processes.

All samples support the following command line options:
  -h           Show help message
  -q           Quiet mode (suppress output)
  -d <num>     Number of devices for multi-device configuration
  -t <type>    Data type: 'f'/'float' (default) or 'd'/'double'
  -k <net>     Device network topology for multi-device: 1=SWITCH (default), 2=FULLMESH
               Network structure for multi-process: 3=SuperPOD (default), 4=GB200NVL, 5=SwitchTree, 6=Communicator

6. interoperability_dot.cpp
   cuStateVec Ex API interoperability with cuBLAS for dot product computation.
   Supports single-device, multi-device, and multi-process configurations.
   - Create two state vector instances with different wire orderings.
   - Modify using cuStateVec API.
   - Retrieve GPU memory pointers and CUDA streams from state vector to compute
     the dot product using cuBLAS.
   - Demonstrate a common way to permute wires for two state vectors to have the
     same wire ordering.

7. quantum_volume.cpp
   Quick performance check using quantum volume circuits.
   Supports single-device and multi-device and multi-process configurations.
   Demonstrates quantum volume circuit generation, scalability analysis
   across multiple qubit counts, and reports performance metrics.

External Communicator Plugin
-----------------------------

The cuStateVec Ex API supports custom inter-process communicators through
an external plugin interface. This allows users to implement their own
communication layer instead of using the built-in OPENMPI/MPICH support.

8. mpiCommunicator.c
   Example C-based external MPI communicator plugin demonstrating how to
   implement a custom communicator. This plugin wraps MPI functions to provide
   inter-process communication for distributed quantum circuit simulation.

Building the External Communicator:
    ./build_mpi_communicator.sh              # Build with default MPI
    ./build_mpi_communicator.sh /opt/mpi     # Build with MPI from custom path

This creates libmpiCommunicator.so which can be loaded by cuStateVec Ex API.

Using the External Communicator:
1. Edit stateVectorConstruction.cpp to activate the configuration for
   external communicator.  (Currently commented out.)
2. Rebuild the samples
3. Run: mpirun -np 4 ./quantum_volume

Build Instructions
------------------

Requirements:
- CUDA Toolkit (CUDA 12 or later)
- cuQuantum/cuStateVec library and headers
- cuBLAS library
- MPI (optional, for multi-process samples)

Using Make:
    make                    # Build all examples
    make estimate_pi        # Build individual example
    make clean              # Remove executables

Running Single Device Examples:
    ./estimate_pi                    # Single-device mode (default)
    ./estimate_pi -q                 # Quiet mode (no output)

Running Multi-Device/Multi-Process Examples:
    ./interoperability_dot                # Single-device mode
    ./interoperability_dot -d 2           # Multi-device mode with 2 devices
    mpirun -np 4 ./interoperability_dot   # Multi-process mode

For more details, refer to the cuQuantum documentation and individual
source file comments.
