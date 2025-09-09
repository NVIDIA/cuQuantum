cuStateVec Ex samples
=====================

This directory contains examples demonstrating various features of the
cuStateVec Ex API for quantum circuit simulation.

Samples
-------

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

Build Instructions
------------------

Prerequisites:
- CUDA Toolkit
- cuQuantum/cuStateVec library
- Set environment variable: CUSTATEVEC_ROOT or CUQUANTUM_ROOT

Using Make:
    make                    # Build all examples
    make estimate_pi        # Build individual example
    make clean              # Remove executables

Running Examples:
    ./estimate_pi
    ./pauli_functions
    ./noise_channel
    ./index_bit_permutation
    ./quantum_state_initialization

Requirements
------------
- C++11 compiler
- CUDA-capable GPU
- cuQuantum library installation

For more details, refer to the cuQuantum documentation and individual
source file comments.
