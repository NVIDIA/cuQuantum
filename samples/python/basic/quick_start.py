#!/usr/bin/env python3
"""
Quick Start Example for cuQuantum

This script demonstrates the basic usage of cuQuantum for quantum circuit simulation.
Perfect for first-time users!

Requirements:
    - cuQuantum installed (conda install -c conda-forge cuquantum)
    - CuPy installed
    - NVIDIA GPU with CUDA support

Usage:
    python quick_start.py
"""

import numpy as np
import cupy as cp
from cuquantum import custatevec as cusv

def main():
    """Run a simple quantum circuit simulation"""
    
    print("=" * 70)
    print("cuQuantum Quick Start Example")
    print("=" * 70)
    
    # Configuration
    n_qubits = 10
    n_shots = 1000
    
    print(f"\n📊 Configuration:")
    print(f"   Number of qubits: {n_qubits}")
    print(f"   State vector size: {2**n_qubits:,} complex numbers")
    print(f"   Memory required: {2**n_qubits * 8 / 1024 / 1024:.2f} MB")
    print(f"   Number of shots: {n_shots}")
    
    # Step 1: Initialize state vector
    print(f"\n🔧 Step 1: Initializing {n_qubits}-qubit state vector...")
    state_vector = cp.zeros(2**n_qubits, dtype=cp.complex64)
    state_vector[0] = 1.0  # |00...0⟩ state
    print(f"   ✓ State initialized to |{'0' * n_qubits}⟩")
    
    # Step 2: Create cuStateVec handle
    print(f"\n🔧 Step 2: Creating cuStateVec handle...")
    handle = cusv.create()
    print(f"   ✓ cuStateVec handle created")
    
    # Step 3: Apply Hadamard gates to create superposition
    print(f"\n🔧 Step 3: Applying Hadamard gates to all qubits...")
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
    
    for i in range(n_qubits):
        cusv.apply_matrix(
            handle, 
            state_vector.data.ptr,
            cusv.cudaDataType.CUDA_C_32F,
            n_qubits,
            [i],
            hadamard.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW,
            0
        )
    print(f"   ✓ All qubits now in equal superposition")
    print(f"   📈 State: (|0⟩ + |1⟩)/√2 for each qubit")
    
    # Step 4: Apply some entangling gates (CNOT)
    print(f"\n🔧 Step 4: Applying CNOT gates for entanglement...")
    cnot = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex64)
    
    # Apply CNOT between adjacent qubits
    for i in range(n_qubits - 1):
        cusv.apply_matrix(
            handle,
            state_vector.data.ptr,
            cusv.cudaDataType.CUDA_C_32F,
            n_qubits,
            [i, i + 1],
            cnot.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW,
            0
        )
    print(f"   ✓ {n_qubits - 1} CNOT gates applied")
    print(f"   🔗 Qubits are now entangled")
    
    # Step 5: Perform measurements
    print(f"\n🔧 Step 5: Measuring the quantum state...")
    print(f"   Taking {n_shots} measurement shots...")
    
    # Create sampler
    sampler = cusv.sampler_create(
        handle,
        state_vector.data.ptr,
        cusv.cudaDataType.CUDA_C_32F,
        n_qubits,
        0,
        n_shots,
        0
    )
    
    # Sample
    bitstrings = np.zeros(n_shots, dtype=np.int64)
    cusv.sampler_sample(
        handle,
        sampler,
        bitstrings.ctypes.data,
        0,
        0,
        n_shots
    )
    
    # Cleanup sampler
    cusv.sampler_destroy(sampler)
    
    print(f"   ✓ Measurements complete")
    
    # Step 6: Analyze results
    print(f"\n📊 Step 6: Analyzing measurement results...")
    
    # Convert to binary strings
    def int_to_bitstring(val, n_bits):
        return format(val, f'0{n_bits}b')
    
    # Count occurrences
    from collections import Counter
    counts = Counter(bitstrings)
    
    # Show top 10 results
    print(f"\n   Top 10 measured states:")
    print(f"   {'State':<{n_qubits+2}} {'Count':>6} {'Probability':>12}")
    print(f"   {'-' * (n_qubits + 22)}")
    
    for state, count in counts.most_common(10):
        bitstring = int_to_bitstring(state, n_qubits)
        probability = count / n_shots * 100
        print(f"   |{bitstring}⟩  {count:>6}  {probability:>10.2f}%")
    
    # Statistics
    print(f"\n   📈 Statistics:")
    print(f"      Unique states measured: {len(counts)}")
    print(f"      Most common state: |{int_to_bitstring(counts.most_common(1)[0][0], n_qubits)}⟩")
    print(f"      Distribution: {'Uniform' if len(counts) > n_shots * 0.5 else 'Peaked'}")
    
    # Step 7: Cleanup
    print(f"\n🔧 Step 7: Cleaning up resources...")
    cusv.destroy(handle)
    print(f"   ✓ Resources released")
    
    # Final summary
    print(f"\n{'=' * 70}")
    print(f"✅ Quick Start Complete!")
    print(f"{'=' * 70}")
    print(f"\nYou've successfully:")
    print(f"  ✓ Initialized a {n_qubits}-qubit quantum state")
    print(f"  ✓ Applied Hadamard gates to create superposition")
    print(f"  ✓ Applied CNOT gates to create entanglement")
    print(f"  ✓ Measured the quantum state {n_shots} times")
    print(f"  ✓ Analyzed the measurement results")
    print(f"\n🎉 Congratulations! You're now ready to explore cuQuantum!")
    print(f"\n📚 Next steps:")
    print(f"   - Try examples/quantum_algorithms/ for more algorithms")
    print(f"   - Check examples/notebooks/ for interactive tutorials")
    print(f"   - Read the documentation: https://docs.nvidia.com/cuda/cuquantum/")
    print(f"\n{'=' * 70}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print(f"\n🔍 Troubleshooting:")
        print(f"   1. Ensure NVIDIA GPU is available")
        print(f"   2. Check CUDA installation: nvidia-smi")
        print(f"   3. Verify cuQuantum installation: python -c 'import cuquantum'")
        print(f"   4. Check CuPy installation: python -c 'import cupy'")
        raise
