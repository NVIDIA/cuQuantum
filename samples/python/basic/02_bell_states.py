"""
Example 2: Creating and Measuring Bell States

This example demonstrates:
- Creating all four Bell states
- Understanding quantum entanglement
- Measuring entangled states
- Statistical analysis of measurement results

Requires: cupy, cuquantum
"""

import numpy as np
try:
    import cupy as cp
    from cuquantum import custatevec as cusv
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPU libraries not available. Install with:")
    print("   pip install cuquantum-python cupy-cuda12x")

def create_bell_state(handle, state, bell_type='phi_plus'):
    """
    Create one of the four Bell states.
    
    Bell states:
    - Φ⁺ (phi_plus):   (|00⟩ + |11⟩)/√2
    - Φ⁻ (phi_minus):  (|00⟩ - |11⟩)/√2
    - Ψ⁺ (psi_plus):   (|01⟩ + |10⟩)/√2
    - Ψ⁻ (psi_minus):  (|01⟩ - |10⟩)/√2
    """
    # Define gates
    H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex64)
    
    # All Bell states start with H on qubit 0, then CNOT(0,1)
    cusv.apply_matrix(
        handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
        2, [0], H.ctypes.data, cusv.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, 0
    )
    
    cusv.apply_matrix(
        handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
        2, [0, 1], CNOT.ctypes.data, cusv.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, 0
    )
    
    # Apply additional gates for other Bell states
    if bell_type == 'phi_minus':
        # Apply Z to qubit 0
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            2, [0], Z.ctypes.data, cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    elif bell_type == 'psi_plus':
        # Apply X to qubit 1
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            2, [1], X.ctypes.data, cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    elif bell_type == 'psi_minus':
        # Apply Z to qubit 0 and X to qubit 1
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            2, [0], Z.ctypes.data, cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            2, [1], X.ctypes.data, cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )

def measure_and_analyze(handle, state, n_shots, bell_type):
    """Measure the Bell state and analyze correlations"""
    
    # Create sampler
    sampler = cusv.sampler_create(
        handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
        2, 0, n_shots, 0
    )
    
    # Sample measurements
    bitstrings = np.zeros(n_shots, dtype=np.int64)
    cusv.sampler_sample(handle, sampler, bitstrings.ctypes.data, 0, 0, n_shots)
    cusv.sampler_destroy(sampler)
    
    # Analyze results
    from collections import Counter
    counts = Counter(bitstrings)
    
    print(f"\n📊 Measurement Results for {bell_type}:")
    print(f"   Total shots: {n_shots}")
    print(f"\n   State    Count    Probability")
    print(f"   " + "-"*35)
    
    for state_val in [0, 1, 2, 3]:  # |00⟩, |01⟩, |10⟩, |11⟩
        count = counts.get(state_val, 0)
        prob = count / n_shots * 100
        binary = format(state_val, '02b')
        print(f"   |{binary}⟩    {count:>5}    {prob:>6.2f}%")
    
    # Check correlations
    correlation = 0
    for bitstring in bitstrings:
        bit0 = (bitstring >> 1) & 1
        bit1 = bitstring & 1
        if bit0 == bit1:
            correlation += 1
    
    correlation_pct = correlation / n_shots * 100
    
    print(f"\n   🔗 Correlation Analysis:")
    print(f"   Measurements where both qubits agree: {correlation_pct:.1f}%")
    
    if bell_type in ['phi_plus', 'phi_minus']:
        print(f"   ✓ Expected: ~100% (perfect correlation)")
    else:
        print(f"   ✓ Expected: ~0% (perfect anti-correlation)")
    
    return counts

def main():
    if not GPU_AVAILABLE:
        print("\n❌ This example requires GPU support.")
        print("   Please install cuQuantum and CuPy.")
        return
    
    print("="*70)
    print("  Bell States: Quantum Entanglement Demo")
    print("="*70)
    
    # Configuration
    n_qubits = 2
    n_shots = 10000
    
    # Create cuStateVec handle
    handle = cusv.create()
    
    # Test all four Bell states
    bell_states = {
        'phi_plus': '(|00⟩ + |11⟩)/√2',
        'phi_minus': '(|00⟩ - |11⟩)/√2',
        'psi_plus': '(|01⟩ + |10⟩)/√2',
        'psi_minus': '(|01⟩ - |10⟩)/√2'
    }
    
    for bell_type, description in bell_states.items():
        print(f"\n{'='*70}")
        print(f"Creating {bell_type.upper()}: {description}")
        print("="*70)
        
        # Initialize state
        state = cp.zeros(2**n_qubits, dtype=cp.complex64)
        state[0] = 1.0
        
        # Create Bell state
        create_bell_state(handle, state, bell_type)
        
        # Show state vector
        state_cpu = state.get()
        print(f"\n📈 State Vector:")
        for i, amp in enumerate(state_cpu):
            if abs(amp) > 1e-10:
                binary = format(i, '02b')
                print(f"   |{binary}⟩: {amp.real:+.4f} {amp.imag:+.4f}j")
        
        # Measure and analyze
        measure_and_analyze(handle, state, n_shots, bell_type)
    
    # Cleanup
    cusv.destroy(handle)
    
    print(f"\n{'='*70}")
    print("🎉 Bell State Demonstration Complete!")
    print("="*70)
    
    print("""
✅ You learned:
   - How to create all four Bell states
   - Quantum entanglement manifestations
   - Measuring entangled qubits
   - Statistical correlation analysis

🔬 Key Observations:
   - Φ states show perfect correlation (both 0 or both 1)
   - Ψ states show perfect anti-correlation (one 0, one 1)
   - Individual qubit measurements are random (50/50)
   - Joint measurements show quantum correlations

📚 Next Steps:
   - Try 03_qft_circuit.py for Quantum Fourier Transform
   - Explore notebooks/tutorial_02_entanglement.ipynb
   - Learn about Bell inequality violations
    """)
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
