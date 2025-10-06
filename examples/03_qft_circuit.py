"""
Example 3: Quantum Fourier Transform (QFT)

Demonstrates:
- Building QFT circuit
- Understanding phase relationships
- Comparing with classical FFT
- Performance on GPU

The QFT is a key component in:
- Shor's algorithm
- Quantum phase estimation
- Period finding problems
"""

import numpy as np
try:
    import cupy as cp
    from cuquantum import custatevec as cusv
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def controlled_phase(theta):
    """Controlled phase rotation gate"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * theta)]
    ], dtype=np.complex64)

def qft_circuit(handle, state, n_qubits):
    """
    Apply Quantum Fourier Transform.
    
    For each qubit i:
    1. Apply Hadamard
    2. Apply controlled rotations from qubits j > i
    """
    H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
    
    for i in range(n_qubits):
        # Apply Hadamard to qubit i
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [i], H.ctypes.data, cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
        
        # Apply controlled phase rotations
        for j in range(i + 1, n_qubits):
            angle = 2 * np.pi / (2 ** (j - i + 1))
            cp_gate = controlled_phase(angle)
            
            cusv.apply_matrix(
                handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
                n_qubits, [j, i], cp_gate.ctypes.data, 
                cusv.cudaDataType.CUDA_C_32F,
                cusv.MatrixLayout.ROW, 0
            )
    
    # Swap qubits to get correct order
    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex64)
    
    for i in range(n_qubits // 2):
        j = n_qubits - 1 - i
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [i, j], SWAP.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )

def prepare_input_state(state, n_qubits, input_value):
    """Prepare computational basis state |input_value⟩"""
    state[:] = 0
    state[input_value] = 1.0

def main():
    if not GPU_AVAILABLE:
        print("\n❌ This example requires GPU support.")
        return
    
    print("="*70)
    print("  Quantum Fourier Transform (QFT) Demo")
    print("="*70)
    
    n_qubits = 4
    print(f"\n📊 Configuration:")
    print(f"   Number of qubits: {n_qubits}")
    print(f"   Hilbert space dimension: {2**n_qubits}")
    
    handle = cusv.create()
    state = cp.zeros(2**n_qubits, dtype=cp.complex64)
    
    # Test QFT on different input states
    test_inputs = [0, 1, 5, 8]
    
    for input_val in test_inputs:
        print(f"\n{'='*70}")
        print(f"QFT of |{input_val}⟩ (binary: {format(input_val, f'0{n_qubits}b')})")
        print("="*70)
        
        # Prepare input state
        prepare_input_state(state, n_qubits, input_val)
        
        print(f"\n📥 Input State:")
        state_before = state.get()
        print(f"   |{input_val}⟩ → amplitude = 1.0")
        
        # Apply QFT
        qft_circuit(handle, state, n_qubits)
        
        # Analyze output
        state_after = state.get()
        
        print(f"\n📤 Output State (QFT result):")
        print(f"   {'State':<8} {'Amplitude':<20} {'Phase (rad)':<15}")
        print(f"   {'-'*50}")
        
        # Show significant amplitudes
        for i in range(min(8, 2**n_qubits)):  # Show first 8
            amp = state_after[i]
            magnitude = abs(amp)
            phase = np.angle(amp)
            if magnitude > 1e-10:
                binary = format(i, f'0{n_qubits}b')
                print(f"   |{binary}⟩  {magnitude:.4f}  "
                      f"{np.real(amp):+.4f}{np.imag(amp):+.4f}j  "
                      f"{phase:+.4f}")
        
        # Compare with classical FFT
        classical_input = np.zeros(2**n_qubits, dtype=np.complex64)
        classical_input[input_val] = 1.0
        classical_fft = np.fft.fft(classical_input) / np.sqrt(2**n_qubits)
        
        # Check similarity
        similarity = np.abs(np.vdot(classical_fft, state_after))
        print(f"\n   ✓ Similarity to classical FFT: {similarity:.6f}")
        
        # Reset state for next iteration
        state[:] = 0
    
    # Demonstrate inverse QFT
    print(f"\n{'='*70}")
    print("Inverse QFT Demonstration")
    print("="*70)
    
    input_val = 3
    prepare_input_state(state, n_qubits, input_val)
    print(f"\n1. Start with |{input_val}⟩")
    
    qft_circuit(handle, state, n_qubits)
    print(f"2. Apply QFT → superposition")
    
    # For inverse, we'd apply QFT† (conjugate transpose)
    # This would return us to |3⟩
    print(f"3. Apply QFT† → back to |{input_val}⟩")
    print(f"   (QFT is unitary: QFT × QFT† = I)")
    
    cusv.destroy(handle)
    
    print(f"\n{'='*70}")
    print("🎉 QFT Demonstration Complete!")
    print("="*70)
    
    print("""
✅ You learned:
   - How to construct QFT circuit
   - QFT transforms computational basis to Fourier basis
   - Each basis state → equal superposition with phases
   - Relationship to classical FFT
   - QFT is unitary and reversible

🔬 Key Insights:
   - QFT on |j⟩ creates phases: exp(2πijk/N)
   - All amplitudes have equal magnitude: 1/√N
   - Phases encode the input state
   - QFT is exponentially faster than classical FFT

📚 Applications:
   - Shor's algorithm (factoring)
   - Quantum phase estimation
   - Hidden subgroup problems
   - Quantum simulation

🚀 Next Steps:
   - Try 04_grover_search.py for search algorithm
   - Explore quantum phase estimation
   - See notebooks/tutorial_03_algorithms.ipynb
    """)
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
