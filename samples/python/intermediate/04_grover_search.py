"""
Example 4: Grover's Search Algorithm

Demonstrates:
- Quantum search for marked items
- Oracle implementation
- Amplitude amplification
- Quadratic speedup over classical search

Searches an unsorted database of N=2^n items in O(√N) operations
"""

import numpy as np
import time
try:
    import cupy as cp
    from cuquantum import custatevec as cusv
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def create_oracle(n_qubits, marked_state):
    """
    Create oracle that marks the target state.
    Oracle: O|x⟩ = (-1)^f(x)|x⟩ where f(x)=1 for marked state
    """
    # Convert marked state to binary
    gates = []
    
    # Apply X gates to flip 0s in marked state
    for i in range(n_qubits):
        if not (marked_state >> i) & 1:
            gates.append(('x', i))
    
    # Multi-controlled Z gate
    gates.append(('mcz', list(range(n_qubits))))
    
    # Undo X gates
    for i in range(n_qubits):
        if not (marked_state >> i) & 1:
            gates.append(('x', i))
    
    return gates

def apply_oracle(handle, state, n_qubits, marked_state):
    """Apply the oracle to mark the solution"""
    X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    
    # Flip bits where marked_state has 0
    for i in range(n_qubits):
        if not (marked_state >> i) & 1:
            cusv.apply_matrix(
                handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
                n_qubits, [i], X.ctypes.data,
                cusv.cudaDataType.CUDA_C_32F,
                cusv.MatrixLayout.ROW, 0
            )
    
    # Multi-controlled Z (flip phase of |111...1⟩)
    if n_qubits == 1:
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [0], Z.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    else:
        # For simplicity, apply Z with phase flip
        # In practice, use multi-controlled gates
        state_cpu = state.get()
        all_ones = (1 << n_qubits) - 1
        state_cpu[all_ones] *= -1
        state[:] = cp.asarray(state_cpu)
    
    # Undo X gates
    for i in range(n_qubits):
        if not (marked_state >> i) & 1:
            cusv.apply_matrix(
                handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
                n_qubits, [i], X.ctypes.data,
                cusv.cudaDataType.CUDA_C_32F,
                cusv.MatrixLayout.ROW, 0
            )

def diffusion_operator(handle, state, n_qubits):
    """
    Apply diffusion operator (inversion about average).
    D = 2|ψ⟩⟨ψ| - I where |ψ⟩ is uniform superposition
    """
    H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    
    # H on all qubits
    for i in range(n_qubits):
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [i], H.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    
    # X on all qubits
    for i in range(n_qubits):
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [i], X.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    
    # Multi-controlled Z
    if n_qubits == 1:
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [0], Z.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    else:
        state_cpu = state.get()
        all_ones = (1 << n_qubits) - 1
        state_cpu[all_ones] *= -1
        state[:] = cp.asarray(state_cpu)
    
    # X on all qubits
    for i in range(n_qubits):
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [i], X.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    
    # H on all qubits
    for i in range(n_qubits):
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [i], H.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )

def grover_search(handle, state, n_qubits, marked_state):
    """Run Grover's algorithm"""
    H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
    
    # Calculate optimal number of iterations
    N = 2 ** n_qubits
    optimal_iterations = int(np.pi / 4 * np.sqrt(N))
    optimal_iterations = max(1, optimal_iterations)
    
    print(f"\n🔧 Grover's Algorithm Setup:")
    print(f"   Search space size: {N}")
    print(f"   Marked state: |{marked_state}⟩ (binary: {format(marked_state, f'0{n_qubits}b')})")
    print(f"   Optimal iterations: {optimal_iterations}")
    print(f"   Classical search: O({N}) operations")
    print(f"   Grover's search: O({int(np.sqrt(N))}) operations")
    print(f"   Speedup: ~{N/np.sqrt(N):.1f}x")
    
    # Initialize: Create uniform superposition
    print(f"\n📊 Step 1: Create uniform superposition")
    for i in range(n_qubits):
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [i], H.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    print(f"   ✓ All {N} states have equal amplitude: 1/√{N}")
    
    # Grover iterations
    print(f"\n📊 Step 2: Grover iterations (amplitude amplification)")
    for iteration in range(optimal_iterations):
        # Oracle
        apply_oracle(handle, state, n_qubits, marked_state)
        
        # Diffusion
        diffusion_operator(handle, state, n_qubits)
        
        # Show progress
        if iteration < 3 or iteration == optimal_iterations - 1:
            state_cpu = state.get()
            prob = abs(state_cpu[marked_state]) ** 2
            print(f"   Iteration {iteration + 1}: P(marked state) = {prob:.4f}")
    
    return optimal_iterations

def main():
    if not GPU_AVAILABLE:
        print("\n❌ This example requires GPU support.")
        return
    
    print("="*70)
    print("  Grover's Search Algorithm - Quantum Speedup Demo")
    print("="*70)
    
    # Test with different problem sizes
    test_cases = [
        (3, 5),   # 8 items, search for |101⟩
        (4, 10),  # 16 items, search for |1010⟩
        (5, 21),  # 32 items, search for |10101⟩
    ]
    
    handle = cusv.create()
    
    for n_qubits, marked_state in test_cases:
        N = 2 ** n_qubits
        
        print(f"\n{'='*70}")
        print(f"Searching {N} items for marked state {marked_state}")
        print("="*70)
        
        # Initialize state
        state = cp.zeros(N, dtype=np.complex64)
        state[0] = 1.0
        
        # Run Grover's algorithm
        start_time = time.time()
        num_iterations = grover_search(handle, state, n_qubits, marked_state)
        end_time = time.time()
        
        # Measure result
        print(f"\n📊 Step 3: Measurement")
        state_cpu = state.get()
        
        # Show top 5 measurement probabilities
        probabilities = np.abs(state_cpu) ** 2
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        print(f"\n   Top 5 most likely outcomes:")
        print(f"   {'State':<10} {'Binary':<12} {'Probability':<12}")
        print(f"   {'-'*40}")
        
        for idx in top_indices:
            binary = format(idx, f'0{n_qubits}b')
            prob = probabilities[idx]
            marker = " ← TARGET!" if idx == marked_state else ""
            print(f"   |{idx}⟩{' '*(8-len(str(idx)))} |{binary}⟩{' '*(8-len(binary))} {prob:.6f}{marker}")
        
        success_prob = probabilities[marked_state]
        
        print(f"\n   ✅ Success probability: {success_prob:.4f} ({success_prob*100:.2f}%)")
        print(f"   ⏱️  Execution time: {(end_time - start_time)*1000:.2f} ms")
        print(f"   🔄 Number of Grover iterations: {num_iterations}")
        
        # Classical comparison
        classical_ops = N  # Would need to check all N items
        quantum_ops = num_iterations * 2  # Oracle + Diffusion per iteration
        
        print(f"\n   📊 Complexity Comparison:")
        print(f"      Classical: {classical_ops} operations (O(N))")
        print(f"      Quantum: {quantum_ops} operations (O(√N))")
        print(f"      Speedup: {classical_ops/quantum_ops:.1f}x")
    
    cusv.destroy(handle)
    
    print(f"\n{'='*70}")
    print("🎉 Grover's Search Demonstration Complete!")
    print("="*70)
    
    print("""
✅ You learned:
   - Grover's algorithm finds marked items in O(√N) time
   - Oracle marks the solution by phase flip
   - Diffusion operator amplifies marked state amplitude
   - Quadratic speedup over classical search
   - Success probability increases with iterations

🔬 Key Concepts:
   - Amplitude amplification technique
   - Quantum interference (constructive/destructive)
   - Oracle design for different problems
   - Optimal iteration count: π/4 √N
   - Trade-off between iterations and success rate

📚 Applications:
   - Unstructured search problems
   - NP-complete problem solving (with modifications)
   - Quantum walks
   - Collision finding in cryptography

🚀 Next Steps:
   - Try different problem sizes
   - Implement multi-target search
   - Explore QAOA (05_qaoa_optimization.py)
   - Study amplitude estimation
    """)
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
