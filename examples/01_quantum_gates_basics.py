"""
Example 1: Basic State Vector Operations with cuStateVec

This example demonstrates fundamental quantum operations:
- Creating and manipulating quantum states
- Applying single-qubit gates
- Applying two-qubit gates
- Measuring quantum states

No GPU required for this conceptual example (uses numpy),
but demonstrates the cuQuantum API patterns.
"""

import numpy as np

def print_header(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    print_header("cuStateVec Basics: Quantum Gate Operations")
    
    # Configuration
    n_qubits = 3
    print(f"\n📊 Working with {n_qubits} qubits")
    print(f"   State space dimension: {2**n_qubits}")
    
    # Initialize state vector |000⟩
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    
    print(f"\n🔧 Initial state: |{'0'*n_qubits}⟩")
    print(f"   Amplitude of |000⟩: {state[0]}")
    
    # Define quantum gates
    print_header("Quantum Gates Library")
    
    # Pauli X (NOT gate)
    X = np.array([[0, 1],
                  [1, 0]], dtype=np.complex128)
    print("\n✓ Pauli-X (NOT) gate:")
    print(X)
    
    # Pauli Y gate
    Y = np.array([[0, -1j],
                  [1j, 0]], dtype=np.complex128)
    print("\n✓ Pauli-Y gate:")
    print(Y)
    
    # Pauli Z gate
    Z = np.array([[1, 0],
                  [0, -1]], dtype=np.complex128)
    print("\n✓ Pauli-Z gate:")
    print(Z)
    
    # Hadamard gate
    H = np.array([[1, 1],
                  [1, -1]], dtype=np.complex128) / np.sqrt(2)
    print("\n✓ Hadamard gate:")
    print(H)
    
    # CNOT gate
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=np.complex128)
    print("\n✓ CNOT gate:")
    print(CNOT)
    
    # Rotation gates
    def RX(theta):
        """Rotation around X axis"""
        return np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
    
    def RY(theta):
        """Rotation around Y axis"""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
    
    def RZ(theta):
        """Rotation around Z axis"""
        return np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=np.complex128)
    
    print("\n✓ Rotation gates: RX(θ), RY(θ), RZ(θ)")
    print(f"   Example RY(π/4):")
    print(RY(np.pi/4))
    
    print_header("Gate Application Examples")
    
    # Example 1: Single qubit gate
    print("\n📌 Example 1: Apply Hadamard to qubit 0")
    print("   Initial: |000⟩")
    print("   Result: (|000⟩ + |100⟩)/√2")
    
    # Example 2: Multiple gates
    print("\n📌 Example 2: Create Bell state")
    print("   Step 1: H on qubit 0 → (|00⟩ + |10⟩)/√2")
    print("   Step 2: CNOT(0,1) → (|00⟩ + |11⟩)/√2")
    print("   Result: Bell state Φ⁺")
    
    # Example 3: Three-qubit entanglement
    print("\n📌 Example 3: Create GHZ state")
    print("   Step 1: H on qubit 0")
    print("   Step 2: CNOT(0,1)")
    print("   Step 3: CNOT(0,2)")
    print("   Result: (|000⟩ + |111⟩)/√2")
    
    print_header("Measurement Basics")
    
    print("\n🎲 Measurement in computational basis:")
    print("   - Projects state onto |0⟩ or |1⟩ for each qubit")
    print("   - Probability of outcome |x⟩ is |⟨x|ψ⟩|²")
    print("   - State collapses to measured outcome")
    
    print("\n📊 Example probabilities for Bell state (|00⟩ + |11⟩)/√2:")
    print("   P(|00⟩) = 50%")
    print("   P(|11⟩) = 50%")
    print("   P(|01⟩) = 0%")
    print("   P(|10⟩) = 0%")
    
    print_header("Common Quantum Circuits")
    
    print("\n🔄 Quantum Fourier Transform (QFT):")
    print("   Used in: Shor's algorithm, phase estimation")
    print("   Gates: Hadamard + Controlled phase rotations")
    
    print("\n🔍 Quantum Phase Estimation:")
    print("   Used in: Finding eigenvalues")
    print("   Components: QFT + Controlled unitaries")
    
    print("\n🎯 Variational Quantum Eigensolver (VQE):")
    print("   Used in: Quantum chemistry, optimization")
    print("   Components: Parameterized gates + Classical optimizer")
    
    print("\n🔗 Quantum Approximate Optimization (QAOA):")
    print("   Used in: Combinatorial optimization")
    print("   Components: Mixer + Problem Hamiltonians")
    
    print_header("cuStateVec API Pattern")
    
    print("""
With actual cuStateVec (on GPU), the pattern is:

1. Create handle:
   handle = custatevec.create()

2. Allocate state on GPU:
   state = cp.zeros(2**n_qubits, dtype=cp.complex64)
   state[0] = 1.0

3. Apply gates:
   custatevec.apply_matrix(
       handle, state, n_qubits, adjoint, 
       targets, gate_matrix, layout, compute_type
   )

4. Measure/Sample:
   results = custatevec.sampler_sample(
       handle, state, n_shots, output
   )

5. Cleanup:
   custatevec.destroy(handle)
    """)
    
    print_header("Next Steps")
    
    print("""
✅ You now understand:
   - Basic quantum gates (X, Y, Z, H, CNOT)
   - Rotation gates (RX, RY, RZ)
   - Common quantum circuits
   - cuStateVec API pattern

📚 Next examples to try:
   - 02_bell_state.py - Create and measure Bell states
   - 03_qft_circuit.py - Quantum Fourier Transform
   - 04_grover_search.py - Grover's search algorithm
   - notebooks/tutorial_01_basics.ipynb - Interactive tutorial

🚀 Ready to run on GPU?
   See: examples/gpu/01_custatevec_basics.py
    """)
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
