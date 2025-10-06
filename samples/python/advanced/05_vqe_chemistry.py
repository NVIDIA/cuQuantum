"""
Example 5: Variational Quantum Eigensolver (VQE)

Demonstrates:
- VQE algorithm for finding ground state energies
- Parameterized quantum circuits (ansatz)
- Hamiltonian expectation values
- Classical optimization loop
- Applications to molecular chemistry

Solves the H₂ molecule ground state energy problem
"""

import numpy as np
import time
try:
    import cupy as cp
    from cuquantum import custatevec as cusv
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def create_h2_hamiltonian():
    """
    Create Hamiltonian for H₂ molecule at equilibrium.
    H = g₀I + g₁Z₀ + g₂Z₁ + g₃Z₀Z₁ + g₄X₀X₁ + g₅Y₀Y₁
    
    Coefficients computed for H₂ at 0.735 Å separation
    """
    # These are pre-computed values for H₂ molecule
    coefficients = {
        'II': -0.4804,   # Identity
        'Z0': 0.3435,    # Z on qubit 0
        'Z1': -0.4347,   # Z on qubit 1
        'Z0Z1': 0.5716,  # Z₀Z₁
        'X0X1': 0.0910,  # X₀X₁
        'Y0Y1': 0.0910,  # Y₀Y₁
    }
    
    return coefficients

def pauli_string_expectation(handle, state, n_qubits, pauli_string):
    """
    Compute expectation value ⟨ψ|P|ψ⟩ for Pauli string P.
    
    Args:
        pauli_string: String like 'X0X1', 'Z0Z1', etc.
    """
    if pauli_string == 'II':
        # Identity always gives 1
        return 1.0
    
    # Define Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    
    pauli_map = {'X': X, 'Y': Y, 'Z': Z}
    
    # Make a copy of the state
    work_state = cp.copy(state)
    
    # Apply Pauli operators
    # Parse string like 'X0X1' or 'Z0'
    i = 0
    while i < len(pauli_string):
        if pauli_string[i] in ['X', 'Y', 'Z']:
            pauli = pauli_string[i]
            # Get qubit index (may be multi-digit)
            j = i + 1
            while j < len(pauli_string) and pauli_string[j].isdigit():
                j += 1
            qubit = int(pauli_string[i+1:j])
            
            matrix = pauli_map[pauli]
            cusv.apply_matrix(
                handle, work_state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
                n_qubits, [qubit], matrix.ctypes.data,
                cusv.cudaDataType.CUDA_C_32F,
                cusv.MatrixLayout.ROW, 0
            )
            i = j
        else:
            i += 1
    
    # Compute ⟨ψ|P|ψ⟩
    expectation = cp.vdot(state, work_state)
    return float(expectation.real)

def hamiltonian_expectation(handle, state, n_qubits, hamiltonian):
    """Compute total Hamiltonian expectation value"""
    energy = 0.0
    
    for pauli_string, coefficient in hamiltonian.items():
        expectation = pauli_string_expectation(handle, state, n_qubits, pauli_string)
        energy += coefficient * expectation
    
    return energy

def apply_ansatz(handle, state, n_qubits, params):
    """
    Apply parameterized circuit (ansatz).
    Uses RY-CNOT ladder structure.
    
    Args:
        params: Array of rotation angles [θ₀, θ₁, θ₂, θ₃]
    """
    # Define gates
    H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=np.complex64)
    
    # Layer 1: Hadamard on all qubits
    for i in range(n_qubits):
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [i], H.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    
    # Layer 2: RY rotations with parameters
    for i in range(n_qubits):
        theta = params[i]
        RY = np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex64)
        
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [i], RY.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    
    # Layer 3: CNOT entangling layer
    if n_qubits > 1:
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [0, 1], CNOT.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )
    
    # Layer 4: Second RY layer
    for i in range(n_qubits):
        theta = params[n_qubits + i]
        RY = np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex64)
        
        cusv.apply_matrix(
            handle, state.data.ptr, cusv.cudaDataType.CUDA_C_32F,
            n_qubits, [i], RY.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW, 0
        )

def vqe_cost_function(params, handle, n_qubits, hamiltonian):
    """Cost function to minimize (energy expectation)"""
    # Initialize state to |0...0⟩
    N = 2 ** n_qubits
    state = cp.zeros(N, dtype=np.complex64)
    state[0] = 1.0
    
    # Apply ansatz with current parameters
    apply_ansatz(handle, state, n_qubits, params)
    
    # Compute energy
    energy = hamiltonian_expectation(handle, state, n_qubits, hamiltonian)
    
    return energy

def gradient_descent_optimize(handle, n_qubits, hamiltonian, initial_params, 
                              learning_rate=0.1, max_iterations=100, tolerance=1e-6):
    """Simple gradient descent optimizer"""
    params = np.array(initial_params, dtype=np.float64)
    epsilon = 1e-7  # For numerical gradient
    
    energies = []
    
    print(f"\n🔧 Optimization Setup:")
    print(f"   Method: Gradient Descent")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Convergence tolerance: {tolerance}")
    
    print(f"\n{'Iter':<6} {'Energy (Ha)':<15} {'Gradient Norm':<15} {'Improvement'}")
    print("-" * 60)
    
    for iteration in range(max_iterations):
        # Compute current energy
        energy = vqe_cost_function(params, handle, n_qubits, hamiltonian)
        energies.append(energy)
        
        # Compute numerical gradient
        gradient = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            energy_plus = vqe_cost_function(params_plus, handle, n_qubits, hamiltonian)
            
            gradient[i] = (energy_plus - energy) / epsilon
        
        grad_norm = np.linalg.norm(gradient)
        
        # Print progress
        if iteration == 0:
            improvement = 0.0
        else:
            improvement = energies[-2] - energy
        
        if iteration < 10 or iteration % 10 == 0 or iteration == max_iterations - 1:
            print(f"{iteration:<6} {energy:<15.8f} {grad_norm:<15.8e} {improvement:+.8f}")
        
        # Check convergence
        if grad_norm < tolerance:
            print(f"\n✅ Converged at iteration {iteration}")
            break
        
        # Update parameters
        params -= learning_rate * gradient
    
    return params, energies

def main():
    if not GPU_AVAILABLE:
        print("\n❌ This example requires GPU support.")
        return
    
    print("="*70)
    print("  Variational Quantum Eigensolver (VQE) - H₂ Molecule")
    print("="*70)
    
    n_qubits = 2  # Minimal encoding for H₂
    
    print(f"""
🧪 Problem: Find ground state energy of H₂ molecule

   System: Hydrogen molecule (H₂)
   Bond length: 0.735 Å (equilibrium)
   Qubits: {n_qubits}
   Encoding: Minimal qubit encoding
   
   Known exact ground state: -1.137 Hartree
    """)
    
    # Create Hamiltonian
    hamiltonian = create_h2_hamiltonian()
    
    print("📊 Hamiltonian Terms:")
    print("   H = Σᵢ cᵢ Pᵢ where Pᵢ are Pauli strings")
    print(f"\n   {'Pauli String':<10} {'Coefficient (Ha)':<20}")
    print("   " + "-" * 35)
    for pauli, coeff in hamiltonian.items():
        print(f"   {pauli:<10} {coeff:>15.6f}")
    
    # Initialize cuStateVec
    handle = cusv.create()
    
    # Initial parameters (random start)
    np.random.seed(42)
    initial_params = np.random.uniform(-np.pi, np.pi, size=2*n_qubits)
    
    print(f"\n🎲 Initial Parameters:")
    print(f"   θ = [{', '.join([f'{p:.4f}' for p in initial_params])}]")
    
    initial_energy = vqe_cost_function(initial_params, handle, n_qubits, hamiltonian)
    print(f"   Initial energy: {initial_energy:.8f} Ha")
    
    # Run VQE optimization
    print(f"\n{'='*70}")
    print("🚀 Running VQE Optimization")
    print("="*70)
    
    start_time = time.time()
    optimal_params, energy_history = gradient_descent_optimize(
        handle, n_qubits, hamiltonian, initial_params,
        learning_rate=0.1, max_iterations=100
    )
    end_time = time.time()
    
    final_energy = energy_history[-1]
    exact_energy = -1.137  # Known exact value
    
    print(f"\n{'='*70}")
    print("📊 VQE Results")
    print("="*70)
    
    print(f"\n🎯 Final Results:")
    print(f"   Optimized energy: {final_energy:.8f} Ha")
    print(f"   Exact energy: {exact_energy:.8f} Ha")
    print(f"   Error: {abs(final_energy - exact_energy):.8f} Ha")
    print(f"   Relative error: {abs(final_energy - exact_energy)/abs(exact_energy)*100:.4f}%")
    print(f"   Optimization time: {(end_time - start_time):.2f} seconds")
    print(f"   Iterations: {len(energy_history)}")
    
    print(f"\n🔧 Optimal Parameters:")
    print(f"   θ = [{', '.join([f'{p:.4f}' for p in optimal_params])}]")
    
    print(f"\n📈 Energy Convergence:")
    print(f"   Starting energy: {energy_history[0]:.8f} Ha")
    print(f"   Final energy: {final_energy:.8f} Ha")
    print(f"   Energy lowered by: {energy_history[0] - final_energy:.8f} Ha")
    
    # Show energy trajectory
    print(f"\n   Energy trajectory (every 10 iterations):")
    for i in range(0, len(energy_history), 10):
        progress = "█" * int((i / len(energy_history)) * 30)
        print(f"   Iter {i:3d}: {energy_history[i]:.8f} Ha {progress}")
    
    cusv.destroy(handle)
    
    print(f"\n{'='*70}")
    print("🎉 VQE Demonstration Complete!")
    print("="*70)
    
    print("""
✅ You learned:
   - VQE finds ground states using quantum-classical hybrid approach
   - Parameterized quantum circuits (ansatz) prepare trial states
   - Classical optimizer minimizes energy expectation value
   - Can achieve chemical accuracy with shallow circuits
   - Suitable for near-term quantum devices (NISQ era)

🔬 Key Concepts:
   - Variational principle: E[ψ(θ)] ≥ E_ground
   - Hamiltonian expectation: ⟨ψ|H|ψ⟩
   - Ansatz design affects convergence
   - Gradient-based vs gradient-free optimizers
   - Measurement shot noise considerations

📚 Applications:
   - Molecular chemistry (drug discovery)
   - Materials science (catalysts, batteries)
   - Condensed matter physics
   - Nuclear physics calculations

🚀 Next Steps:
   - Try different ansatz designs
   - Implement UCCSD ansatz for chemistry
   - Explore QAOA (05_qaoa_optimization.py)
   - Use hardware-efficient ansatzes
   - Add shot noise simulation
    """)
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
