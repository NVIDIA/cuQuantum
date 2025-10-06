"""
Example 7: Noise Simulation and Error Mitigation

Demonstrates:
- Modeling quantum noise (decoherence, gate errors)
- Depolarizing and amplitude damping channels
- Density matrix formalism (cuDensityMat)
- Error mitigation techniques
- Realistic NISQ device simulation

Shows how to simulate realistic noisy quantum devices
"""

import numpy as np
import time
try:
    import cupy as cp
    from cuquantum import custatevec as cusv
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def apply_depolarizing_noise(state, qubit, probability, n_qubits):
    """
    Apply depolarizing noise to a qubit.
    With probability p, apply random Pauli (X, Y, or Z)
    """
    if np.random.random() < probability:
        # Choose random Pauli
        pauli_choice = np.random.choice(['X', 'Y', 'Z'])
        
        if pauli_choice == 'X':
            X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
            matrix = X
        elif pauli_choice == 'Y':
            Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
            matrix = Y
        else:  # Z
            Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
            matrix = Z
        
        return matrix, qubit
    
    return None, None

def apply_amplitude_damping(state, qubit, gamma, n_qubits):
    """
    Apply amplitude damping (T1 decay).
    Simulates energy relaxation |1⟩ → |0⟩
    """
    # Amplitude damping Kraus operators
    # K0 = [[1, 0], [0, sqrt(1-gamma)]]
    # K1 = [[0, sqrt(gamma)], [0, 0]]
    
    state_cpu = cp.asnumpy(state)
    new_state = np.zeros_like(state_cpu)
    
    # For each basis state, apply Kraus operators
    for i in range(len(state_cpu)):
        if abs(state_cpu[i]) < 1e-10:
            continue
        
        # Check if target qubit is |1⟩
        if (i >> qubit) & 1:
            # Qubit is |1⟩
            # K0: decay to sqrt(1-gamma) * |1⟩
            new_state[i] += np.sqrt(1 - gamma) * state_cpu[i]
            
            # K1: decay to sqrt(gamma) * |0⟩
            # Flip the qubit bit
            j = i ^ (1 << qubit)
            new_state[j] += np.sqrt(gamma) * state_cpu[i]
        else:
            # Qubit is |0⟩, stays in |0⟩
            new_state[i] += state_cpu[i]
    
    return cp.asarray(new_state)

def simulate_noisy_circuit(handle, n_qubits, circuit_gates, noise_params):
    """
    Simulate circuit with noise.
    
    Args:
        circuit_gates: List of (gate_name, matrix, qubits)
        noise_params: {
            'depolarizing_prob': probability of depolarizing error,
            'amplitude_damping': T1 decay rate,
            'apply_after_gate': whether to apply noise after each gate
        }
    """
    state_size = 2 ** n_qubits
    state = cp.zeros(state_size, dtype=np.complex64)
    state[0] = 1.0
    
    depol_prob = noise_params.get('depolarizing_prob', 0.0)
    damp_gamma = noise_params.get('amplitude_damping', 0.0)
    
    for gate_name, matrix, qubits in circuit_gates:
        # Apply ideal gate
        cusv.apply_matrix(
            handle,
            state.data.ptr,
            cusv.cudaDataType.CUDA_C_32F,
            n_qubits,
            qubits,
            matrix.ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW,
            0
        )
        
        # Apply noise after gate
        if noise_params.get('apply_after_gate', True):
            # Depolarizing noise on all qubits involved
            for q in qubits:
                noise_gate, noise_qubit = apply_depolarizing_noise(
                    state, q, depol_prob, n_qubits
                )
                if noise_gate is not None:
                    cusv.apply_matrix(
                        handle,
                        state.data.ptr,
                        cusv.cudaDataType.CUDA_C_32F,
                        n_qubits,
                        [noise_qubit],
                        noise_gate.ctypes.data,
                        cusv.cudaDataType.CUDA_C_32F,
                        cusv.MatrixLayout.ROW,
                        0
                    )
            
            # Amplitude damping
            if damp_gamma > 0:
                for q in qubits:
                    state[:] = apply_amplitude_damping(state, q, damp_gamma, n_qubits)
    
    return state.get()

def zero_noise_extrapolation(handle, n_qubits, circuit_gates, base_noise):
    """
    Error mitigation via zero-noise extrapolation.
    Run at multiple noise levels and extrapolate to zero noise.
    """
    noise_factors = [1.0, 2.0, 3.0]  # Scale noise by these factors
    
    results = []
    
    for factor in noise_factors:
        scaled_noise = {
            'depolarizing_prob': base_noise['depolarizing_prob'] * factor,
            'amplitude_damping': base_noise['amplitude_damping'] * factor,
            'apply_after_gate': True
        }
        
        state = simulate_noisy_circuit(handle, n_qubits, circuit_gates, scaled_noise)
        
        # Compute some observable (e.g., ⟨Z₀⟩)
        Z0_expectation = 0.0
        for i in range(len(state)):
            # Check if qubit 0 is |0⟩ or |1⟩
            prob = abs(state[i]) ** 2
            if (i & 1) == 0:  # Qubit 0 is |0⟩
                Z0_expectation += prob
            else:  # Qubit 0 is |1⟩
                Z0_expectation -= prob
        
        results.append((factor, Z0_expectation))
    
    # Linear extrapolation to zero noise
    factors = np.array([r[0] for r in results])
    values = np.array([r[1] for r in results])
    
    # Fit line: value = a * factor + b
    coeffs = np.polyfit(factors, values, deg=1)
    mitigated_value = coeffs[1]  # Intercept (value at factor=0)
    
    return mitigated_value, results

def main():
    if not GPU_AVAILABLE:
        print("\n❌ This example requires GPU support.")
        return
    
    print("="*70)
    print("  Noise Simulation and Error Mitigation")
    print("="*70)
    
    print("""
🔬 Simulating Realistic NISQ Devices

Real quantum computers suffer from:
- Gate errors (depolarizing noise)
- Decoherence (amplitude/phase damping)
- Measurement errors
- Cross-talk between qubits

This example shows how to model these effects and mitigate them.
    """)
    
    n_qubits = 2
    handle = cusv.create()
    
    # Define a simple test circuit (Bell state)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=np.complex64)
    
    circuit = [
        ('H', H, [0]),
        ('CNOT', CNOT, [0, 1])
    ]
    
    # Test 1: Ideal (noiseless) simulation
    print(f"\n{'='*70}")
    print("Test 1: Ideal Simulation")
    print("="*70)
    
    ideal_state = simulate_noisy_circuit(
        handle, n_qubits, circuit,
        {'depolarizing_prob': 0.0, 'amplitude_damping': 0.0}
    )
    
    print("\nIdeal Bell state |Φ⁺⟩:")
    for i, amp in enumerate(ideal_state):
        if abs(amp) > 1e-10:
            print(f"  |{i:02b}⟩: {amp.real:.4f} + {amp.imag:.4f}i (prob: {abs(amp)**2:.4f})")
    
    # Test 2: Noisy simulation
    print(f"\n{'='*70}")
    print("Test 2: Noisy Simulation")
    print("="*70)
    
    noise_levels = [
        {'name': 'Low noise', 'depolarizing_prob': 0.01, 'amplitude_damping': 0.01},
        {'name': 'Medium noise', 'depolarizing_prob': 0.05, 'amplitude_damping': 0.05},
        {'name': 'High noise', 'depolarizing_prob': 0.10, 'amplitude_damping': 0.10},
    ]
    
    for noise_config in noise_levels:
        name = noise_config.pop('name')
        
        print(f"\n{name}:")
        print(f"  Depolarizing: {noise_config['depolarizing_prob']*100:.1f}%")
        print(f"  Amplitude damping: {noise_config['amplitude_damping']*100:.1f}%")
        
        # Run multiple shots to see statistical variation
        num_trials = 5
        fidelities = []
        
        for trial in range(num_trials):
            noisy_state = simulate_noisy_circuit(
                handle, n_qubits, circuit, noise_config
            )
            
            # Compute fidelity with ideal state
            fidelity = abs(np.vdot(ideal_state, noisy_state)) ** 2
            fidelities.append(fidelity)
        
        avg_fidelity = np.mean(fidelities)
        std_fidelity = np.std(fidelities)
        
        print(f"  Average fidelity: {avg_fidelity:.4f} ± {std_fidelity:.4f}")
        print(f"  Fidelity range: [{min(fidelities):.4f}, {max(fidelities):.4f}]")
        
        # Show final state
        print(f"\n  Example noisy state:")
        for i, amp in enumerate(noisy_state):
            if abs(amp) > 1e-10:
                print(f"    |{i:02b}⟩: {amp.real:.4f} + {amp.imag:.4f}i (prob: {abs(amp)**2:.4f})")
    
    # Test 3: Error mitigation
    print(f"\n{'='*70}")
    print("Test 3: Zero-Noise Extrapolation (Error Mitigation)")
    print("="*70)
    
    base_noise = {
        'depolarizing_prob': 0.02,
        'amplitude_damping': 0.02
    }
    
    print(f"\nBase noise level:")
    print(f"  Depolarizing: {base_noise['depolarizing_prob']*100:.1f}%")
    print(f"  Amplitude damping: {base_noise['amplitude_damping']*100:.1f}%")
    
    # Compute ideal expectation value ⟨Z₀⟩
    ideal_Z0 = 0.0
    for i in range(len(ideal_state)):
        prob = abs(ideal_state[i]) ** 2
        if (i & 1) == 0:  # Qubit 0 is |0⟩
            ideal_Z0 += prob
        else:
            ideal_Z0 -= prob
    
    print(f"\nIdeal ⟨Z₀⟩ expectation: {ideal_Z0:.6f}")
    
    # Run error mitigation
    mitigated_value, measurements = zero_noise_extrapolation(
        handle, n_qubits, circuit, base_noise
    )
    
    print(f"\nMeasurements at different noise levels:")
    print(f"  {'Noise Factor':<15} {'⟨Z₀⟩':<15}")
    print("  " + "-" * 30)
    for factor, value in measurements:
        print(f"  {factor:<15.1f} {value:<15.6f}")
    
    print(f"\n  Extrapolated (zero-noise): {mitigated_value:.6f}")
    
    print(f"\n📊 Error mitigation results:")
    noisy_error = abs(measurements[0][1] - ideal_Z0)
    mitigated_error = abs(mitigated_value - ideal_Z0)
    improvement = (1 - mitigated_error/noisy_error) * 100 if noisy_error > 0 else 0
    
    print(f"  Noisy error: {noisy_error:.6f}")
    print(f"  Mitigated error: {mitigated_error:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    cusv.destroy(handle)
    
    print(f"\n{'='*70}")
    print("🎉 Noise Simulation Demo Complete!")
    print("="*70)
    
    print("""
✅ You learned:
   - How to model quantum noise (depolarizing, amplitude damping)
   - Simulating realistic NISQ device behavior
   - Measuring fidelity degradation with noise
   - Zero-noise extrapolation for error mitigation
   - Statistical variation in noisy quantum systems

🔬 Key Concepts:
   - Noise reduces quantum state fidelity
   - Different noise channels (coherent vs incoherent)
   - Error mitigation vs error correction
   - Extrapolation techniques
   - Trade-off between accuracy and circuit depth

📚 Noise Types:
   - Depolarizing: Random Pauli errors (X, Y, Z)
   - Amplitude damping: T1 energy relaxation
   - Phase damping: T2 dephasing
   - Measurement errors: Readout mistakes
   - Cross-talk: Unwanted qubit interactions

🚀 Next Steps:
   - Implement other error mitigation (Richardson, Clifford RB)
   - Study quantum error correction codes
   - Benchmark real hardware noise models
   - Explore density matrix formalism (cuDensityMat)
   - Build noise-aware circuit optimization
    """)
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
