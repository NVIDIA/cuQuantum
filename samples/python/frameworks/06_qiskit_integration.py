"""
Example 6: Framework Integration - Qiskit Backend

Demonstrates:
- Using cuQuantum as a Qiskit backend
- Converting Qiskit circuits to cuQuantum
- GPU-accelerated simulation of Qiskit circuits
- Performance comparison with default simulator

Shows how to leverage existing Qiskit workflows with GPU acceleration
"""

import numpy as np
import time

# Check if Qiskit is available
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFT
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️  Qiskit not installed. Install with: pip install qiskit")

try:
    import cupy as cp
    from cuquantum import custatevec as cusv
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def qiskit_to_cuquantum(circuit):
    """Convert Qiskit circuit to cuQuantum operations"""
    n_qubits = circuit.num_qubits
    
    # Gate mapping
    gate_mapping = {
        'h': np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2),
        'x': np.array([[0, 1], [1, 0]], dtype=np.complex64),
        'y': np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
        'z': np.array([[1, 0], [0, -1]], dtype=np.complex64),
        's': np.array([[1, 0], [0, 1j]], dtype=np.complex64),
        't': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex64),
        'cx': np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype=np.complex64),
    }
    
    operations = []
    
    for instruction in circuit.data:
        gate = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        
        gate_name = gate.name.lower()
        
        if gate_name in gate_mapping:
            operations.append({
                'gate': gate_name,
                'matrix': gate_mapping[gate_name],
                'qubits': qubits
            })
        elif gate_name in ['rx', 'ry', 'rz']:
            # Parameterized rotation gates
            theta = gate.params[0]
            if gate_name == 'rx':
                matrix = np.array([
                    [np.cos(theta/2), -1j*np.sin(theta/2)],
                    [-1j*np.sin(theta/2), np.cos(theta/2)]
                ], dtype=np.complex64)
            elif gate_name == 'ry':
                matrix = np.array([
                    [np.cos(theta/2), -np.sin(theta/2)],
                    [np.sin(theta/2), np.cos(theta/2)]
                ], dtype=np.complex64)
            else:  # rz
                matrix = np.array([
                    [np.exp(-1j*theta/2), 0],
                    [0, np.exp(1j*theta/2)]
                ], dtype=np.complex64)
            
            operations.append({
                'gate': gate_name,
                'matrix': matrix,
                'qubits': qubits
            })
        elif gate_name == 'measure':
            # Skip measurements for statevector simulation
            continue
        else:
            print(f"⚠️  Warning: Gate '{gate_name}' not yet supported, skipping")
    
    return operations

def simulate_with_cuquantum(circuit):
    """Simulate Qiskit circuit using cuQuantum"""
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")
    
    n_qubits = circuit.num_qubits
    state_size = 2 ** n_qubits
    
    # Initialize state on GPU
    state = cp.zeros(state_size, dtype=np.complex64)
    state[0] = 1.0
    
    # Create cuStateVec handle
    handle = cusv.create()
    
    # Convert circuit to operations
    operations = qiskit_to_cuquantum(circuit)
    
    # Apply operations
    for op in operations:
        cusv.apply_matrix(
            handle,
            state.data.ptr,
            cusv.cudaDataType.CUDA_C_32F,
            n_qubits,
            op['qubits'],
            op['matrix'].ctypes.data,
            cusv.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW,
            0
        )
    
    # Get final state
    final_state = state.get()
    
    cusv.destroy(handle)
    
    return final_state

def main():
    if not QISKIT_AVAILABLE:
        print("\n❌ This example requires Qiskit.")
        print("Install with: pip install qiskit")
        return
    
    if not GPU_AVAILABLE:
        print("\n❌ This example requires GPU support.")
        return
    
    print("="*70)
    print("  Framework Integration: Qiskit + cuQuantum")
    print("="*70)
    
    # Example 1: Simple Bell State Circuit
    print(f"\n{'='*70}")
    print("Example 1: Bell State Circuit")
    print("="*70)
    
    bell_circuit = QuantumCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    
    print("\n📊 Circuit:")
    print(bell_circuit)
    
    # Simulate with Qiskit
    start = time.time()
    qiskit_state = Statevector.from_instruction(bell_circuit)
    qiskit_time = time.time() - start
    
    # Simulate with cuQuantum
    start = time.time()
    cuquantum_state = simulate_with_cuquantum(bell_circuit)
    cuquantum_time = time.time() - start
    
    print(f"\n⏱️  Performance:")
    print(f"   Qiskit: {qiskit_time*1000:.3f} ms")
    print(f"   cuQuantum: {cuquantum_time*1000:.3f} ms")
    print(f"   Speedup: {qiskit_time/cuquantum_time:.2f}x")
    
    # Verify results match
    fidelity = abs(np.vdot(qiskit_state.data, cuquantum_state)) ** 2
    print(f"\n✅ State fidelity: {fidelity:.10f}")
    
    # Example 2: Quantum Fourier Transform
    print(f"\n{'='*70}")
    print("Example 2: Quantum Fourier Transform (n=6 qubits)")
    print("="*70)
    
    n_qubits = 6
    qft_circuit = QuantumCircuit(n_qubits)
    
    # Build QFT circuit manually (Qiskit's QFT uses unsupported gates)
    for i in range(n_qubits):
        qft_circuit.h(i)
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            # Use Rz and CNOT to implement controlled phase
            qft_circuit.cx(j, i)
            qft_circuit.rz(angle, i)
            qft_circuit.cx(j, i)
    
    print(f"\n📊 Circuit depth: {qft_circuit.depth()}")
    print(f"   Number of gates: {len(qft_circuit.data)}")
    
    # Simulate with Qiskit
    print("\n⚙️  Simulating with Qiskit...")
    start = time.time()
    qiskit_state = Statevector.from_instruction(qft_circuit)
    qiskit_time = time.time() - start
    
    # Simulate with cuQuantum
    print("⚙️  Simulating with cuQuantum...")
    start = time.time()
    cuquantum_state = simulate_with_cuquantum(qft_circuit)
    cuquantum_time = time.time() - start
    
    print(f"\n⏱️  Performance:")
    print(f"   Qiskit: {qiskit_time*1000:.3f} ms")
    print(f"   cuQuantum: {cuquantum_time*1000:.3f} ms")
    print(f"   Speedup: {qiskit_time/cuquantum_time:.2f}x")
    
    # Verify results
    fidelity = abs(np.vdot(qiskit_state.data, cuquantum_state)) ** 2
    print(f"\n✅ State fidelity: {fidelity:.10f}")
    
    # Example 3: Scaling Test
    print(f"\n{'='*70}")
    print("Example 3: Scaling Test - Performance vs Circuit Size")
    print("="*70)
    
    print(f"\n{'Qubits':<8} {'Gates':<8} {'Qiskit (ms)':<15} {'cuQuantum (ms)':<15} {'Speedup'}")
    print("-" * 70)
    
    for n in [4, 6, 8, 10]:
        # Create random circuit
        circuit = QuantumCircuit(n)
        np.random.seed(42)
        
        # Add random gates
        for _ in range(n * 5):
            gate_choice = np.random.choice(['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'])
            qubit = np.random.randint(0, n)
            
            if gate_choice in ['rx', 'ry', 'rz']:
                angle = np.random.uniform(0, 2*np.pi)
                getattr(circuit, gate_choice)(angle, qubit)
            else:
                getattr(circuit, gate_choice)(qubit)
        
        # Add some CNOTs
        for _ in range(n):
            q1, q2 = np.random.choice(n, size=2, replace=False)
            circuit.cx(q1, q2)
        
        num_gates = len(circuit.data)
        
        # Benchmark Qiskit
        try:
            start = time.time()
            qiskit_state = Statevector.from_instruction(circuit)
            qiskit_time = (time.time() - start) * 1000
        except MemoryError:
            qiskit_time = float('inf')
            print(f"{n:<8} {num_gates:<8} {'OOM':<15} ", end="")
        else:
            print(f"{n:<8} {num_gates:<8} {qiskit_time:<15.3f} ", end="")
        
        # Benchmark cuQuantum
        try:
            start = time.time()
            cuquantum_state = simulate_with_cuquantum(circuit)
            cuquantum_time = (time.time() - start) * 1000
            
            print(f"{cuquantum_time:<15.3f} ", end="")
            
            if qiskit_time != float('inf'):
                speedup = qiskit_time / cuquantum_time
                print(f"{speedup:.2f}x")
                
                # Verify
                fidelity = abs(np.vdot(qiskit_state.data, cuquantum_state)) ** 2
                if fidelity < 0.999:
                    print(f"   ⚠️  Warning: Low fidelity {fidelity:.6f}")
            else:
                print("N/A")
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*70}")
    print("🎉 Framework Integration Demo Complete!")
    print("="*70)
    
    print("""
✅ You learned:
   - How to use cuQuantum as a Qiskit backend
   - Converting Qiskit circuits to cuQuantum operations
   - GPU acceleration for Qiskit workflows
   - Performance comparison: GPU vs CPU simulation
   - Scaling benefits for larger circuits

🔬 Key Concepts:
   - Qiskit provides high-level quantum programming
   - cuQuantum provides low-level GPU acceleration
   - Combine both for best of both worlds
   - Gate set mapping between frameworks
   - State fidelity validation

📚 Use Cases:
   - Accelerate existing Qiskit projects
   - Prototype algorithms in Qiskit, run on GPU
   - Benchmark quantum algorithms at scale
   - Develop quantum software with familiar tools

🚀 Next Steps:
   - Try with Cirq (06_cirq_integration.py)
   - Explore PennyLane integration
   - Use cuTensorNet for very large systems
   - Build custom GPU-optimized circuits
    """)
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
