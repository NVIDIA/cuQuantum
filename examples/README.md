# cuQuantum Examples

Comprehensive collection of examples and tutorials for learning and using NVIDIA cuQuantum.

## 📁 Directory Structure

```
examples/
├── README.md                          # This file
├── quick_start.py                     # Simple introduction
├── 01_quantum_gates_basics.py         # Fundamental quantum gates
├── 02_bell_states.py                  # Quantum entanglement
├── 03_qft_circuit.py                  # Quantum Fourier Transform
├── 04_grover_search.py                # Grover's search algorithm
├── 05_vqe_chemistry.py                # Variational Quantum Eigensolver
├── 06_qiskit_integration.py           # Qiskit framework integration
├── 07_noise_simulation.py             # Realistic NISQ simulation
├── 08_tensor_networks.py              # Advanced MPS/tensor networks
└── notebooks/
    ├── 01_getting_started.ipynb       # Interactive tutorial
    └── ... (more notebooks)
```

## 🚀 Quick Start

### Option 1: Run Example Scripts

```bash
# Start with the basics
python quick_start.py

# Learn quantum gates
python 01_quantum_gates_basics.py

# Explore quantum algorithms
python 04_grover_search.py
```

### Option 2: Interactive Notebooks

```bash
# Launch Jupyter
jupyter notebook notebooks/01_getting_started.ipynb
```

## 📚 Example Categories

### Beginner Level

#### **quick_start.py** - Your First Quantum Circuit
- Creating quantum states on GPU
- Applying basic gates (H, CNOT)
- Creating Bell states
- Understanding cuQuantum API

**What you'll learn:** cuQuantum setup, state vectors, basic gates, GPU memory management

**Run time:** ~1 second | **GPU required:** Yes

---

#### **01_quantum_gates_basics.py** - Quantum Gates Deep Dive
- Complete gate library (Pauli, Hadamard, Phase, CNOT, Toffoli)
- Rotation gates (RX, RY, RZ)
- Measurement simulation
- Gate matrix representations

**What you'll learn:** How quantum gates transform states, single vs multi-qubit gates, unitary operations

**Run time:** ~2 seconds | **GPU required:** No (conceptual)

---

### Intermediate Level

#### **02_bell_states.py** - Quantum Entanglement
- Creating all 4 Bell states (Φ⁺, Φ⁻, Ψ⁺, Ψ⁻)
- Measurement correlations
- Entanglement verification
- Statistical analysis

**What you'll learn:** Quantum entanglement, non-local correlations, EPR pairs, measurement statistics

**Run time:** ~3 seconds | **GPU required:** Yes

---

#### **03_qft_circuit.py** - Quantum Fourier Transform
- QFT implementation from scratch
- Controlled phase rotations
- Comparison with classical FFT
- Inverse QFT

**What you'll learn:** Fourier analysis on quantum computers, phase estimation, period finding

**Run time:** ~5 seconds | **GPU required:** Yes

---

#### **06_qiskit_integration.py** - Framework Integration
- Using cuQuantum as Qiskit backend
- Circuit conversion
- Performance benchmarking
- Scaling analysis

**What you'll learn:** Qiskit → cuQuantum workflow, GPU acceleration, multi-framework development

**Run time:** ~10 seconds | **GPU required:** Yes | **Prerequisites:** `pip install qiskit`

---

### Advanced Level

#### **04_grover_search.py** - Grover's Search Algorithm
- Unstructured search with quadratic speedup
- Oracle implementation
- Amplitude amplification
- Optimal iteration calculation

**What you'll learn:** Quantum search algorithms, oracle design, quantum vs classical complexity

**Run time:** ~8 seconds | **GPU required:** Yes

---

#### **05_vqe_chemistry.py** - Variational Quantum Eigensolver
- H₂ molecule ground state energy
- Parameterized quantum circuits (ansatz)
- Classical optimization loop
- Chemical accuracy

**What you'll learn:** Hybrid quantum-classical algorithms, molecular Hamiltonians, NISQ applications

**Run time:** ~30 seconds | **GPU required:** Yes

---

#### **07_noise_simulation.py** - Realistic NISQ Devices
- Depolarizing noise channels
- Amplitude damping (T1 decay)
- Zero-noise extrapolation
- Error mitigation

**What you'll learn:** Quantum noise models, decoherence effects, error mitigation techniques

**Run time:** ~15 seconds | **GPU required:** Yes

---

#### **08_tensor_networks.py** - Large-Scale Simulation
- Matrix Product State (MPS) simulation
- 30+ qubit systems
- Memory-efficient computation
- Bond dimension optimization

**What you'll learn:** Tensor network representations, scaling beyond state vectors, cuTensorNet API

**Run time:** ~20 seconds | **GPU required:** Yes

---

## 🎯 Learning Paths

### Path 1: Complete Beginner
```
quick_start.py → 01_quantum_gates_basics.py → 02_bell_states.py
→ 03_qft_circuit.py → notebooks/01_getting_started.ipynb
```

### Path 2: Algorithm Developer
```
quick_start.py → 04_grover_search.py → 05_vqe_chemistry.py
→ 06_qiskit_integration.py
```

### Path 3: NISQ Researcher
```
02_bell_states.py → 05_vqe_chemistry.py → 07_noise_simulation.py
→ 08_tensor_networks.py
```

### Path 4: Performance Engineer
```
quick_start.py → 06_qiskit_integration.py → 08_tensor_networks.py
```

## 📊 Feature Comparison

| Example | Qubits | Gates | GPU Required | Difficulty | Key Algorithm |
|---------|--------|-------|--------------|------------|---------------|
| quick_start.py | 2 | 2 | Yes | ⭐ | Bell state |
| 01_quantum_gates_basics.py | 1-3 | 10+ | No | ⭐ | Gate tutorial |
| 02_bell_states.py | 2 | 2-4 | Yes | ⭐⭐ | Entanglement |
| 03_qft_circuit.py | 3-5 | 15+ | Yes | ⭐⭐ | QFT |
| 04_grover_search.py | 3-5 | ~20 | Yes | ⭐⭐⭐ | Grover's |
| 05_vqe_chemistry.py | 2 | 4+ | Yes | ⭐⭐⭐ | VQE |
| 06_qiskit_integration.py | 4-10 | varies | Yes | ⭐⭐ | Qiskit backend |
| 07_noise_simulation.py | 2 | 2+ | Yes | ⭐⭐⭐ | Error mitigation |
| 08_tensor_networks.py | 20-30 | 100+ | Yes | ⭐⭐⭐⭐ | MPS |

## 🛠️ Requirements

### Minimum
- Python 3.11+
- NVIDIA GPU (Compute Capability 7.0+)
- CUDA 12.0+
- cuQuantum SDK
- CuPy

### Optional
- Qiskit (for 06_qiskit_integration.py)
- Jupyter (for notebooks)
- Matplotlib (for visualizations)

See [Setup Guide](../SETUP_GUIDE.md) for detailed installation.

## 🎓 Educational Use

All examples are designed with education in mind:
- ✅ Extensive inline documentation
- ✅ Step-by-step explanations
- ✅ Visual output and progress tracking
- ✅ Educational summaries at the end
- ✅ References to further reading

## 🤝 Contributing Examples

Want to add an example? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

**Suggested topics:**
- QAOA (Quantum Approximate Optimization)
- Quantum phase estimation
- Shor's factoring algorithm
- Quantum machine learning
- Custom noise models
- Multi-GPU examples
- Custom CUDA kernels
- Production deployments

## 📖 Usage

Each example is self-contained and includes:
- **Description**: What the example demonstrates
- **Prerequisites**: Required packages
- **Usage**: How to run it
- **Expected Output**: What you should see
- **Explanation**: Line-by-line code explanation

## 🔧 Setup

```bash
# Install cuQuantum
conda install -c conda-forge cuquantum

# Or with pip
pip install cuquantum-python

# Install example dependencies
pip install -r examples/requirements.txt
```

## 📝 Contributing Examples

Have a great example? We'd love to include it!

1. Create a new file following the template
2. Add clear documentation
3. Include expected output
4. Test on multiple systems
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## 🆘 Getting Help

- Check the [main README](../README.md)
- Review [cuQuantum documentation](https://docs.nvidia.com/cuda/cuquantum/)
- Ask on [GitHub Discussions](https://github.com/NVIDIA/cuQuantum/discussions)

---

*Coming Soon: More examples will be added regularly!*
