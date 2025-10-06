# Python Examples

Python examples for cuQuantum, organized by difficulty level.

## 📁 Structure

```
python/
├── basic/                  # Beginner-friendly examples
├── intermediate/           # Intermediate algorithms
├── advanced/              # Advanced techniques
└── frameworks/            # Framework integrations
```

## 🚀 Quick Start

### Basic Examples (Start Here!)
```bash
cd basic
python quick_start.py
python 01_quantum_gates_basics.py
python 02_bell_states.py
```

### Intermediate Examples
```bash
cd intermediate
python 03_qft_circuit.py
python 04_grover_search.py
```

### Advanced Examples
```bash
cd advanced
python 05_vqe_chemistry.py
python 07_noise_simulation.py
python 08_tensor_networks.py
```

### Framework Integration
```bash
cd frameworks
python 06_qiskit_integration.py  # Requires: pip install qiskit
```

## 📚 Examples by Category

### Basic Level 🌟
| File | Description | GPU Required | Lines |
|------|-------------|--------------|-------|
| `quick_start.py` | Your first quantum circuit | Yes | ~150 |
| `01_quantum_gates_basics.py` | Complete gate tutorial | No | ~250 |
| `02_bell_states.py` | Quantum entanglement | Yes | ~280 |

**Prerequisites:** Basic Python, basic quantum mechanics

---

### Intermediate Level 🌟🌟
| File | Description | GPU Required | Lines |
|------|-------------|--------------|-------|
| `03_qft_circuit.py` | Quantum Fourier Transform | Yes | ~270 |
| `04_grover_search.py` | Grover's search algorithm | Yes | ~320 |

**Prerequisites:** Understanding of quantum gates, algorithms

---

### Advanced Level 🌟🌟🌟
| File | Description | GPU Required | Lines |
|------|-------------|--------------|-------|
| `05_vqe_chemistry.py` | Variational eigensolver | Yes | ~400 |
| `07_noise_simulation.py` | NISQ device simulation | Yes | ~370 |
| `08_tensor_networks.py` | MPS for large systems | Yes | ~450 |

**Prerequisites:** Advanced algorithms, optimization, tensor networks

---

### Framework Integration 🔌
| File | Description | GPU Required | Dependencies |
|------|-------------|--------------|--------------|
| `06_qiskit_integration.py` | Qiskit backend | Yes | `pip install qiskit` |

---

## 🎯 Learning Paths

### Path 1: Complete Beginner
```
basic/quick_start.py
→ basic/01_quantum_gates_basics.py
→ basic/02_bell_states.py
→ intermediate/03_qft_circuit.py
```

### Path 2: Algorithm Developer
```
basic/quick_start.py
→ intermediate/04_grover_search.py
→ advanced/05_vqe_chemistry.py
→ frameworks/06_qiskit_integration.py
```

### Path 3: NISQ Researcher
```
basic/02_bell_states.py
→ advanced/05_vqe_chemistry.py
→ advanced/07_noise_simulation.py
→ advanced/08_tensor_networks.py
```

## 🛠️ Requirements

```bash
# Core dependencies
pip install cuquantum-python cupy-cuda12x numpy

# Optional (for specific examples)
pip install qiskit matplotlib jupyter
```

## 💡 Tips

1. **Start with basic/** - Even if experienced, review basics first
2. **GPU not available?** - Try `01_quantum_gates_basics.py` (CPU only)
3. **Modify parameters** - Experiment with qubit counts, gate sequences
4. **Read comments** - Every example has detailed inline documentation

## 🐛 Troubleshooting

**ImportError: No module named 'cuquantum'**
```bash
pip install cuquantum-python
```

**CUDA errors**
- Check GPU availability: `nvidia-smi`
- Verify CUDA installation: `nvcc --version`
- Ensure cuQuantum SDK is installed

**Out of memory**
- Reduce qubit count
- Use tensor network examples (08_tensor_networks.py)
- Close other GPU applications

## 📖 Next Steps

After completing Python examples:
- Explore [Jupyter notebooks](../notebooks/)
- Try [C++/CUDA examples](../cuda_cpp/)
- Read [cuQuantum documentation](https://docs.nvidia.com/cuda/cuquantum/)

## 🤝 Contributing

Have an example idea? See [CONTRIBUTING.md](../../../CONTRIBUTING.md)
