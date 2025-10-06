# Examples Directory

This directory contains working code examples for using cuQuantum in various scenarios.

## 📁 Structure

```
examples/
├── quick_start.py              # 5-minute getting started
├── quantum_algorithms/         # Common quantum algorithms
│   ├── qft_example.py         # Quantum Fourier Transform
│   ├── grover_search.py       # Grover's algorithm
│   ├── vqe_example.py         # Variational Quantum Eigensolver
│   └── qaoa_example.py        # QAOA for MaxCut
├── framework_integration/      # Integration examples
│   ├── qiskit_backend.py      # Using with Qiskit
│   ├── cirq_backend.py        # Using with Cirq
│   └── pennylane_backend.py   # Using with PennyLane
├── advanced/                   # Advanced topics
│   ├── multi_gpu.py           # Multi-GPU simulation
│   ├── custom_gates.py        # Custom gate implementations
│   └── noise_modeling.py      # Noise and decoherence
└── notebooks/                  # Jupyter notebooks
    ├── tutorial_01_basics.ipynb
    ├── tutorial_02_algorithms.ipynb
    └── tutorial_03_performance.ipynb
```

## 🚀 Quick Start Example

Run your first cuQuantum simulation:

```bash
python examples/quick_start.py
```

## 📚 Example Categories

### Beginner Examples
- `quick_start.py` - Your first quantum circuit
- Basic gate applications
- Simple measurements and sampling

### Intermediate Examples
- Quantum algorithms (QFT, Grover, etc.)
- Framework integrations
- Performance optimization

### Advanced Examples
- Multi-GPU programming
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
