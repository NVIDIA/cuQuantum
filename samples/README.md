# cuQuantum Samples

Comprehensive examples demonstrating cuQuantum's GPU-accelerated quantum computing capabilities.

## 📁 Directory Structure

```
samples/
├── python/              # Python examples (beginner → advanced)
├── cuda_cpp/            # C++/CUDA examples (high performance)
├── custatevec/          # cuStateVec API examples (state vector)
├── cutensornet/         # cuTensorNet API examples (tensor networks)
├── cudensitymat/        # cuDensityMat API examples (density matrices)
└── notebooks/           # Interactive Jupyter notebooks
```

## 🚀 Quick Start

### For Python Users
```bash
cd python/basic
python quick_start.py
```

### For C++ Users
```bash
cd cuda_cpp
make basic
./basic/01_bell_state
```

### For Interactive Learning
```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

## 📚 Sample Categories

### 🐍 Python Examples (`python/`)
Organized by difficulty level with clear learning paths.

- **basic/** - Quick start, quantum gates, Bell states
- **intermediate/** - QFT, Grover's algorithm
- **advanced/** - VQE, noise simulation, tensor networks
- **frameworks/** - Qiskit integration

**Requirements:** Python 3.8+, cuQuantum Python, CUDA-capable GPU

**See:** [`python/README.md`](python/README.md) for detailed guide.

---

### ⚡ C++/CUDA Examples (`cuda_cpp/`)
Native CUDA implementations for maximum performance.

- **basic/** - Bell state creation, QFT
- **advanced/** - Coming soon (Grover's, VQE, multi-GPU)

**Requirements:** CUDA Toolkit 12.0+, cuQuantum SDK, NVIDIA GPU

**See:** [`cuda_cpp/README.md`](cuda_cpp/README.md) for build instructions.

---

### 🔷 cuStateVec Examples (`custatevec/`)
Low-level state vector simulation API examples.

**custatevec/** - 20+ examples covering:
- Gate application and batched operations
- Measurement and sampling
- Expectation values
- Multi-GPU support
- Pauli operators

**custatevecex/** - Extended examples:
- Noise channels
- Quantum state initialization
- Custom index bit permutations

**See:** [`custatevec/custatevec/README.md`](custatevec/custatevec/README.md)

---

### 🔶 cuTensorNet Examples (`cutensornet/`)
Tensor network contraction and MPS/MPO examples.

**Main examples:**
- `tensornet_example.cu` - Basic tensor network contraction
- `tensornet_example_gradients.cu` - Gradient computation
- `tensornet_example_mpi.cu` - Multi-node MPI support

**high_level/** - Quantum circuit simulations:
- Amplitude computation
- Expectation values
- Marginal distributions
- MPS sampling

**approxTN/** - Approximate tensor network methods:
- Gate splitting
- MPS compression
- QR/SVD decomposition

**See:** [`cutensornet/README.md`](cutensornet/README.md)

---

### 🔸 cuDensityMat Examples (`cudensitymat/`)
Density matrix formalism for open quantum systems.

Examples covering:
- Operator action on density matrices
- Gradient computation
- Eigenspectrum calculation
- MPI-based distributed computation

**See:** [`cudensitymat/README.md`](cudensitymat/README.md)

---

### 📓 Notebooks (`notebooks/`)
Interactive Jupyter notebooks for hands-on learning.

- `01_getting_started.ipynb` - Introduction to cuQuantum Python

---

## 🎯 Learning Paths

### Path 1: Python Beginner
```
python/basic/quick_start.py
  ↓
python/basic/01_quantum_gates_basics.py
  ↓
python/basic/02_bell_states.py
  ↓
python/intermediate/03_qft_circuit.py
```

### Path 2: Performance Engineering
```
python/basic/quick_start.py
  ↓
cuda_cpp/basic/01_bell_state.cu
  ↓
custatevec/custatevec/gate_application.cu
  ↓
custatevec/custatevec/mgpu_sampler.cu
```

### Path 3: Algorithm Development
```
python/intermediate/04_grover_search.py
  ↓
python/advanced/05_vqe_chemistry.py
  ↓
cutensornet/high_level/expectation_example.cu
```

### Path 4: Tensor Networks
```
python/advanced/08_tensor_networks.py
  ↓
cutensornet/approxTN/mps_example.cu
  ↓
cutensornet/high_level/mps_sampling_example.cu
```

### Path 5: Noise and Open Systems
```
python/advanced/07_noise_simulation.py
  ↓
custatevecex/noise_channel.cpp
  ↓
cudensitymat/operator_action_example.cpp
```

---

## 🛠️ Requirements

### General
- NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, or newer)
- CUDA Toolkit 11.0+ (12.0+ recommended)
- cuQuantum SDK

### Python Examples
```bash
pip install cuquantum-python
```

### C++/CUDA Examples
```bash
# Ensure CUDA and cuQuantum SDK are in your path
export CUDA_PATH=/usr/local/cuda
export CUQUANTUM_ROOT=/path/to/cuquantum
```

---

## 🔧 Building Examples

### Python
No build required - just run!
```bash
python samples/python/basic/quick_start.py
```

### C++/CUDA
```bash
# Using provided Makefiles
cd samples/cuda_cpp
make basic

# Or individual examples
cd samples/custatevec/custatevec
make gate_application
./gate_application
```

---

## 📖 Documentation

- **API Reference:** https://docs.nvidia.com/cuda/cuquantum/
- **Python API:** https://docs.nvidia.com/cuda/cuquantum/python/
- **User Guide:** https://docs.nvidia.com/cuda/cuquantum/getting_started_guide.html

---

## 🐛 Troubleshooting

### GPU Not Found
```bash
nvidia-smi  # Check GPU availability
echo $CUDA_VISIBLE_DEVICES  # Check CUDA device visibility
```

### Import Errors (Python)
```bash
pip install --upgrade cuquantum-python
python -c "import cuquantum; print(cuquantum.__version__)"
```

### Compilation Errors (C++)
```bash
# Check CUDA installation
nvcc --version

# Check library paths
echo $LD_LIBRARY_PATH

# Add cuQuantum libraries
export LD_LIBRARY_PATH=$CUQUANTUM_ROOT/lib:$LD_LIBRARY_PATH
```

### Out of Memory
```bash
# Reduce problem size or use multi-GPU examples
# See mgpu_*.cu examples in custatevec/
```

---

## 💡 Tips

- **Start with Python** for rapid prototyping
- **Move to C++/CUDA** for production performance (3-10x speedup)
- **Use notebooks** for interactive exploration
- **Check example READMEs** in each subdirectory for specific details
- **Run on smaller problems first** to verify setup

---

## 📊 Performance Comparison

| Operation | Python | C++/CUDA | Speedup |
|-----------|--------|----------|---------|
| Bell state (2 qubits) | ~5 ms | ~0.5 ms | **10x** |
| QFT (10 qubits) | ~15 ms | ~5 ms | **3x** |
| Grover (15 qubits) | ~100 ms | ~25 ms | **4x** |
| VQE iteration | ~200 ms | ~50 ms | **4x** |

*Benchmarks on NVIDIA A100 GPU*

---

## 🤝 Contributing

Found a bug or want to add an example? See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for guidelines.

---

## 📝 License

See [`LICENSE`](../LICENSE) for details.

---

**Happy Quantum Computing! 🚀**
