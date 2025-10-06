# Setup & Installation Guide

Comprehensive guide for setting up cuQuantum on various systems.

## 🎯 Prerequisites Checklist

Before installing cuQuantum, ensure you have:

- [ ] **NVIDIA GPU** with Compute Capability 7.0 or higher
  - Supported: V100, A100, H100, RTX 30/40 series, T4, etc.
  - Check: `nvidia-smi` or visit [NVIDIA GPU Compatibility](https://developer.nvidia.com/cuda-gpus)

- [ ] **NVIDIA Driver**
  - CUDA 12: Driver version ≥ 525.60.13
  - CUDA 13: Driver version ≥ 580.65.06
  - Check: `nvidia-smi` (shows driver version)

- [ ] **CUDA Toolkit** (12.x or 13.x)
  - Check: `nvcc --version`
  - Download: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

- [ ] **Python** (3.11, 3.12, or 3.13)
  - Check: `python --version`
  - Download: [Python Downloads](https://www.python.org/downloads/)

## 🚀 Quick Installation (Recommended)

### Option 1: Conda (Easiest)

```bash
# Create new environment
conda create -n cuquantum python=3.12
conda activate cuquantum

# Install cuQuantum
conda install -c conda-forge cuquantum

# Verify installation
python -c "import cuquantum; print(f'cuQuantum {cuquantum.__version__} installed!')"
```

**Advantages:**
- ✅ Handles all dependencies automatically
- ✅ Includes CUDA libraries
- ✅ No manual CUDA configuration needed
- ✅ Easy to manage and update

### Option 2: pip (Simple)

```bash
# Create virtual environment
python -m venv cuquantum-env
source cuquantum-env/bin/activate  # On Windows: cuquantum-env\Scripts\activate

# Install cuQuantum for CUDA 12
pip install cuquantum-cu12

# Or for CUDA 11
pip install cuquantum-cu11

# Install optional dependencies
pip install qiskit cirq pennylane

# Verify installation
python -c "import cuquantum; print(f'cuQuantum {cuquantum.__version__} installed!')"
```

**Advantages:**
- ✅ Fast installation
- ✅ Works with existing Python setup
- ✅ pip-compatible workflow

**Note:** Ensure CUDA libraries are in your `LD_LIBRARY_PATH`

## 🔧 Advanced Installation

### Option 3: From Source (Development)

Perfect for contributing or customizing cuQuantum.

#### Step 1: System Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    nvidia-cuda-toolkit

# Check CUDA installation
nvcc --version
```

#### Step 2: Clone Repository

```bash
# Clone your fork
git clone https://github.com/khlaifiabilel/cuQuantum.git
cd cuQuantum

# Add upstream remote (for syncing)
git remote add upstream https://github.com/NVIDIA/cuQuantum.git
```

#### Step 3: Set Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc

# CUDA path
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Reload
source ~/.bashrc
```

#### Step 4: Create Virtual Environment

```bash
# Using venv
python -m venv cuquantum-dev
source cuquantum-dev/bin/activate

# Or using conda
conda create -n cuquantum-dev python=3.12
conda activate cuquantum-dev
```

#### Step 5: Install Build Dependencies

```bash
# Update pip
pip install --upgrade pip setuptools wheel

# Install build dependencies
pip install \
    cython>=3.0.4 \
    packaging \
    numpy>=1.21 \
    cupy-cuda12x  # or cupy-cuda11x for CUDA 11
```

#### Step 6: Install Python Package

```bash
# Navigate to Python directory
cd python

# Install in editable mode (for development)
pip install -e .

# Or regular installation
pip install .

# Verify
python -c "import cuquantum; print('Success!')"
```

#### Step 7: Install Benchmark Suite

```bash
# Navigate to benchmarks
cd ../benchmarks

# Install with all backends
pip install -e .[all]

# Or install specific backends
pip install -e .[qiskit]  # Just Qiskit
pip install -e .[cirq]    # Just Cirq

# Verify
nv-quantum-benchmarks --help
```

#### Step 8: Run Tests

```bash
# Python tests
cd ../python
pytest tests/ -v

# Benchmark tests
cd ../benchmarks
pytest tests/ -v
```

### Option 4: Docker (Isolated Environment)

Perfect for reproducible environments and deployment.

```bash
# Pull cuQuantum Appliance
docker pull nvcr.io/nvidia/cuquantum-appliance:latest

# Run container
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/cuquantum-appliance:latest

# Inside container, cuQuantum is pre-installed
python -c "import cuquantum; print(cuquantum.__version__)"
```

**Create Custom Dockerfile:**

```dockerfile
FROM nvcr.io/nvidia/cuquantum-appliance:latest

# Install additional packages
RUN pip install \
    jupyter \
    matplotlib \
    pandas

# Set working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

Build and run:

```bash
docker build -t cuquantum-jupyter .
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace cuquantum-jupyter
```

## 🔍 Troubleshooting

### Issue 1: CUDA_PATH Not Set

**Error:**
```
RuntimeError: CUDA is not found, please set $CUDA_PATH
```

**Solution:**
```bash
# Find CUDA installation
which nvcc
# Usually: /usr/local/cuda/bin/nvcc

# Set CUDA_PATH
export CUDA_PATH=/usr/local/cuda

# Add to ~/.bashrc for permanence
echo 'export CUDA_PATH=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc
```

### Issue 2: No GPU Detected

**Error:**
```
RuntimeError: No CUDA-capable device is detected
```

**Solution:**
```bash
# Check GPU
nvidia-smi

# If not found, check driver installation
lsmod | grep nvidia

# Reinstall NVIDIA driver if needed
sudo apt-get install nvidia-driver-535
```

### Issue 3: Version Mismatch

**Error:**
```
RuntimeError: CUDA version mismatch
```

**Solution:**
```bash
# Check CUDA version
nvcc --version

# Install matching cuQuantum version
pip install cuquantum-cu12  # For CUDA 12
# or
pip install cuquantum-cu11  # For CUDA 11
```

### Issue 4: Import Errors

**Error:**
```
ImportError: libcustatevec.so.1: cannot open shared object file
```

**Solution:**
```bash
# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or use conda (handles this automatically)
conda install -c conda-forge cuquantum
```

### Issue 5: Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Don't use sudo with pip (use virtual environment instead)
python -m venv myenv
source myenv/bin/activate
pip install cuquantum-python

# Or use user installation
pip install --user cuquantum-python
```

## ✅ Verification Checklist

Run these commands to verify your installation:

```bash
# 1. Check Python version
python --version
# Expected: Python 3.11, 3.12, or 3.13

# 2. Check CUDA
nvidia-smi
# Should show GPU info and CUDA version

# 3. Check cuQuantum
python -c "import cuquantum; print(cuquantum.__version__)"
# Should print version number

# 4. Check components
python -c "
from cuquantum import custatevec, cutensornet, cudensitymat
print('✓ All components imported successfully')
"

# 5. Check CuPy
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s) detected')"

# 6. Run quick test
python -c "
import cupy as cp
from cuquantum import custatevec as cusv
handle = cusv.create()
state = cp.zeros(2**10, dtype=cp.complex64)
state[0] = 1.0
print('✓ cuStateVec test passed')
cusv.destroy(handle)
"

# 7. Test benchmark suite (if installed)
nv-quantum-benchmarks --help

# 8. Run example
python examples/quick_start.py
```

If all checks pass: **🎉 Installation successful!**

## 📚 Post-Installation

### Set Up Development Environment

```bash
# Install development tools
pip install \
    pytest pytest-cov \
    black flake8 mypy \
    jupyter ipython \
    matplotlib seaborn \
    pandas

# Install optional frameworks
pip install \
    qiskit qiskit-aer \
    cirq cirq-core \
    pennylane
```

### Configure IDE

**VS Code:**
1. Install Python extension
2. Install Jupyter extension
3. Set Python interpreter to your virtual environment
4. Configure linter (flake8, black)

**PyCharm:**
1. File → Settings → Project → Python Interpreter
2. Select your virtual environment
3. Enable scientific mode for notebooks

### Run First Example

```bash
# Clone repository if not done
git clone https://github.com/khlaifiabilel/cuQuantum.git
cd cuQuantum

# Run quick start
python examples/quick_start.py

# Run benchmark
nv-quantum-benchmarks circuit \
    --frontend qiskit \
    --backend cutn \
    --benchmark qft \
    --nqubits 8 \
    --ngpus 1
```

## 🐳 Docker Compose (Multi-GPU Setup)

For multi-GPU setups, create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  cuquantum:
    image: nvcr.io/nvidia/cuquantum-appliance:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./:/workspace
    working_dir: /workspace
    command: bash
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Run:
```bash
docker-compose up -d
docker-compose exec cuquantum bash
```

## 🔄 Updating cuQuantum

### Conda

```bash
conda activate cuquantum
conda update -c conda-forge cuquantum
```

### pip

```bash
pip install --upgrade cuquantum-python
```

### From Source

```bash
cd cuQuantum
git pull origin main
cd python
pip install --upgrade -e .
```

## 🆘 Getting Help

If you're still having issues:

1. **Check Documentation**: [NVIDIA cuQuantum Docs](https://docs.nvidia.com/cuda/cuquantum/)
2. **Search Issues**: [GitHub Issues](https://github.com/NVIDIA/cuQuantum/issues)
3. **Ask Community**: [GitHub Discussions](https://github.com/NVIDIA/cuQuantum/discussions)
4. **Stack Overflow**: Tag `cuquantum`

## 📝 System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | Compute 7.0+ | A100, H100 |
| **Driver** | 525.60.13+ | Latest |
| **CUDA** | 12.x or 13.x | 12.x |
| **Python** | 3.11 | 3.12 |
| **RAM** | 16 GB | 64+ GB |
| **Storage** | 10 GB | 50+ GB |

---

**🎉 Ready to quantum compute at light speed!**

Next: [Quick Start Guide](../README.md#quick-start) • [Examples](../examples/README.md) • [Benchmarks](../benchmarks/README.md)
