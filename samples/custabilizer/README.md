# cuStabilizer Demo

This sample demonstrates how to simulate a simple quantum circuit using cuStabilizer.

## Circuit Description

The demo simulates a Bell state preparation circuit:
```
Z_ERROR(0.3) 0   # Randomize z bits of the frame on qubit 0
H 0              # Swap X and Z bits on qubit 0
CNOT 0 1         # X_target ^= X_control
M 1 2            # Measure qubits 1 and 2
```

## Building

### Prerequisites

1. CUDA Toolkit (11.0 or later)
2. CMake (3.27.0 or later)
3. cuStabilizer

### Build the example

From this directory:
```bash
cmake -B build -S .
cmake --build build --verbose
./build/custabilizer_demo
```

#### CMake Package Discovery

This sample uses a `FindCuStabilizer.cmake` package that automatically searches for cuStabilizer in:
- The local distribution: `../{build/lib,include}`
- Environment variables: `CUSTABILIZER_ROOT`, `LD_LIBRARY_PATH`, `LIBRARY_PATH`, `CPATH`, etc.
- Standard system paths: `/usr/local/{lib,include}` and `/usr/{lib,include}`

If cuStabilizer is in a non-standard location, use environment variables:
```bash
export CUSTABILIZER_ROOT=/path/to/custabilizer
cmake -B build -S .
...
```

#### Using cuStabilizer in Your Own CMake Project

Copy `cmake/FindCuStabilizer.cmake` to your project and use:
```cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(CuStabilizer REQUIRED)
target_link_libraries(your_target PRIVATE cuStabilizer::cuStabilizer)
```


## Understanding the Output

- **X table**: Tracks X-type Pauli errors on each qubit across shots
- **Z table**: Tracks Z-type Pauli errors on each qubit across shots
- **Measurement table**: Shows measurement outcomes for each shot


