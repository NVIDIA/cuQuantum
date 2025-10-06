/**
 * Example 2: Quantum Fourier Transform with cuStateVec (CUDA C++)
 * 
 * Demonstrates:
 * - Building QFT circuit in CUDA C++
 * - Controlled phase rotation gates
 * - GPU kernel for efficient state preparation
 * - Performance measurement
 * 
 * Compilation:
 *   nvcc -o qft 02_qft.cu -lcustatevec -lcublas
 * 
 * Usage:
 *   ./qft [n_qubits]
 */

#include <cuda_runtime.h>
#include <custatevec.h>
#include <cuComplex.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUSTATEVEC_CHECK(call) \
    do { \
        custatevecStatus_t status = call; \
        if (status != CUSTATEVEC_STATUS_SUCCESS) { \
            fprintf(stderr, "cuStateVec error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void applyHadamard(custatevecHandle_t handle, cuDoubleComplex* d_state, 
                   int nQubits, int target, cuDoubleComplex* d_H) {
    int targets[] = {target};
    CUSTATEVEC_CHECK(custatevecApplyMatrix(
        handle, d_state, CUDA_C_64F, nQubits,
        d_H, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0, targets, 1, nullptr, nullptr, 0,
        CUSTATEVEC_COMPUTE_64F, nullptr, 0
    ));
}

void applyControlledPhase(custatevecHandle_t handle, cuDoubleComplex* d_state,
                          int nQubits, int control, int target, double angle,
                          cuDoubleComplex* d_gate_buffer) {
    // Create controlled phase gate: diag(1, 1, 1, e^(i*angle))
    cuDoubleComplex CP[16];
    for (int i = 0; i < 16; i++) {
        CP[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    CP[0] = make_cuDoubleComplex(1.0, 0.0);
    CP[5] = make_cuDoubleComplex(1.0, 0.0);
    CP[10] = make_cuDoubleComplex(1.0, 0.0);
    CP[15] = make_cuDoubleComplex(cos(angle), sin(angle));
    
    CUDA_CHECK(cudaMemcpy(d_gate_buffer, CP, 16 * sizeof(cuDoubleComplex), 
                         cudaMemcpyHostToDevice));
    
    int targets[] = {control, target};
    CUSTATEVEC_CHECK(custatevecApplyMatrix(
        handle, d_state, CUDA_C_64F, nQubits,
        d_gate_buffer, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0, targets, 2, nullptr, nullptr, 0,
        CUSTATEVEC_COMPUTE_64F, nullptr, 0
    ));
}

void applyQFT(custatevecHandle_t handle, cuDoubleComplex* d_state, int nQubits,
              cuDoubleComplex* d_H, cuDoubleComplex* d_gate_buffer) {
    printf("\n📊 Applying QFT circuit...\n");
    
    for (int i = 0; i < nQubits; i++) {
        // Hadamard on qubit i
        applyHadamard(handle, d_state, nQubits, i, d_H);
        
        // Controlled phase rotations
        for (int j = i + 1; j < nQubits; j++) {
            double angle = M_PI / (1 << (j - i));
            applyControlledPhase(handle, d_state, nQubits, j, i, angle, d_gate_buffer);
        }
        
        if ((i + 1) % 3 == 0 || i == nQubits - 1) {
            printf("  ✓ Processed qubit %d/%d\n", i + 1, nQubits);
        }
    }
    
    printf("  ✓ QFT circuit complete\n\n");
}

void printAmplitudes(const cuDoubleComplex* state, int nQubits, int maxShow) {
    int stateSize = 1 << nQubits;
    int showCount = (stateSize < maxShow) ? stateSize : maxShow;
    
    printf("First %d amplitudes:\n", showCount);
    printf("%-10s %-15s %-15s %-15s\n", "Index", "Real", "Imag", "Magnitude");
    printf("----------------------------------------------------------\n");
    
    for (int i = 0; i < showCount; i++) {
        double real = cuCreal(state[i]);
        double imag = cuCimag(state[i]);
        double mag = sqrt(real * real + imag * imag);
        printf("%-10d %+.8f  %+.8f  %.8f\n", i, real, imag, mag);
    }
    if (stateSize > maxShow) {
        printf("... (%d more states)\n", stateSize - maxShow);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    int nQubits = 5;  // Default
    if (argc > 1) {
        nQubits = atoi(argv[1]);
        if (nQubits < 2 || nQubits > 20) {
            fprintf(stderr, "Error: n_qubits must be between 2 and 20\n");
            return 1;
        }
    }
    
    printf("====================================================================\n");
    printf("  Quantum Fourier Transform with cuStateVec (CUDA C++)\n");
    printf("====================================================================\n\n");
    
    const int stateSize = 1 << nQubits;
    printf("Configuration:\n");
    printf("  Qubits: %d\n", nQubits);
    printf("  State size: %d\n", stateSize);
    printf("  Memory: %.2f MB\n\n", (stateSize * sizeof(cuDoubleComplex)) / (1024.0 * 1024.0));
    
    // Initialize cuStateVec
    custatevecHandle_t handle;
    CUSTATEVEC_CHECK(custatevecCreate(&handle));
    
    // Allocate state vector
    cuDoubleComplex *d_state;
    size_t stateBytes = stateSize * sizeof(cuDoubleComplex);
    CUDA_CHECK(cudaMalloc(&d_state, stateBytes));
    
    // Initialize to |0...0⟩
    cuDoubleComplex *h_state = (cuDoubleComplex*)malloc(stateBytes);
    for (int i = 0; i < stateSize; i++) {
        h_state[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    h_state[0] = make_cuDoubleComplex(1.0, 0.0);
    
    CUDA_CHECK(cudaMemcpy(d_state, h_state, stateBytes, cudaMemcpyHostToDevice));
    
    // Allocate gate matrices
    cuDoubleComplex *d_H, *d_gate_buffer;
    CUDA_CHECK(cudaMalloc(&d_H, 4 * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_gate_buffer, 16 * sizeof(cuDoubleComplex)));
    
    // Hadamard matrix
    cuDoubleComplex H[4];
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    H[0] = make_cuDoubleComplex(inv_sqrt2, 0.0);
    H[1] = make_cuDoubleComplex(inv_sqrt2, 0.0);
    H[2] = make_cuDoubleComplex(inv_sqrt2, 0.0);
    H[3] = make_cuDoubleComplex(-inv_sqrt2, 0.0);
    CUDA_CHECK(cudaMemcpy(d_H, H, 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Apply QFT
    clock_t start = clock();
    applyQFT(handle, d_state, nQubits, d_H, d_gate_buffer);
    cudaDeviceSynchronize();
    clock_t end = clock();
    
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    
    // Read results
    CUDA_CHECK(cudaMemcpy(h_state, d_state, stateBytes, cudaMemcpyDeviceToHost));
    
    printf("====================================================================\n");
    printf("📊 Results\n");
    printf("====================================================================\n\n");
    
    printAmplitudes(h_state, nQubits, 10);
    
    printf("Performance:\n");
    printf("  Execution time: %.3f ms\n", elapsed);
    printf("  Gates applied: %d Hadamards + %d controlled phases\n", 
           nQubits, (nQubits * (nQubits - 1)) / 2);
    printf("\n");
    
    printf("====================================================================\n");
    printf("🎉 QFT Complete!\n");
    printf("====================================================================\n\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_gate_buffer));
    free(h_state);
    CUSTATEVEC_CHECK(custatevecDestroy(handle));
    
    return 0;
}
