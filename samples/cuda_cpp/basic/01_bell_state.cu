/**
 * Example 1: Bell State Creation with cuStateVec (CUDA C++)
 * 
 * Demonstrates:
 * - Initializing cuStateVec in CUDA C++
 * - Creating quantum state vectors on GPU
 * - Applying Hadamard and CNOT gates
 * - Creating Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
 * - Reading and displaying quantum state
 * 
 * Compilation:
 *   nvcc -o bell_state 01_bell_state.cu -lcustatevec -lcublas
 * 
 * Usage:
 *   ./bell_state
 */

#include <cuda_runtime.h>
#include <custatevec.h>
#include <cuComplex.h>
#include <stdio.h>
#include <math.h>

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

void printState(const cuDoubleComplex* state, int nQubits) {
    int stateSize = 1 << nQubits;
    
    printf("\nQuantum State:\n");
    printf("%-10s %-20s %-20s %-15s\n", "Basis", "Real", "Imag", "Probability");
    printf("--------------------------------------------------------------------\n");
    
    for (int i = 0; i < stateSize; i++) {
        double real = cuCreal(state[i]);
        double imag = cuCimag(state[i]);
        double prob = real * real + imag * imag;
        
        if (prob > 1e-10) {
            printf("|");
            for (int j = nQubits - 1; j >= 0; j--) {
                printf("%d", (i >> j) & 1);
            }
            printf("⟩    %+.6f          %+.6f          %.6f\n", real, imag, prob);
        }
    }
    printf("\n");
}

int main() {
    printf("====================================================================\n");
    printf("  Bell State Creation with cuStateVec (CUDA C++)\n");
    printf("====================================================================\n\n");
    
    // Setup
    const int nQubits = 2;
    const int stateSize = 1 << nQubits;
    
    // Initialize cuStateVec
    custatevecHandle_t handle;
    CUSTATEVEC_CHECK(custatevecCreate(&handle));
    
    printf("✓ cuStateVec initialized\n");
    printf("  Number of qubits: %d\n", nQubits);
    printf("  State vector size: %d\n", stateSize);
    
    // Allocate state vector on device
    cuDoubleComplex *d_state;
    size_t stateBytes = stateSize * sizeof(cuDoubleComplex);
    CUDA_CHECK(cudaMalloc(&d_state, stateBytes));
    
    // Initialize to |00⟩ on host
    cuDoubleComplex *h_state = (cuDoubleComplex*)malloc(stateBytes);
    for (int i = 0; i < stateSize; i++) {
        h_state[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    h_state[0] = make_cuDoubleComplex(1.0, 0.0);  // |00⟩
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_state, h_state, stateBytes, cudaMemcpyHostToDevice));
    
    printf("\n✓ Initial state |00⟩ created\n");
    
    // Define Hadamard gate: H = 1/√2 * [[1, 1], [1, -1]]
    cuDoubleComplex H[4];
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    H[0] = make_cuDoubleComplex(inv_sqrt2, 0.0);   // H[0,0]
    H[1] = make_cuDoubleComplex(inv_sqrt2, 0.0);   // H[0,1]
    H[2] = make_cuDoubleComplex(inv_sqrt2, 0.0);   // H[1,0]
    H[3] = make_cuDoubleComplex(-inv_sqrt2, 0.0);  // H[1,1]
    
    // Copy gate to device
    cuDoubleComplex *d_H;
    CUDA_CHECK(cudaMalloc(&d_H, 4 * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_H, H, 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Apply Hadamard to qubit 0
    int targets[] = {0};
    int nTargets = 1;
    
    printf("\n📊 Applying Hadamard gate to qubit 0...\n");
    
    CUSTATEVEC_CHECK(custatevecApplyMatrix(
        handle,
        d_state,
        CUDA_C_64F,
        nQubits,
        d_H,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,  // adjoint flag
        targets,
        nTargets,
        nullptr,  // controls
        nullptr,  // control bit values
        0,  // nControls
        CUSTATEVEC_COMPUTE_64F,
        nullptr,  // extra workspace
        0   // workspace size
    ));
    
    // Read state after Hadamard
    CUDA_CHECK(cudaMemcpy(h_state, d_state, stateBytes, cudaMemcpyDeviceToHost));
    printf("After H gate:");
    printState(h_state, nQubits);
    
    // Define CNOT gate: [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]
    cuDoubleComplex CNOT[16];
    for (int i = 0; i < 16; i++) {
        CNOT[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    CNOT[0] = make_cuDoubleComplex(1.0, 0.0);   // [0,0]
    CNOT[5] = make_cuDoubleComplex(1.0, 0.0);   // [1,1]
    CNOT[10] = make_cuDoubleComplex(1.0, 0.0);  // [2,2] -> becomes [2,3]
    CNOT[15] = make_cuDoubleComplex(1.0, 0.0);  // [3,3] -> becomes [3,2]
    
    // Actually for CNOT, correct indices:
    CNOT[10] = make_cuDoubleComplex(0.0, 0.0);
    CNOT[15] = make_cuDoubleComplex(0.0, 0.0);
    CNOT[11] = make_cuDoubleComplex(1.0, 0.0);  // [2,3]
    CNOT[14] = make_cuDoubleComplex(1.0, 0.0);  // [3,2]
    
    // Copy CNOT to device
    cuDoubleComplex *d_CNOT;
    CUDA_CHECK(cudaMalloc(&d_CNOT, 16 * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_CNOT, CNOT, 16 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Apply CNOT (control=0, target=1)
    int cnot_targets[] = {0, 1};
    
    printf("📊 Applying CNOT gate (control=0, target=1)...\n");
    
    CUSTATEVEC_CHECK(custatevecApplyMatrix(
        handle,
        d_state,
        CUDA_C_64F,
        nQubits,
        d_CNOT,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,
        cnot_targets,
        2,
        nullptr,
        nullptr,
        0,
        CUSTATEVEC_COMPUTE_64F,
        nullptr,
        0
    ));
    
    // Read final state
    CUDA_CHECK(cudaMemcpy(h_state, d_state, stateBytes, cudaMemcpyDeviceToHost));
    printf("Final Bell state |Φ⁺⟩:");
    printState(h_state, nQubits);
    
    printf("====================================================================\n");
    printf("🎉 Successfully created Bell state!\n");
    printf("   |Φ⁺⟩ = (|00⟩ + |11⟩)/√2\n");
    printf("====================================================================\n\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_CNOT));
    free(h_state);
    CUSTATEVEC_CHECK(custatevecDestroy(handle));
    
    printf("✓ Resources cleaned up\n\n");
    
    return 0;
}
