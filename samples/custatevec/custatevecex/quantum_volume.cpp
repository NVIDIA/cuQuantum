/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

//
// Quantum Volume Circuit Example
//
// This example demonstrates a quick performance check using quantum volume circuits:
// 1. Generate random quantum circuits with 2-qubit gates
// 2. Use SVUpdater for efficient gate fusion and application
// 3. Sample all qubits with 1024 shots to demonstrate execution
// 4. Show execution times across different circuit sizes for educational purposes
//

#include <custatevecEx.h>              // custatevecEx API
#include <cuda_runtime.h>              // CUDA runtime
#include "stateVectorConstruction.hpp" // State vector factory
#include <algorithm>                   // std::shuffle
#include <chrono>                      // std::chrono for CPU timer
#include <cmath>                       // M_PI
#include <complex>                     // std::complex<>
#include <cstdarg>                     // va_list, va_start, va_end
#include <cstdio>                      // printf
#include <cstring>                     // strcmp
#include <numeric>                     // std::iota
#include <random>                      // std::uniform_real_distribution<>
#include <stdlib.h>                    // exit()
#include <vector>                      // std::vector<>

// Use complex128 for state vector and matrix elements
typedef std::complex<double> ComplexType;

typedef custatevecExStateVectorDescriptor_t ExStateVector;
typedef custatevecExSVUpdaterDescriptor_t ExSVUpdater;
typedef custatevecExDictionaryDescriptor_t ExDictionary;

//
// Create SVUpdater (following estimate_pi.cpp style)
//
ExSVUpdater createSVUpdater(cudaDataType_t svDataType)
{
    ExDictionary svUpdaterConfig{nullptr};
    ExSVUpdater svUpdater{nullptr};

    ERRCHK(custatevecExConfigureSVUpdater(&svUpdaterConfig, svDataType, nullptr, 0));
    ERRCHK(custatevecExSVUpdaterCreate(&svUpdater, svUpdaterConfig, nullptr));
    ERRCHK(custatevecExDictionaryDestroy(svUpdaterConfig));

    return svUpdater;
}

//
// Generate random SU(4) unitary matrix using QR decomposition
//
void generateRandomTwoQubitGate(ComplexType matrix[16], std::mt19937& gen)
{
    // STEP 1: Generate Random Complex Matrix
    // Create 4x4 matrix with Gaussian-distributed complex entries
    // This ensures rotational invariance → uniform distribution over unitary group

    std::normal_distribution<double> gaussDist(0.0, 1.0);
    ComplexType A[16];

    for (int i = 0; i < 16; ++i)
    {
        double real = gaussDist(gen);
        double imag = gaussDist(gen);
        A[i] = ComplexType(real, imag);
    }

    // STEP 2: Gram-Schmidt Orthogonalization
    // Convert random matrix A into orthonormal matrix Q
    // For each column j: q_j = (a_j - Σ⟨q_i,a_j⟩q_i) / ||a_j - Σ⟨q_i,a_j⟩q_i||

    ComplexType Q[16];

    for (int j = 0; j < 4; ++j)
    {
        // Column j: Orthogonalize against all previous columns q_0, q_1, ..., q_{j-1}
        // Step 1: Remove projections onto all previous orthonormal vectors
        for (int i = 0; i < j; ++i)
        {
            // Compute projection: proj_i = ⟨q_i, a_j⟩
            ComplexType proj = ComplexType(0, 0);
            for (int row = 0; row < 4; ++row)
                proj += std::conj(Q[row * 4 + i]) * A[row * 4 + j]; // ⟨q_i, a_j⟩
            // Subtract projection: a_j' = a_j - proj_i * q_i
            for (int row = 0; row < 4; ++row)
                A[row * 4 + j] -= proj * Q[row * 4 + i];
        }

        // Step 2: Normalize the orthogonalized vector
        double norm = 0.0;
        for (int row = 0; row < 4; ++row)
            norm += std::norm(A[row * 4 + j]); // Compute ||a_j'||²
        norm = sqrt(norm);                     // ||a_j'||

        // q_j = a_j' / ||a_j'||
        for (int row = 0; row < 4; ++row)
            Q[row * 4 + j] = A[row * 4 + j] / norm; // Normalize column j
    }

    // STEP 3: Ensure Special Unitary (det = 1)
    // Compute determinant and scale last column to force det(Q) = 1
    // This converts from U(4) to SU(4)

    ComplexType det =
        Q[0] * (Q[5] * (Q[10] * Q[15] - Q[11] * Q[14]) - Q[6] * (Q[9] * Q[15] - Q[11] * Q[13]) +
                Q[7] * (Q[9] * Q[14] - Q[10] * Q[13])) -
        Q[1] * (Q[4] * (Q[10] * Q[15] - Q[11] * Q[14]) - Q[6] * (Q[8] * Q[15] - Q[11] * Q[12]) +
                Q[7] * (Q[8] * Q[14] - Q[10] * Q[12])) +
        Q[2] * (Q[4] * (Q[9] * Q[15] - Q[11] * Q[13]) - Q[5] * (Q[8] * Q[15] - Q[11] * Q[12]) +
                Q[7] * (Q[8] * Q[13] - Q[9] * Q[12])) -
        Q[3] * (Q[4] * (Q[9] * Q[14] - Q[10] * Q[13]) - Q[5] * (Q[8] * Q[14] - Q[10] * Q[12]) +
                Q[6] * (Q[8] * Q[13] - Q[9] * Q[12]));

    // Scale last column: Q_new = Q × diag(1,1,1,1/det)
    // Result: det(Q_new) = det(Q) × (1/det) = 1
    ComplexType detInv = ComplexType(1, 0) / det;
    for (int i = 0; i < 4; ++i)
        Q[i * 4 + 3] *= detInv; // Scale fourth column by 1/det

    // STEP 4: Copy Result to Output Matrix
    // Now Q is a proper SU(4) matrix: unitary and det(Q) = 1

    for (int i = 0; i < 16; ++i)
        matrix[i] = Q[i];
}

//
// Enqueue random quantum volume circuit using SVUpdater
//
void enqueueQuantumVolumeCircuit(ExSVUpdater svUpdater, int numWires, int depth, std::mt19937& gen)
{
    ComplexType matrix[16];
    const auto matrixDataType = CUDA_C_64F;

    std::vector<int> qubits(numWires);
    std::iota(qubits.begin(), qubits.end(), 0);

    for (int layer = 0; layer < depth; ++layer)
    {
        // Apply random 2-qubit gates to random qubit pairs

        // Shuffle qubits for random pairing
        std::shuffle(qubits.begin(), qubits.end(), gen);

        for (int gate = 0; gate < numWires / 2; ++gate)
        {
            // Pick shuffled qubit pair to avoid systematic bias
            int qubit0 = qubits[gate * 2];
            int qubit1 = qubits[gate * 2 + 1];

            generateRandomTwoQubitGate(matrix, gen);

            int32_t targets[] = {qubit0, qubit1};
            ERRCHK(custatevecExSVUpdaterEnqueueMatrix(
                svUpdater, matrix, matrixDataType, CUSTATEVEC_EX_MATRIX_DENSE,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, /*adjoint=*/0, targets, 2, nullptr, nullptr, 0));
        }
    }
}

//
// Execute quantum volume circuit
//
float runQuantumVolumeCircuit(int numWires, int depth, cudaDataType_t svDataType,
                              custatevecExDictionaryDescriptor_t svConfig)
{
    // Start timing for entire end-to-end execution
    auto start = std::chrono::high_resolution_clock::now();

    // Create state vector from provided configuration
    auto stateVector = createStateVector(svConfig);
    ERRCHK(custatevecExStateVectorSetZeroState(stateVector));

    // Create random number generator
    std::mt19937 gen(42); // Fixed seed for reproducibility

    // Create SVUpdater and enqueue circuit
    auto svUpdater = createSVUpdater(svDataType);
    enqueueQuantumVolumeCircuit(svUpdater, numWires, depth, gen);

    // Apply circuit using SVUpdater (with gate fusion)
    ERRCHK(custatevecExSVUpdaterApply(svUpdater, stateVector, nullptr, 0));

    // Sample all qubits with 1024 shots
    constexpr int numShots = 1024;
    std::vector<double> randnums(numShots);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < numShots; ++i)
    {
        randnums[i] = dist(gen);
    }

    std::vector<int32_t> allWires(numWires);
    std::iota(allWires.begin(), allWires.end(), 0);
    std::vector<custatevecIndex_t> bitStrings(numShots);

    ERRCHK(custatevecExSample(stateVector, bitStrings.data(), allWires.data(), numWires,
                              randnums.data(), numShots, CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER,
                              nullptr));
    ERRCHK(custatevecExStateVectorSynchronize(stateVector));

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float elapsedMs = duration.count() / 1000.0f;

    // Cleanup
    ERRCHK(custatevecExSVUpdaterDestroy(svUpdater));
    ERRCHK(custatevecExStateVectorDestroy(stateVector));

    return elapsedMs / 1000.0f; // Return time in seconds
}

int main(int argc, char* argv[])
{
    // Initialize multi-process environment
    bootstrapMultiProcessEnvironment(&argc, &argv);

    output("cuStateVec Ex Example: Quantum Volume Circuits\n");
    output("Demonstrating circuit execution with depth=30\n");

    std::vector<int> numWiresList = {20, 25, 29};
    const int depth = 30;
    const int nLoops = 5; // Number of runs per configuration

    // Store timing results for later output
    std::vector<float> medianTimes;

    for (auto numWires : numWiresList)
    {
        // Create svConfig for this specific numWires
        auto svConfig = configureStateVector(argc, argv, numWires);
        cudaDataType_t svDataType = getStateVectorDataType();

        std::vector<float> times;
        // Run multiple times for timing measurement
        for (int loop = 0; loop < nLoops; ++loop)
        {
            float executionTime = runQuantumVolumeCircuit(numWires, depth, svDataType, svConfig);
            times.push_back(executionTime);
        }

        // Compute median
        std::sort(times.begin(), times.end());
        float medianTime = times[nLoops / 2];

        // Save the medianTime
        medianTimes.push_back(medianTime);

        // Clean up configuration for this numWires
        ERRCHK(custatevecExDictionaryDestroy(svConfig));
    }

    // Generate table by using stored medianTime
    output("\n");

    // Output the execution times
    output("qubits | depth | gates | time(s)\n");
    output("-------|-------|-------|--------\n");
    for (size_t i = 0; i < numWiresList.size(); ++i)
    {
        int numWires = numWiresList[i];
        int numGates = depth * (numWires / 2);
        float medianTime = medianTimes[i];
        output("%6d | %5d | %5d | %7.3f\n", numWires, depth, numGates, medianTime);
    }
    output("\n");

    // Finalize multi-process environment
    finalizeMultiProcessEnvironment();

    printf("PASSED\n");
    return EXIT_SUCCESS;
}
