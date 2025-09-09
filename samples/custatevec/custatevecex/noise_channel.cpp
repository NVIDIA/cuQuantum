/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//
// This example demonstrates the StateVectorUpdater (SVUpdater) workflow: queuing operators once and
// applying them repeatedly with different random numbers.
// Using custatevecExSVUpdaterClear() to reset the operator queue for new circuit configurations.
//
// Features GHZ entangled states of different sizes to show noise scaling effects:
// 1. 2-qubit GHZ (Bell state): Baseline for comparison
// 2. Multi-qubit GHZ (4-25 qubits): Show exponential fragility scaling
//
// Noise experiment workflow:
// 1. Queue operators once (expensive setup)
// 2. Apply multiple times with different random numbers (efficient reuse)
// 3. Clear the queue (reset for new circuit)
// 4. Repeat with different circuit configurations
// 5. Scale up to larger systems to show noise accumulation effects
//
// Key insight: Larger entangled systems are exponentially more fragile to noise,
// demonstrating why quantum error correction is essential for practical quantum computing.
//
// APIs utilized:
// - custatevecExSVUpdaterEnqueueMatrix: Build quantum circuit (with anti-diagonal optimization)
// - custatevecExSVUpdaterApply: Execute circuit (called multiple times)
// - custatevecExSVUpdaterClear: Reset queue for new circuit
// - custatevecExSVUpdaterEnqueueUnitaryChannel: Add probabilistic noise
// - custatevecExSVUpdaterEnqueueGeneralChannel: Add general quantum channels
// - custatevecExSVUpdaterGetMaxNumRequiredRandnums: Get random number requirements
//
#include <custatevecEx.h> // custatevecEx API
#include <stdlib.h>       // exit()
#include <cstdio>         // printf
#include <complex>        // std::complex<>
#include <vector>         // std::vector<>
#include <random>         // std::mt19937, std::uniform_real_distribution
#include <algorithm>      // std::max_element
#include <cmath>          // M_PI
#include <cstring>        // strcmp
#include <cstdarg>        // va_list, va_start, va_end

// Global variable for output control
static bool output_enabled = true;

// Output function that respects output control
void output(const char* format, ...)
{
    if (!output_enabled)
        return;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

// Output function for single characters that respects output control
void output_char(char c)
{
    if (!output_enabled)
        return;
    putchar(c);
    fflush(stdout);
}

#if 1
// Use complex128 for state vector
constexpr cudaDataType_t svDataType = CUDA_C_64F;
typedef std::complex<double> ComplexType;
#else
// Use complex64 for state vector
constexpr cudaDataType_t svDataType = CUDA_C_32F;
typedef std::complex<float> ComplexType;
#endif

// Matrix elements always use double precision
typedef std::complex<double> DblComplex;

#define ERRCHK(s)                                                                                  \
    {                                                                                              \
        auto status = (s);                                                                         \
        if (status != CUSTATEVEC_STATUS_SUCCESS)                                                   \
        {                                                                                          \
            printf("%s, %s\n", custatevecGetErrorName(status), #s);                                \
            exit(EXIT_FAILURE); /* Error exit skips resource cleanup for simplicity */             \
        }                                                                                          \
    }

typedef custatevecExStateVectorDescriptor_t ExStateVector;
typedef custatevecExSVUpdaterDescriptor_t ExSVUpdater;

//
// Create a state vector with specified number of qubits
//
ExStateVector createStateVector(int numWires)
{
    custatevecExDictionaryDescriptor_t svConfig;
    ERRCHK(custatevecExConfigureStateVectorSingleDevice(&svConfig, svDataType, numWires, numWires,
                                                        /*deviceId=*/0, /*capability=*/0));

    ExStateVector stateVector;
    ERRCHK(custatevecExStateVectorCreateSingleProcess(&stateVector, svConfig, nullptr, 0, nullptr));

    ERRCHK(custatevecExDictionaryDestroy(svConfig));
    return stateVector;
}

//
// Create SVUpdater for repeated operations
//
ExSVUpdater createSVUpdater()
{
    custatevecExDictionaryDescriptor_t svUpdaterConfig;
    // Use the same data type as the state vector
    ERRCHK(custatevecExConfigureSVUpdater(&svUpdaterConfig, svDataType, nullptr, 0));

    ExSVUpdater svUpdater;
    ERRCHK(custatevecExSVUpdaterCreate(&svUpdater, svUpdaterConfig, nullptr));

    ERRCHK(custatevecExDictionaryDestroy(svUpdaterConfig));
    return svUpdater;
}

//
// Enqueue GHZ state preparation circuit: H(0) -> CNOT(0,1) -> CNOT(1,2) -> ... -> CNOT(n-2,n-1)
// For numWires=2, this creates a Bell state (special case of GHZ)
//
void enqueueGHZCircuit(ExSVUpdater svUpdater, int numWires)
{
    // Hadamard gate on qubit 0
    DblComplex H[4] = {
        {1.0 / sqrt(2), 0}, {1.0 / sqrt(2), 0}, {1.0 / sqrt(2), 0}, {-1.0 / sqrt(2), 0}};
    int32_t h_target = 0;
    ERRCHK(custatevecExSVUpdaterEnqueueMatrix(svUpdater, H, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DENSE,
                                              CUSTATEVEC_MATRIX_LAYOUT_ROW,
                                              /*adjoint=*/0, &h_target, 1, nullptr, nullptr, 0));

    // CNOT cascade: 0->1, 1->2, 2->3, ..., (n-2)->(n-1)
    // CNOT gates are added as anti-diagonal matrix with a control.
    DblComplex X_anti_diag[2] = {{1, 0}, {1, 0}};
    for (int i = 0; i < numWires - 1; ++i)
    {
        int32_t cnot_control = i;
        int32_t cnot_target = i + 1;
        int32_t cnot_control_bit = 1; // Apply X when control is |1>
        ERRCHK(custatevecExSVUpdaterEnqueueMatrix(
            svUpdater, X_anti_diag, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL,
            CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /*adjoint=*/0, &cnot_target, 1, &cnot_control, &cnot_control_bit, 1));
    }
}

//
// Enqueue depolarizing noise on all qubits
//
void enqueueDepolarizingNoise(ExSVUpdater svUpdater, double errorRate, int numWires)
{
    // Depolarizing channel: (1-epsilon)I + (epsilon/3)(X + Y + Z)
    const int numMatrices = 4; // I, X, Y, Z Pauli matrices
    double p_identity = 1.0 - errorRate;
    double p_pauli = errorRate / 3.0;
    double probabilities[numMatrices] = {p_identity, p_pauli, p_pauli, p_pauli};

    // Define Pauli matrices using efficient representations
    // I and Z are diagonal matrices - store only diagonal elements
    DblComplex I_diag[2] = {{1, 0}, {1, 0}};  // [1, 1]
    DblComplex Z_diag[2] = {{1, 0}, {-1, 0}}; // [1, -1]

    // X and Y are anti-diagonal matrices - store only anti-diagonal elements
    DblComplex X_antidiag[2] = {{1, 0}, {1, 0}};  // [1, 1] (off-diagonal)
    DblComplex Y_antidiag[2] = {{0, -1}, {0, 1}}; // [-i, i] (off-diagonal)

    const void* unitaries[numMatrices] = {I_diag, X_antidiag, Y_antidiag, Z_diag};
    custatevecExMatrixType_t matrixTypes[numMatrices] = {
        CUSTATEVEC_EX_MATRIX_DIAGONAL, CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL,
        CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL, CUSTATEVEC_EX_MATRIX_DIAGONAL};

    // Apply depolarizing noise to each qubit
    for (int wire = 0; wire < numWires; ++wire)
    {
        int32_t wires[1] = {wire};
        ERRCHK(custatevecExSVUpdaterEnqueueUnitaryChannel(
            svUpdater, unitaries, CUDA_C_64F, matrixTypes, numMatrices,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, probabilities, wires, 1));
    }
}

//
// Enqueue amplitude damping noise (T1 decay)
//
void enqueueAmplitudeDampingNoise(ExSVUpdater svUpdater, double gamma, int numWires)
{
    // Amplitude damping Kraus operators:
    // K0 = [[1, 0], [0, sqrt(1-gamma)]]
    // K1 = [[0, sqrt(gamma)], [0, 0]]
    // Note: These Kraus operators are defined as dense matrices in this sample.
    // Alternatively, K0 could use CUSTATEVEC_EX_MATRIX_DIAGONAL and K1 could use
    // CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL.
    DblComplex K0[4] = {{1, 0}, {0, 0}, {0, 0}, {sqrt(1.0 - gamma), 0}};
    DblComplex K1[4] = {{0, 0}, {sqrt(gamma), 0}, {0, 0}, {0, 0}};

    const void* krausOps[2] = {K0, K1};
    custatevecExMatrixType_t matrixTypes[2] = {CUSTATEVEC_EX_MATRIX_DENSE,
                                               CUSTATEVEC_EX_MATRIX_DENSE};

    // Apply amplitude damping to each qubit
    for (int wire = 0; wire < numWires; ++wire)
    {
        int32_t wires[1] = {wire};
        ERRCHK(
            custatevecExSVUpdaterEnqueueGeneralChannel(svUpdater, krausOps, CUDA_C_64F, matrixTypes,
                                                       2, CUSTATEVEC_MATRIX_LAYOUT_ROW, wires, 1));
    }
}

//
// Measure fidelity with ideal GHZ state: (|00...0> + |11...1>)/sqrt(2)
// Optimized: Only check the 2 non-zero amplitudes instead of creating full target vector
//
double measureGHZFidelity(ExStateVector stateVector, int numWires)
{
    const auto numElements = 1LL << numWires;
    const auto lastIndex = numElements - 1;

    // Get only the two relevant amplitudes: |00...0> and |11...1>
    ComplexType amplitude_00(0.0, 0.0); // |00...0> amplitude
    ComplexType amplitude_11(0.0, 0.0); // |11...1> amplitude

    // Get amplitude at index 0 (|00...0>)
    ERRCHK(custatevecExStateVectorGetState(stateVector, &amplitude_00, svDataType, 0, 1, 1));
    // Get amplitude at index (2^n - 1) (|11...1>)
    ERRCHK(custatevecExStateVectorGetState(stateVector, &amplitude_11, svDataType, lastIndex,
                                           numElements, 1));
    ERRCHK(custatevecExStateVectorSynchronize(stateVector));

    // GHZ target state: (|00...0> + |11...1>)/sqrt(2)
    // Fidelity = |<psi_target|psi_actual>|^2 = |(amplitude_00 + amplitude_11)/sqrt(2)|^2
    const double sqrt2_inv = 1.0 / sqrt(2.0);
    ComplexType overlap = (amplitude_00 + amplitude_11) * sqrt2_inv;
    return std::norm(overlap);
}

//
// Table printing helpers
//
void printTableHeader(const char* title, const std::vector<double>& noiseParams)
{
    output("Table: %s\n", title);
    output("Qubits  |");
    // Determine if this is error rates (max <= 0.10) or damping parameters (max > 0.10)
    double maxParam = *std::max_element(noiseParams.begin(), noiseParams.end());
    bool isErrorRate = (maxParam <= 0.10);

    for (double param : noiseParams)
    {
        if (isErrorRate)
            output("  %4.1f%%   |", param * 100); // Show as percentage for error rates
        else
            output("  %.3f   |", param); // Show raw value for damping parameters
    }
    output("\n");

    output("--------|");
    for (size_t i = 0; i < noiseParams.size(); ++i)
        output("----------|");
    output("\n");
}

void printTableRow(int numQubits, const std::vector<double>& fidelities)
{
    output("   %2d   |", numQubits);
    for (double fidelity : fidelities)
        output(" %.6f |", fidelity);
    output("\n");
}

// Collect fidelity data for a specific noise configuration
double collectFidelityData(ExSVUpdater svUpdater, int numQubits,
                           void (*enqueueNoise)(ExSVUpdater, double, int), double noiseParam,
                           std::mt19937& gen)
{
    // Create state vector for this qubit size
    auto stateVector = createStateVector(numQubits);

    // Clear and rebuild circuit
    ERRCHK(custatevecExSVUpdaterClear(svUpdater));
    enqueueGHZCircuit(svUpdater, numQubits);
    if (noiseParam > 0.0)
        enqueueNoise(svUpdater, noiseParam, numQubits);

    // Get required random numbers
    int32_t maxRandnums;
    ERRCHK(custatevecExSVUpdaterGetMaxNumRequiredRandnums(svUpdater, &maxRandnums));

    // Average over multiple runs
    double totalFidelity = 0.0;
    constexpr int numRuns = 5;

    for (int run = 0; run < numRuns; ++run)
    {
        ERRCHK(custatevecExStateVectorSetZeroState(stateVector));
        // custatevecExSVUpdaterApply can be called multiple times with different set of randnum
        // numbers.
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<double> randnums(maxRandnums);
        for (int i = 0; i < maxRandnums; ++i)
            randnums[i] = dist(gen);
        ERRCHK(custatevecExSVUpdaterApply(svUpdater, stateVector, randnums.data(), maxRandnums));

        double fidelity = measureGHZFidelity(stateVector, numQubits);
        totalFidelity += fidelity;
    }

    // Cleanup
    ERRCHK(custatevecExStateVectorDestroy(stateVector));

    return totalFidelity / numRuns;
}

int main(int argc, char* argv[])
{
    // Check for quiet mode option
    if (argc >= 2 && strcmp(argv[1], "-q") == 0)
        output_enabled = false;

    output("=== Quantum Noise Analysis: GHZ State Fidelity ===\n\n");
    // Random number generator
    std::mt19937 gen(12345);

    // Create SVUpdater for all experiments
    auto svUpdater = createSVUpdater();

    // Define experimental parameters
    std::vector<int> qubitSizes = {2, 4, 8, 16, 25};
    std::vector<double> depolarizingRates = {0.0, 0.005, 0.01, 0.02, 0.05, 0.10};
    std::vector<double> dampingParams = {0.0, 0.05, 0.1, 0.2, 0.3, 0.5};

    // Data collection arrays
    std::vector<std::vector<double>> depolarizingData(
        qubitSizes.size(), std::vector<double>(depolarizingRates.size()));
    std::vector<std::vector<double>> dampingData(qubitSizes.size(),
                                                 std::vector<double>(dampingParams.size()));

    //
    // Data Collection: Depolarizing Noise
    //
    output("GHZ fidelity measurement with depolarizing noise\n");
    output("Qubit systems: {2, 4, 8, 16, 25} qubits\n");
    output("Depolarizing noise rates: {0.0%%, 0.5%%, 1.0%%, 2.0%%, 5.0%%, 10.0%%}\n");
    output("Total measurements: %zu fidelity samples\n\n",
           qubitSizes.size() * depolarizingRates.size());
    output("Collecting depolarizing noise data");
    for (size_t i = 0; i < qubitSizes.size(); ++i)
    {
        for (size_t j = 0; j < depolarizingRates.size(); ++j)
        {
            depolarizingData[i][j] = collectFidelityData(
                svUpdater, qubitSizes[i], enqueueDepolarizingNoise, depolarizingRates[j], gen);
            output_char('.');
        }
    }
    output("done.\n");
    // Show depolarizing noise table immediately
    output("\n");
    printTableHeader("Depolarizing Noise (Error Rate %)", depolarizingRates);
    for (size_t i = 0; i < qubitSizes.size(); ++i)
        printTableRow(qubitSizes[i], depolarizingData[i]);
    output("\n\n");

    //
    // Data Collection: Amplitude Damping
    //
    output("GHZ fidelity measurement with amplitude damping noise\n");
    output("Qubit systems: {2, 4, 8, 16, 25} qubits\n");
    output("Amplitude damping parameters: {gamma=0.0, 0.05, 0.1, 0.2, 0.3, 0.5}\n");
    output("Total measurements: %zu fidelity samples\n\n",
           qubitSizes.size() * dampingParams.size());
    output("\nCollecting amplitude damping data");
    for (size_t i = 0; i < qubitSizes.size(); ++i)
    {
        for (size_t j = 0; j < dampingParams.size(); ++j)
        {
            dampingData[i][j] = collectFidelityData(
                svUpdater, qubitSizes[i], enqueueAmplitudeDampingNoise, dampingParams[j], gen);
            output_char('.');
        }
    }
    output("done.\n");
    // Show amplitude damping table immediately
    output("\n");
    printTableHeader("Amplitude Damping (Damping Parameter gamma)", dampingParams);
    for (size_t i = 0; i < qubitSizes.size(); ++i)
        printTableRow(qubitSizes[i], dampingData[i]);

    output("\n=== Analysis completed successfully! ===\n");
    // Cleanup
    ERRCHK(custatevecExSVUpdaterDestroy(svUpdater));
    printf("PASSED\n");
    return EXIT_SUCCESS;
}
