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
// This example is derived from Qiskit's example to estimate pi.
//
// The algorithm rotates the target bit by 1.  Thus, the estimated
// phase will be 1/(2pi).
//
// The highest order qubit is the target qubit whose phase is measured.
// Other lower order qubits are phase registers.
//

#include <custatevecEx.h> // custatevecEx API
#include <stdlib.h>       // exit()
#include <cstdio>         // printf
#include <complex>        // std::complex<>
#include <vector>         // std::vector<>
#include <random>         // std::uniform_real_distribution<>
#include <numeric>        // std::iota
#include <cmath>          // std::abs, M_PI
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

#define ERRCHK(s)                                                                                  \
    {                                                                                              \
        auto status = (s);                                                                         \
        if (status != CUSTATEVEC_STATUS_SUCCESS)                                                   \
        {                                                                                          \
            printf("%s, %s\n", custatevecGetErrorName(status), #s);                                \
            exit(EXIT_FAILURE); /* Error exit skips resource cleanup for simplicity */             \
        }                                                                                          \
    }

#define ERRCHK_CUDA(s)                                                                             \
    {                                                                                              \
        auto status = (s);                                                                         \
        if (status != cudaSuccess)                                                                 \
        {                                                                                          \
            printf("%s, %s\n", cudaGetErrorName(status), #s);                                      \
            exit(EXIT_FAILURE); /* Error exit skips resource cleanup for simplicity */             \
        }                                                                                          \
    }

// typedef's for short names
typedef custatevecExStateVectorDescriptor_t ExStateVector;
typedef custatevecExSVUpdaterDescriptor_t ExSVUpdater;
typedef custatevecExDictionaryDescriptor_t ExDictionary;
typedef custatevecIndex_t Index_t;

// Matrices are expressed by c128 values.
typedef std::complex<double> DblComplex;

//
// create state vector
//
custatevecExStateVectorDescriptor_t createStateVector(cudaDataType_t svDataType, int numWires)
{
    ExDictionary svConfig{nullptr};
    custatevecExStateVectorDescriptor_t stateVector{nullptr};
    // numWires should be specified twice for device state vector.
    ERRCHK(custatevecExConfigureStateVectorSingleDevice(&svConfig, svDataType, numWires, numWires,
                                                        0, 0));
    // create state vector
    // stream is not specified here, then, stateVector instance will use the default stream.
    // Note: custom memory allocator is not supported in the current release.
    ERRCHK(custatevecExStateVectorCreateSingleProcess(&stateVector, svConfig, nullptr, 0, nullptr));
    // destroy dictionary
    ERRCHK(custatevecExDictionaryDestroy(svConfig));

    // Set math mode for state vector operations
    // These modes are particularly effective on GPUs with compute capability 10.0+ (e.g., B200)
    // BF16x9 modes can provide performance benefits for certain operations
    // Uncomment one of the following lines to select the desired math mode:

    custatevecMathMode_t mathMode = CUSTATEVEC_MATH_MODE_DEFAULT;
    ERRCHK(custatevecExStateVectorSetMathMode(stateVector, mathMode));

    // mathMode = CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9;
    // ERRCHK(custatevecExStateVectorSetMathMode(stateVector, mathMode));

    // mathMode = CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9;
    // ERRCHK(custatevecExStateVectorSetMathMode(stateVector, mathMode));

    // Print the selected math mode
    const char* mathModeStr = "UNKNOWN";
    switch (mathMode)
    {
    case CUSTATEVEC_MATH_MODE_DEFAULT:
        mathModeStr = "DEFAULT";
        break;
    case CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9:
        mathModeStr = "ALLOW_FP32_EMULATED_BF16X9";
        break;
    case CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9:
        mathModeStr = "DISALLOW_FP32_EMULATED_BF16X9";
        break;
    default:
        mathModeStr = "UNKNOWN";
        break;
    }
    output("Math mode: %s\n", mathModeStr);

    // Use custatevecExStateVectorGetProperty() to confirm
    // the data type and the number of wires.
    int numWiresProp = -1;
    cudaDataType_t svDataTypeProp{cudaDataType_t(0)};

    ERRCHK(custatevecExStateVectorGetProperty(stateVector, CUSTATEVEC_EX_SV_PROP_NUM_WIRES,
                                              &numWiresProp, sizeof(numWiresProp)));
    ERRCHK(custatevecExStateVectorGetProperty(stateVector, CUSTATEVEC_EX_SV_PROP_DATA_TYPE,
                                              &svDataTypeProp, sizeof(svDataTypeProp)));

    const char* dataTypeStr = "unknown";
    if (svDataTypeProp == CUDA_C_64F)
        dataTypeStr = "c128";
    else if (svDataTypeProp == CUDA_C_32F)
        dataTypeStr = "c64";
    output("dataType=%s, numWires=%d\n", dataTypeStr, numWires);

    return stateVector;
}

//
// create SVUpdater (= state vector updater)
//
ExSVUpdater createSVUpdater(cudaDataType_t svUpdaterDataType)
{
    ExDictionary svUpdaterConfig{nullptr};
    ExSVUpdater svUpdater{nullptr};

    // create config for SVUpdater
    ERRCHK(custatevecExConfigureSVUpdater(&svUpdaterConfig, svUpdaterDataType, nullptr, 0));
    // create SVUpdater
    // NOTE: custom memory allocator is not supported in the current release.
    ERRCHK(custatevecExSVUpdaterCreate(&svUpdater, svUpdaterConfig, nullptr));

    // destroy dictionary
    ERRCHK(custatevecExDictionaryDestroy(svUpdaterConfig));

    return svUpdater;
}

//
// Apply the circuit by using custatevecExApplyMatrix()
//
void applyCircuitByApplyMatrix(custatevecExStateVectorDescriptor_t stateVector, int numWires)
{
    auto target = numWires - 1;
    auto nPhaseRegisters = numWires - 1;

    // 2x2 unitary matrix in row-major order.
    DblComplex H[] = {1, 1, 1, -1};
    for (auto& elm : H)
        elm *= 1. / std::sqrt(2.);
    // X gate
    DblComplex X[] = {0, 1, 1, 0};
    int adjoint = 0; // matrices are not adjoint.

    for (int reg = 0; reg < nPhaseRegisters; ++reg)
    {
        ERRCHK(custatevecExApplyMatrix(stateVector, H, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DENSE,
                                       CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &reg, 1, nullptr,
                                       nullptr, 0));
    }
    // apply X on target
    // NOTE: The current release does not support CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL for
    // custatevecExApplyMatrix() API.  X is given as a dense matrix.
    ERRCHK(custatevecExApplyMatrix(stateVector, X, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DENSE,
                                   CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &target, 1, nullptr,
                                   nullptr, 0));
    // apply controlled phase gates
    for (int reg = 0; reg < nPhaseRegisters; ++reg)
    {
        auto theta = double(1LL << reg);
        DblComplex phase[2] = {1., std::exp(DblComplex(0., theta))};
        ERRCHK(custatevecExApplyMatrix(stateVector, phase, CUDA_C_64F,
                                       CUSTATEVEC_EX_MATRIX_DIAGONAL, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                                       adjoint, &target, 1, &reg, nullptr, 1));
    }
    // inverse fourier transform.
    for (int reg0 = 0; reg0 < nPhaseRegisters / 2; ++reg0)
    {
        auto reg1 = (nPhaseRegisters - 1) - reg0;
        // SWAP
        ERRCHK(custatevecExApplyMatrix(stateVector, X, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DENSE,
                                       CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &reg0, 1, &reg1,
                                       nullptr, 1));
        ERRCHK(custatevecExApplyMatrix(stateVector, X, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DENSE,
                                       CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &reg1, 1, &reg0,
                                       nullptr, 1));
        ERRCHK(custatevecExApplyMatrix(stateVector, X, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DENSE,
                                       CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &reg0, 1, &reg1,
                                       nullptr, 1));
    }
    for (int regJ = 0; regJ < nPhaseRegisters; ++regJ)
    {
        for (int regM = 0; regM < regJ; ++regM)
        {
            double theta = -M_PI / double(1LL << (regJ - regM));
            DblComplex phase[2] = {1., std::exp(DblComplex(0., theta))};
            ERRCHK(custatevecExApplyMatrix(
                stateVector, phase, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DIAGONAL,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &regJ, 1, &regM, nullptr, 1));
        }
        ERRCHK(custatevecExApplyMatrix(stateVector, H, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DENSE,
                                       CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &regJ, 1, nullptr,
                                       nullptr, 0));
    }
}

void applyCircuitBySVUpdater(custatevecExStateVectorDescriptor_t stateVector, int numWires,
                             cudaDataType_t svUpdaterDataType)
{
    auto target = numWires - 1;
    auto nPhaseRegisters = numWires - 1;

    // 2x2 unitary matrix in row-major order.
    DblComplex H[] = {1, 1, 1, -1};
    for (auto& elm : H)
        elm *= 1. / std::sqrt(2.);
    // Anti-diagonal elements of X gate
    DblComplex Xanti[] = {1, 1};
    int adjoint = 0; // matrices are not adjoint.

    auto svUpdater = createSVUpdater(svUpdaterDataType);

    for (int reg = 0; reg < nPhaseRegisters; ++reg)
    {
        ERRCHK(custatevecExSVUpdaterEnqueueMatrix(
            svUpdater, H, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DENSE, CUSTATEVEC_MATRIX_LAYOUT_ROW,
            adjoint, &reg, 1, nullptr, nullptr, 0));
    }
    // apply X on target by giving X as a anti-diagonal matrix.
    ERRCHK(custatevecExSVUpdaterEnqueueMatrix(
        svUpdater, Xanti, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &target, 1, nullptr, nullptr, 0));
    // apply controlled phase gates
    for (int reg = 0; reg < nPhaseRegisters; ++reg)
    {
        auto theta = double(1LL << reg);
        DblComplex phase[] = {1., std::exp(DblComplex(0., theta))};
        ERRCHK(custatevecExSVUpdaterEnqueueMatrix(
            svUpdater, phase, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DIAGONAL,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &target, 1, &reg, nullptr, 1));
    }
    // inverse fourier transform.
    for (int reg0 = 0; reg0 < nPhaseRegisters / 2; ++reg0)
    {
        auto reg1 = (nPhaseRegisters - 1) - reg0;
        // SWAP
        ERRCHK(custatevecExSVUpdaterEnqueueMatrix(
            svUpdater, Xanti, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &reg0, 1, &reg1, nullptr, 1));
        ERRCHK(custatevecExSVUpdaterEnqueueMatrix(
            svUpdater, Xanti, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &reg1, 1, &reg0, nullptr, 1));
        ERRCHK(custatevecExSVUpdaterEnqueueMatrix(
            svUpdater, Xanti, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &reg0, 1, &reg1, nullptr, 1));
    }
    for (int regJ = 0; regJ < nPhaseRegisters; ++regJ)
    {
        for (int regM = 0; regM < regJ; ++regM)
        {
            double theta = -M_PI / double(1LL << (regJ - regM));
            DblComplex phase[] = {1., std::exp(DblComplex(0., theta))};
            ERRCHK(custatevecExSVUpdaterEnqueueMatrix(
                svUpdater, phase, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DIAGONAL,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &regJ, 1, &regM, nullptr, 1));
        }
        ERRCHK(custatevecExSVUpdaterEnqueueMatrix(
            svUpdater, H, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DENSE, CUSTATEVEC_MATRIX_LAYOUT_ROW,
            adjoint, &regJ, 1, nullptr, nullptr, 0));
    }

    // Apply queued gate matrices
    ERRCHK(custatevecExSVUpdaterApply(svUpdater, stateVector, nullptr, 0));
    // Destroy SVUpdater
    ERRCHK(custatevecExSVUpdaterDestroy(svUpdater));
}

Index_t getBitStringBySampling(ExStateVector stateVector, int numWires)
{
    // Example of sample() API

    constexpr int nShots = 20000;
    std::vector<double> randnums(nShots);

    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    // Generate random double numbers
    for (int idx = 0; idx < nShots; ++idx)
        randnums[idx] = dist(gen);

    int numPhaseRegisters = numWires - 1;
    std::vector<int> wires(numPhaseRegisters);
    std::iota(wires.begin(), wires.end(), 0);
    std::vector<Index_t> bitStrings(nShots);
    // get the sampled result in the ascending order to create histogram
    ERRCHK(custatevecExSample(stateVector, bitStrings.data(), wires.data(), numPhaseRegisters,
                              randnums.data(), nShots, CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER,
                              nullptr));
    // synchronize here because device to host memory copy happens.
    // On Grace-Hopper/Blackwell systems, the copy is asynchronous.
    ERRCHK(custatevecExStateVectorSynchronize(stateVector));

    // Create histogram
    output("Histogram, ");
    std::vector<Index_t> observed;
    std::vector<int> occurrences;
    for (Index_t idx = 0; idx < static_cast<Index_t>(bitStrings.size());)
    {
        auto bitString = bitStrings[idx];
        auto first = idx;
        auto last = static_cast<Index_t>(bitStrings.size());
        while (first < last)
        {
            if (bitString == bitStrings[first])
                ++first;
            else
                break;
        }
        observed.push_back(bitString);
        auto occurrence = first - idx;
        occurrences.push_back(occurrence);
        output("(%lx, %ld)", bitString, occurrence);
        idx = first;
    }
    output("\n");

    // Get the bitstrings most frequently observed
    Index_t mostObserved = 0;
    int maxOccurrence = 0;
    for (int idx = 0; idx < static_cast<int>(occurrences.size()); ++idx)
    {
        if (occurrences[idx] <= maxOccurrence)
            continue;
        mostObserved = observed[idx];
        maxOccurrence = occurrences[idx];
    }
    return mostObserved;
}

Index_t getBitStringByMeasure(ExStateVector stateVector, int numWires)
{
    // Check if the same string is returned by using measure() API
    // measure() API executes batched single qubit measurements

    // For this time, use 0.5 as the random number.
    // The observation probability of the expected bitString is very high.
    // Almost any random numbers will return the same bitString.
    const double randnum = 0.5;
    int numPhaseWires = numWires - 1;

    Index_t bitString;
    std::vector<int> wireOrdering(numPhaseWires);
    std::iota(wireOrdering.begin(), wireOrdering.end(), 0);

    ERRCHK(custatevecExMeasure(stateVector, &bitString, wireOrdering.data(), numPhaseWires, randnum,
                               CUSTATEVEC_COLLAPSE_NONE, nullptr));
    return bitString;
}

Index_t getBitStringByAbs2SumArray(ExStateVector stateVector, int numWires)
{
    // For this time, use 0.5 as the random number.
    // The observation probability of the expected bitString is very high.
    // Almost any random number will return the same bitString.
    double randnum = 0.5;
    double randnumOffset = 0.;
    std::vector<int> observed;
    std::vector<int> maskOrdering;

    // custaevecExAbs2SumArray() can compute the norm of the state vector
    // by passing empty wireOrdering
    double norm;
    ERRCHK(custatevecExAbs2SumArray(stateVector, &norm, nullptr, 0, nullptr, nullptr, 0));
    ERRCHK(custatevecExStateVectorSynchronize(stateVector));

    // Measure bitstring by computing the probability from high-order to low-order wires.
    int numPhaseRegisters = numWires - 1;
    for (int reg = numPhaseRegisters - 1; 0 <= reg; --reg)
    {
        double abs2sums[2];
        ERRCHK(custatevecExAbs2SumArray(stateVector, abs2sums, &reg, 1, observed.data(),
                                        maskOrdering.data(), observed.size()));
        ERRCHK(custatevecExStateVectorSynchronize(stateVector));
        auto prob0 = abs2sums[0] / norm + randnumOffset;
        if (randnum < prob0)
        {
            observed.insert(observed.begin(), 0);
        }
        else
        {
            observed.insert(observed.begin(), 1);
            randnumOffset = prob0;
        }
        maskOrdering.insert(maskOrdering.begin(), reg);
    }

    Index_t bitString = 0;
    for (int reg = 0; reg < numPhaseRegisters; ++reg)
        bitString |= Index_t(observed[reg]) << reg;
    return bitString;
}

double computePi(Index_t bitString, int numWires)
{
    // estimate pi from the measured counts
    auto numPhaseRegisters = numWires - 1;
    auto theta = double(bitString) / double(1LL << numPhaseRegisters);
    auto piEstimate = 1. / (2 * theta);
    auto delta = piEstimate - M_PI;
    output("pi=%.16f, delta=%g\n\n", piEstimate, delta);

    return delta;
}

int main(int argc, char* argv[])
{
    // Check for quiet mode option
    if (argc >= 2 && strcmp(argv[1], "-q") == 0)
        output_enabled = false;

    int numWires = 25;
    auto svDataType = CUDA_C_32F;

    //
    // create state vector
    //
    output("create state vector\n");
    auto stateVector = createStateVector(svDataType, numWires);
    // set |00..0>
    ERRCHK(custatevecExStateVectorSetZeroState(stateVector));

    //
    // Apply circuit
    //
    output("apply circuit.\n");

    cudaEvent_t evBegin, evEnd;
    ERRCHK_CUDA(cudaEventCreate(&evBegin));
    ERRCHK_CUDA(cudaEventCreate(&evEnd));
    // CUDA kernels will be launched on the default (null) stream.
    ERRCHK_CUDA(cudaEventRecord(evBegin, nullptr));

    // Use SVUpdater to apply circuit.
    // Gates are fused internally in SVUpdater
    applyCircuitBySVUpdater(stateVector, numWires, svDataType);

    // Apply circuit by custatevecExApplyMatrix()
    // applyCircuitByApplyMatrix(stateVector, numWires);

    ERRCHK_CUDA(cudaEventRecord(evEnd, nullptr));
    ERRCHK_CUDA(cudaEventSynchronize(evEnd));
    float elapsedMs;
    ERRCHK_CUDA(cudaEventElapsedTime(&elapsedMs, evBegin, evEnd));
    ERRCHK_CUDA(cudaEventDestroy(evBegin));
    ERRCHK_CUDA(cudaEventDestroy(evEnd));
    output("Elapsed time, %g [s]\n", elapsedMs / 1000.);

    //
    // get bit string by using several custatevecEx API
    //
    output("get bit strings.\n");

    // custatevecExSample()
    auto bitStringSample = getBitStringBySampling(stateVector, numWires);
    output("bitString, sampling, %lx\n", bitStringSample);

    // custatevecExMeasure()
    auto bitStringByMeasure = getBitStringByMeasure(stateVector, numWires);
    output("bitString, measure, %lx\n", bitStringByMeasure);

    // custatevecExAbs2SumArray()
    auto bitStringByAbs2SumArray = getBitStringByAbs2SumArray(stateVector, numWires);
    output("bitString, abs2SumArray, %lx\n", bitStringByAbs2SumArray);

    output("\nCompute pi.\n");
    auto delta = computePi(bitStringSample, numWires);

    ERRCHK(custatevecExStateVectorDestroy(stateVector));

    constexpr double ep = 1.e-6;
    bool pass = std::abs(delta) <= ep;
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
