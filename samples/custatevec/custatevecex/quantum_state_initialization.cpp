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
// This example demonstrates wire ordering reassignment vs. index bit permutation.
// It also shows the use cases of device resource getter APIs for direct GPU memory access.
//
// 1. Initialize state vector elements to their index values (element[i] = i)
// 2. Get the initial wire ordering from the state vector using custatevecExStateVectorGetProperty
// API
// 3. Reassign wire ordering to a random one (metadata only, elements unchanged)
// 4. Verify wire ordering changed but state vector elements remain the same
// 5. Use scatter permutation to revert wire ordering back to initial state
// 6. Verify wire ordering is reverted and state vector elements are now permuted
//
// Key difference: custatevecExStateVectorReassignWireOrdering changes only metadata,
// custatevecExStateVectorPermuteIndexBits changes both wire ordering and rearranges state vector
// elements.
//
// Device resource getter APIs demonstrated:
// - custatevecExStateVectorGetResourcesFromDeviceSubSV: Get mutable device pointer
// - custatevecExStateVectorGetResourcesFromDeviceSubSVView: Get read-only device pointer
// - cudaMemcpyAsync + cudaStreamSynchronize: Direct async GPU memory operations
//

#include <custatevecEx.h> // custatevecEx API
#include <cuda_runtime.h> // CUDA runtime API
#include <stdlib.h>       // exit()
#include <cstdio>         // printf
#include <complex>        // std::complex<>
#include <vector>         // std::vector<>
#include <numeric>        // std::iota
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

typedef custatevecExStateVectorDescriptor_t ExStateVector;
typedef custatevecIndex_t Index_t;
typedef std::complex<double> DblComplex;

// Forward declarations for utility functions
void printVector(const std::vector<int32_t>& vec, const char* label);
void setStateVectorElements(ExStateVector stateVector, const std::vector<DblComplex>& elements);
std::vector<DblComplex> getStateVectorElements(ExStateVector stateVector, int numWires);
std::vector<int32_t> getWireOrdering(ExStateVector stateVector);
bool verifyElements(ExStateVector stateVector, int numWires,
                    const std::vector<int32_t>& expectedIndices);

//
// Print vector in format: label: [1, 2, 3, 4]
//
void printVector(const std::vector<int32_t>& vec, const char* label)
{
    output("%s: [", label);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        output("%d", vec[i]);
        if (i < vec.size() - 1)
            output(", ");
    }
    output("]\n");
}

//
// Set state vector elements using device resource getter APIs and async memory copy
//
void setStateVectorElements(ExStateVector stateVector, const std::vector<DblComplex>& elements)
{
    auto numElements = elements.size();

    // This function demonstrates how to set state vector elements using device resource getter
    // APIs, as an alternative to the high-level API shown below.
    // ERRCHK(custatevecExStateVectorSetState(stateVector, elements.data(), CUDA_C_64F, 0,
    // numElements, 1)); ERRCHK(custatevecExStateVectorSynchronize(stateVector));

    // Advanced implementation using device pointer and async memory copy:
    // Get device pointer and stream from state vector (subSVIndex=0 for single-device)
    int32_t subSVIndex = 0;
    int32_t deviceId = -1;
    void* devicePtr = nullptr;
    cudaStream_t stream = nullptr;
    ERRCHK(custatevecExStateVectorGetResourcesFromDeviceSubSV(stateVector, subSVIndex, &deviceId,
                                                              &devicePtr, &stream, nullptr));

    // Copy data from host to device asynchronously
    size_t sizeBytes = numElements * sizeof(DblComplex);
    ERRCHK_CUDA(
        cudaMemcpyAsync(devicePtr, elements.data(), sizeBytes, cudaMemcpyHostToDevice, stream));

    // Synchronize the stream to ensure transfer completion
    ERRCHK_CUDA(cudaStreamSynchronize(stream));
}

//
// Get state vector elements using device resource getter APIs and async memory copy
//
std::vector<DblComplex> getStateVectorElements(ExStateVector stateVector, int numWires)
{
    auto numElements = 1LL << numWires;
    std::vector<DblComplex> elements(numElements);

    // This function demonstrates how to get state vector elements using device resource getter
    // APIs, as an alternative to the high-level API shown below.
    // ERRCHK(custatevecExStateVectorGetState(stateVector, elements.data(), CUDA_C_64F, 0,
    // numElements, 1)); ERRCHK(custatevecExStateVectorSynchronize(stateVector));

    // Advanced implementation using device pointer and async memory copy:
    // Get device pointer and stream from state vector view (subSVIndex=0 for single-device)
    int32_t subSVIndex = 0;
    int32_t deviceId = -1;
    const void* devicePtr = nullptr;
    cudaStream_t stream = nullptr;
    ERRCHK(custatevecExStateVectorGetResourcesFromDeviceSubSVView(
        stateVector, subSVIndex, &deviceId, &devicePtr, &stream, nullptr));

    // Copy data from device to host asynchronously
    size_t sizeBytes = numElements * sizeof(DblComplex);
    ERRCHK_CUDA(
        cudaMemcpyAsync(elements.data(), devicePtr, sizeBytes, cudaMemcpyDeviceToHost, stream));

    // Synchronize the stream to ensure transfer completion
    ERRCHK_CUDA(cudaStreamSynchronize(stream));

    return elements;
}

//
// Get wire ordering from state vector
//
std::vector<int32_t> getWireOrdering(ExStateVector stateVector)
{
    // Get number of wires first
    int32_t numWires = 0;
    ERRCHK(custatevecExStateVectorGetProperty(stateVector, CUSTATEVEC_EX_SV_PROP_NUM_WIRES,
                                              &numWires, sizeof(numWires)));

    // Get wire ordering
    std::vector<int32_t> wireOrdering(numWires);
    ERRCHK(custatevecExStateVectorGetProperty(stateVector, CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING,
                                              wireOrdering.data(),
                                              wireOrdering.size() * sizeof(int32_t)));
    return wireOrdering;
}

//
// Verify state vector elements match expected indices
//
bool verifyElements(ExStateVector stateVector, int numWires,
                    const std::vector<int32_t>& expectedIndices)
{
    auto numElements = 1LL << numWires;
    auto elements = getStateVectorElements(stateVector, numWires);

    bool success = true;
    for (Index_t i = 0; i < numElements; ++i)
    {
        Index_t actualValue = static_cast<Index_t>(std::round(elements[i].real()));
        Index_t expectedValue = static_cast<Index_t>(expectedIndices[i]);

        if (actualValue != expectedValue)
        {
            printf("ERROR: Element %lld expected %lld but got %lld\n", static_cast<long long>(i),
                   static_cast<long long>(expectedValue), static_cast<long long>(actualValue));
            success = false;
            break;
        }
    }

    output("Elements: [%.0f, %.0f, %.0f, %.0f", elements[0].real(), elements[1].real(),
           elements[2].real(), elements[3].real());
    if (numElements > 4)
        output(", ...");
    output("]\n");

    return success;
}

int main(int argc, char* argv[])
{
    // Check for quiet mode option
    if (argc >= 2 && strcmp(argv[1], "-q") == 0)
        output_enabled = false;

    const int numWires = 4;
    const cudaDataType_t svDataType = CUDA_C_64F;
    auto numElements = 1LL << numWires;

    output("Number of wires: %d\n", numWires);
    output("Number of elements: %lld\n\n", static_cast<long long>(numElements));

    //
    // Create state vector
    //
    output("Step 1: Creating state vector and setting initial state\n");

    custatevecExDictionaryDescriptor_t svConfig;
    ERRCHK(custatevecExConfigureStateVectorSingleDevice(&svConfig, svDataType, numWires, numWires,
                                                        /*deviceId=*/0, /*capability=*/0));

    ExStateVector stateVector;
    ERRCHK(custatevecExStateVectorCreateSingleProcess(&stateVector, svConfig, nullptr, 0, nullptr));

    ERRCHK(custatevecExDictionaryDestroy(svConfig));

    // Initialize state vector elements to their index values
    std::vector<DblComplex> initialElements(numElements);
    for (Index_t i = 0; i < numElements; ++i)
        initialElements[i] = DblComplex(static_cast<double>(i), 0.0);
    setStateVectorElements(stateVector, initialElements);

    // Get and display initial wire ordering
    auto initialOrdering = getWireOrdering(stateVector);
    printVector(initialOrdering, "Initial wire ordering");

    // Verify initial state vector elements
    std::vector<int32_t> initialIndices(numElements);
    std::iota(initialIndices.begin(), initialIndices.end(), 0);
    output("Initial state: ");
    verifyElements(stateVector, numWires, initialIndices);

    //
    // Step 2: Reassign wire ordering to a random one
    //
    output("\nStep 2: Reassigning wire ordering (metadata only)\n");

    // Create a simple random ordering: [2, 3, 1, 0]
    std::vector<int32_t> randomOrdering = {2, 3, 1, 0};
    printVector(randomOrdering, "Random ordering");

    ERRCHK(
        custatevecExStateVectorReassignWireOrdering(stateVector, randomOrdering.data(), numWires));

    auto reassignedOrdering = getWireOrdering(stateVector);
    printVector(reassignedOrdering, "After reassignment");

    // Verify wire ordering changed but state vector elements remain unchanged
    if (reassignedOrdering == randomOrdering)
    {
        output("Wire ordering successfully reassigned\n");
    }
    else
    {
        printf("ERROR: Wire ordering reassignment failed\n");
        printf("FAILED\n");
        return EXIT_FAILURE; // Skip releasing resources on failure
    }

    output("After reassignment: ");
    bool elementsUnchanged = verifyElements(stateVector, numWires, initialIndices);
    if (elementsUnchanged)
    {
        output("State vector elements unchanged (as expected for reassignment)\n");
    }
    else
    {
        printf("ERROR: State vector elements should not change during reassignment\n");
        printf("FAILED\n");
        return EXIT_FAILURE; // Skip releasing resources on failure
    }

    //
    // Step 3: Revert wire ordering using scatter permutation
    //
    output("\nStep 3: Reverting wire ordering using scatter permutation\n");

    // Use the current reassigned wire ordering as scatter permutation
    // This will rearrange the state vector elements according to the wire ordering
    printVector(reassignedOrdering, "Using reassigned ordering for scatter");

    ERRCHK(custatevecExStateVectorPermuteIndexBits(stateVector, reassignedOrdering.data(), numWires,
                                                   CUSTATEVEC_EX_PERMUTATION_SCATTER));

    auto revertedOrdering = getWireOrdering(stateVector);
    printVector(revertedOrdering, "After scatter permutation");

    //
    // Step 4: Validate results
    //
    output("\nStep 4: Validating final results\n");

    // Check the final wire ordering after scatter permutation
    // Note: After scatter permutation, the wire ordering will change based on the permutation
    // applied
    output("Final wire ordering after scatter permutation:\n");
    printVector(initialOrdering, "Original");
    printVector(revertedOrdering, "Final   ");

    // Compute expected state vector elements after scatter permutation
    // For scatter permutation with reassigned ordering [2, 3, 1, 0], element i goes to position
    // where bits are rearranged
    std::vector<int32_t> expectedElements(numElements);
    for (Index_t originalIdx = 0; originalIdx < numElements; ++originalIdx)
    {
        Index_t newIdx = 0;
        for (int j = 0; j < numWires; ++j)
        {
            Index_t srcBit = (originalIdx >> j) & 1;
            newIdx |= (srcBit << reassignedOrdering[j]);
        }
        expectedElements[newIdx] = static_cast<int32_t>(originalIdx);
    }

    output("After permutation: ");
    bool elementsCorrect = verifyElements(stateVector, numWires, expectedElements);

    if (elementsCorrect)
    {
        output("State vector elements correctly permuted\n");
    }
    else
    {
        printf("ERROR: State vector elements permutation failed\n");
        printf("FAILED\n");
        return EXIT_FAILURE; // Skip releasing resources on failure
    }

    output("\n=== Sample completed successfully! ===\n");
    output("ReassignWireOrdering: Changes wire ordering metadata only, elements unchanged\n");
    output("PermuteIndexBits: Changes both wire ordering AND rearranges state vector elements\n");
    output("GetResourcesFromDeviceSubSV/View + cudaMemcpyAsync for direct device access\n");

    //
    // Cleanup
    //
    ERRCHK(custatevecExStateVectorDestroy(stateVector));

    printf("PASSED\n");
    return EXIT_SUCCESS;
}
