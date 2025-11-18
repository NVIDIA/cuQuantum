/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

//
// This example demonstrates wire management and index bit permutation.
//
// 1. Initialize state vector elements to their index values (element[i] = i)
// 2. Get the wire ordering from the state vector using custatevecExStateVectorGetProperty API
// 3. Generate a random permutation by shuffling the wire ordering
// 4. Apply the permutation using custatevecExStateVectorPermuteIndexBits API
// 5. Verify that the state vector elements reflect the permuted indices
//

#include <custatevecEx.h> // custatevecEx API
#include <cuda_runtime.h> // CUDA runtime API
#include <stdlib.h>       // exit()
#include <cstdio>         // printf
#include <complex>        // std::complex<>
#include <vector>         // std::vector<>
#include <random>         // std::random_device, std::mt19937
#include <algorithm>      // std::shuffle
#include <numeric>        // std::iota
#include <string>         // std::string, std::to_string
#include <cstring>        // strcmp
#include <cstdarg>        // va_list, va_start, va_end
#include "common.hpp"     // custatevecEx error checking utilities

// typedef's for short names
typedef custatevecExStateVectorDescriptor_t ExStateVector;
typedef custatevecExDictionaryDescriptor_t ExDictionary;
typedef custatevecIndex_t Index_t;

// State vector elements are complex doubles
typedef std::complex<double> DblComplex;

// Forward declarations for utility functions

void printWireOrderingComparison(
    const std::vector<int32_t>& original, const std::vector<int32_t>& permuted,
    const std::vector<int32_t>& permutation, custatevecExPermutationType_t permutationType);
void applyIndexBitPermutation(ExStateVector stateVector, const std::vector<int32_t>& permutation,
                              custatevecExPermutationType_t permutationType);

//
// create state vector
//
ExStateVector createStateVector(cudaDataType_t svDataType, int numWires)
{
    ExDictionary svConfig{nullptr};
    ExStateVector stateVector{nullptr};
    // numWires should be specified twice for device state vector.
    ERRCHK(custatevecExConfigureStateVectorSingleDevice(&svConfig, svDataType, numWires, numWires,
                                                        0, 0));
    // create state vector
    ERRCHK(custatevecExStateVectorCreateSingleProcess(&stateVector, svConfig, nullptr, 0, nullptr));
    // destroy dictionary
    ERRCHK(custatevecExDictionaryDestroy(svConfig));

    // Use custatevecExStateVectorGetProperty() to confirm the data type and number of wires
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
    output("Created state vector: dataType=%s, numWires=%d\n", dataTypeStr, numWires);

    return stateVector;
}

//
// Initialize state vector elements to their index values
//
void initializeStateVectorWithIndices(ExStateVector stateVector, int numWires)
{
    auto numElements = 1LL << numWires;
    std::vector<DblComplex> elements(numElements);
    // Set element[i] = i (as a real number)
    for (Index_t i = 0; i < numElements; ++i)
    {
        elements[i] = DblComplex(static_cast<double>(i), 0.0);
    }
    output("Initializing state vector elements to their index values...\n");
    ERRCHK(custatevecExStateVectorSetState(stateVector, elements.data(), CUDA_C_64F, 0, numElements,
                                           1));
    ERRCHK(custatevecExStateVectorSynchronize(stateVector));
}

//
// Format vector as Python-like list string: [0, 1, 2, 3]
//
std::string formatVector(const std::vector<int32_t>& vec)
{
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        result += std::to_string(vec[i]);
        if (i < vec.size() - 1)
            result += ", ";
    }
    result += "]";
    return result;
}

//
// Get wire ordering from state vector
//
std::vector<int32_t> getWireOrdering(ExStateVector stateVector)
{
    // Get number of wires from state vector
    int numWires = -1;
    ERRCHK(custatevecExStateVectorGetProperty(stateVector, CUSTATEVEC_EX_SV_PROP_NUM_WIRES,
                                              &numWires, sizeof(numWires)));
    // Get wire ordering from state vector
    std::vector<int32_t> wireOrdering(numWires);
    size_t expectedSize = numWires * sizeof(int32_t);
    ERRCHK(custatevecExStateVectorGetProperty(stateVector, CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING,
                                              wireOrdering.data(), expectedSize));
    return wireOrdering;
}

//
// Generate random permutation by shuffling [0, 1, 2, 3...]
//
std::vector<int32_t> generateRandomPermutation(int numWires, std::mt19937& gen)
{
    // Start with identity permutation [0, 1, 2, 3...]
    std::vector<int32_t> permutation(numWires);
    std::iota(permutation.begin(), permutation.end(), 0);
    // Shuffle the array to create a random permutation
    std::shuffle(permutation.begin(), permutation.end(), gen);
    return permutation;
}

//
// Apply index bit permutation
//
void applyIndexBitPermutation(ExStateVector stateVector, const std::vector<int32_t>& permutation,
                              custatevecExPermutationType_t permutationType)
{
    const char* typeStr =
        (permutationType == CUSTATEVEC_EX_PERMUTATION_SCATTER) ? "SCATTER" : "GATHER";
    output("Applying index bit permutation (%s type)...\n", typeStr);
    ERRCHK(custatevecExStateVectorPermuteIndexBits(stateVector, permutation.data(),
                                                   static_cast<int32_t>(permutation.size()),
                                                   permutationType));
}

//
// Verify the permutation by reading back state vector elements
//
bool verifyPermutation(
    ExStateVector stateVector, int numWires, const std::vector<int32_t>& originalOrdering,
    const std::vector<int32_t>& permutation, custatevecExPermutationType_t permutationType)
{
    bool success = true;

    // First validation: Check wire ordering matches expected permutation
    auto actualWireOrdering = getWireOrdering(stateVector);
    std::vector<int32_t> expectedWireOrdering(numWires);

    if (permutationType == CUSTATEVEC_EX_PERMUTATION_SCATTER)
    {
        // For SCATTER: expectedWireOrdering[permutation[i]] = originalOrdering[i]
        for (int i = 0; i < numWires; ++i)
            expectedWireOrdering[permutation[i]] = originalOrdering[i];
    }
    else
    {
        // For GATHER: expectedWireOrdering[i] = originalOrdering[permutation[i]]
        for (int i = 0; i < numWires; ++i)
            expectedWireOrdering[i] = originalOrdering[permutation[i]];
    }

    if (actualWireOrdering != expectedWireOrdering)
    {
        printf("ERROR: Wire ordering mismatch\n");
        printf("Expected: %s\n", formatVector(expectedWireOrdering).c_str());
        printf("Actual:   %s\n", formatVector(actualWireOrdering).c_str());
        success = false;
    }

    // Second validation: Check state vector elements using permuted wire ordering
    auto numElements = 1LL << numWires;
    std::vector<DblComplex> elements(numElements);
    ERRCHK(custatevecExStateVectorGetState(stateVector, elements.data(), CUDA_C_64F, 0, numElements,
                                           1));
    ERRCHK(custatevecExStateVectorSynchronize(stateVector));

    // Validate all elements first
    for (Index_t i = 0; i < numElements; ++i)
    {
        double realPart = elements[i].real();
        Index_t actualValue = static_cast<Index_t>(std::round(realPart));

        // Compute expected value using scatter bit permutation
        Index_t expectedValue = 0;
        for (size_t j = 0; j < actualWireOrdering.size(); ++j)
        {
            // Extract bit at position j and place it at position actualWireOrdering[j]
            Index_t srcBit = (i >> j) & 1;
            expectedValue |= (srcBit << actualWireOrdering[j]);
        }

        if (actualValue != expectedValue)
        {
            printf("ERROR: Element %lld expected %lld but got %lld\n", static_cast<long long>(i),
                   static_cast<long long>(expectedValue), static_cast<long long>(actualValue));
            success = false;
            break;
        }
    }

    // Then dump elements for display (without validation)
    output("State vector elements: [");
    const int maxPrint = std::min(16LL, numElements);
    for (int i = 0; i < maxPrint; ++i)
    {
        double realPart = elements[i].real();
        output("%.0f", realPart);
        if (i < maxPrint - 1)
            output(", ");
    }
    if (maxPrint < numElements)
    {
        output(", ...");
    }
    output("]\n");
    return success;
}

//
// Print wire ordering for comparison
//
void printWireOrderingComparison(
    const std::vector<int32_t>& original, const std::vector<int32_t>& permuted,
    const std::vector<int32_t>& permutation, custatevecExPermutationType_t permutationType)
{
    output("\nWire ordering comparison:\n");
    output("Input wire ordering:     %s\n", formatVector(original).c_str());
    if (permutationType == CUSTATEVEC_EX_PERMUTATION_SCATTER)
        output("Permutation (SCATTER):   %s\n", formatVector(permutation).c_str());
    else
        output("Permutation (GATHER):    %s\n", formatVector(permutation).c_str());
    output("Permuted wire ordering:  %s\n", formatVector(permuted).c_str());
}

int main(int argc, char* argv[])
{
    // Check for quiet mode option
    if (argc >= 2 && strcmp(argv[1], "-q") == 0)
        setOutputEnabled(false);

    const int numWires = 4;             // Use small number for easy verification
    const auto svDataType = CUDA_C_64F; // Use double precision for exact index representation

    output("=== Index Bit Permutation Example ===\n");
    output("This example demonstrates wire management using index bit permutation.\n");
    output("Story: Create state vector -> Get initial wire ordering -> Initialize with indices\n");
    output("       -> Loop 10 times: Generate+Apply permutation -> Show comparison -> Verify "
           "elements\n");
    output("       -> Finally revert to initial wire ordering\n\n");

    output("Step 1: Creating state vector and getting initial wire ordering\n");
    auto stateVector = createStateVector(svDataType, numWires);
    auto initialOrdering = getWireOrdering(stateVector);
    output("Initial wire ordering (%zu wires): %s\n", initialOrdering.size(),
           formatVector(initialOrdering).c_str());

    output("\nStep 2: Initializing state vector with index values\n");
    // State vector elements will be filled with state vector indices.
    initializeStateVectorWithIndices(stateVector, numWires);

    // Random number generator for permutation type selection
    std::mt19937 gen(12345); // Use constant seed for reproducible results
    std::uniform_int_distribution<int> dist(0, 1);

    bool success = true;

    output("\nApplying 10 random permutations:\n");
    for (int iteration = 1; iteration <= 10; ++iteration)
    {
        output("\n--- Iteration %d ---\n", iteration);

        output("Step 3: Generating and applying random permutation\n");
        auto permutation = generateRandomPermutation(numWires, gen);
        custatevecExPermutationType_t permutationType =
            (dist(gen) == 0) ? CUSTATEVEC_EX_PERMUTATION_SCATTER : CUSTATEVEC_EX_PERMUTATION_GATHER;
        const char* typeStr =
            (permutationType == CUSTATEVEC_EX_PERMUTATION_SCATTER) ? "SCATTER" : "GATHER";
        output("Applying permutation %s (%s type)...\n", formatVector(permutation).c_str(),
               typeStr);
        auto currentOrdering = getWireOrdering(stateVector);
        ERRCHK(custatevecExStateVectorPermuteIndexBits(stateVector, permutation.data(),
                                                       static_cast<int32_t>(permutation.size()),
                                                       permutationType));

        auto newOrdering = getWireOrdering(stateVector);
        printWireOrderingComparison(currentOrdering, newOrdering, permutation, permutationType);

        output("\nStep 4: Verifying state vector elements\n");
        bool iterationSuccess =
            verifyPermutation(stateVector, numWires, currentOrdering, permutation, permutationType);
        if (!iterationSuccess)
        {
            success = false;
        }
    }

    output("\nStep 5: Reverting to initial wire ordering\n");
    auto finalOrdering = getWireOrdering(stateVector);
    output("Applying revert permutation %s (SCATTER type)...\n",
           formatVector(finalOrdering).c_str());
    ERRCHK(custatevecExStateVectorPermuteIndexBits(stateVector, finalOrdering.data(),
                                                   static_cast<int32_t>(finalOrdering.size()),
                                                   CUSTATEVEC_EX_PERMUTATION_SCATTER));

    auto revertedOrdering = getWireOrdering(stateVector);
    printWireOrderingComparison(finalOrdering, revertedOrdering, finalOrdering,
                                CUSTATEVEC_EX_PERMUTATION_SCATTER);

    // Verify state vector elements after revert
    output("Verifying state vector elements after revert\n");
    bool revertElementsSuccess = verifyPermutation(
        stateVector, numWires, finalOrdering, finalOrdering, CUSTATEVEC_EX_PERMUTATION_SCATTER);

    // Verify we're back to initial ordering
    bool revertSuccess = (revertedOrdering == initialOrdering) && revertElementsSuccess;
    if (revertSuccess)
    {
        output("Successfully reverted to initial wire ordering\n");
    }
    else
    {
        printf("ERROR: Failed to revert to initial wire ordering\n");
        success = false;
    }

    ERRCHK(custatevecExStateVectorDestroy(stateVector));
    output("\n=== Summary ===\n");
    if (success)
    {
        output("Index bit permutation example completed successfully!\n");
        output("Applied 10 random permutations using both SCATTER and GATHER types.\n");
        output("Wire ordering was successfully updated after each permutation.\n");
        output("State vector elements were correctly transformed throughout the process.\n");
        output("Successfully reverted to initial wire ordering at the end.\n\n");
    }
    else
    {
        printf("Index bit permutation example encountered errors.\n\n");
    }
    printf("%s\n", success ? "PASSED" : "FAILED");
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
