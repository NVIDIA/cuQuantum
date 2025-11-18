/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

//
// Interoperability Demo: Dot Product with Z-Rotations
//
// This example demonstrates the interoperability features of cuStateVec Ex API
// by computing the dot product of quantum state vectors with Z-rotations using:
// 1. cuStateVec Ex APIs for state vector creation and Pauli rotations
// 2. GetResources APIs to extract GPU memory pointers and CUDA streams
// 3. Custom cuBLAS operations for direct dot product computation on GPU memory
// 4. Analytical validation showing <++++|RZ(θ)^⊗n|++++> = cos^n(θ)
//

#include <custatevecEx.h>              // custatevecEx API
#include <custatevecEx_ext.h>          // custatevecEx extension for communicator
#include <cublas_v2.h>                 // cuBLAS for dot product
#include <cuda_runtime.h>              // CUDA runtime
#include <cuComplex.h>                 // CUDA complex numbers
#include <stdlib.h>                    // exit()
#include <cstdio>                      // printf
#include <complex>                     // std::complex<>
#include <vector>                      // std::vector<>
#include <cmath>                       // std::abs, M_PI
#include <cstring>                     // strcmp
#include <cstdarg>                     // va_list, va_start, va_end
#include <numeric>                     // std::iota
#include <random>                      // std::random_device, std::mt19937, std::shuffle
#include <algorithm>                   // std::shuffle
#include "stateVectorConstruction.hpp" // Shared state vector construction module
#include "common.hpp"                  // Error checking utilities

// Use complex64 for state vector and matrix elements
constexpr cudaDataType_t svDataType = CUDA_C_32F;
typedef std::complex<float> ComplexType;
typedef cuFloatComplex CudaComplexType;

typedef custatevecExStateVectorDescriptor_t ExStateVector;

//
// Apply Hadamard gates to create |++++...> state (following pauli_functions.cpp style)
//
void applyH(ExStateVector stateVector, int numWires)
{
    // 2x2 unitary matrix in row-major order.
    ComplexType H[] = {{1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f}};
    for (auto& elm : H)
        elm *= 1.0f / sqrtf(2.0f);
    const int adjoint = 0;

    for (int target = 0; target < numWires; ++target)
    {
        ERRCHK(custatevecExApplyMatrix(stateVector, H, CUDA_C_32F, CUSTATEVEC_EX_MATRIX_DENSE,
                                       CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &target, 1, nullptr,
                                       nullptr, 0));
    }
}

//
// Apply Z-rotation to each qubit individually
//
// Rz(θ) gate matrix:
// Rz(θ) = [e^(-iθ/2)    0      ]
//         [0       e^(iθ/2) ]
//
// We apply e^(-iθ/2*Z) by calling custatevecExApplyPauliRotation with -θ/2
// (since API computes e^(i*angle*Z), we use negative angle)
//
void applyRotationByZ(ExStateVector stateVector, int numWires, float theta)
{
    // Apply Rz(θ) = e^(-iθ/2*Z) to each qubit
    // The API computes e^(i*angle*Z), so we pass -θ/2
    float angle = -theta / 2.0f;

    for (int wire = 0; wire < numWires; ++wire)
    {
        custatevecPauli_t pauli = CUSTATEVEC_PAULI_Z;
        int32_t target = wire;

        ERRCHK(custatevecExApplyPauliRotation(stateVector, angle, &pauli, &target, 1, nullptr,
                                              nullptr, 0));
    }
}

//
// Compute dot product using cuBLAS
//
ComplexType
computeDotProductCublas(int deviceId, const void* devicePtr1, cudaStream_t stream1,
                        const void* devicePtr2, cudaStream_t stream2, size_t numElements)
{
    ERRCHK_CUDA(cudaSetDevice(deviceId));
    // Synchronize on stream2
    ERRCHK_CUDA(cudaStreamSynchronize(stream2));

    // Create cuBLAS handle and compute dot product
    cublasHandle_t cublasHandle;
    ERRCHK_CUBLAS(cublasCreate(&cublasHandle));
    // Use stream 1
    ERRCHK_CUBLAS(cublasSetStream(cublasHandle, stream1));

    // Use single precision cuBLAS routine
    cuFloatComplex result;
    ERRCHK_CUBLAS(cublasCdotc(cublasHandle, static_cast<int>(numElements),
                              reinterpret_cast<const cuFloatComplex*>(devicePtr1), 1,
                              reinterpret_cast<const cuFloatComplex*>(devicePtr2), 1, &result));
    ERRCHK_CUDA(cudaStreamSynchronize(stream1));
    // cleanup
    ERRCHK_CUBLAS(cublasDestroy(cublasHandle));

    ComplexType dotProduct(cuCrealf(result), cuCimagf(result));
    output("Dot product computed: (%.6f, %.6f)\n", dotProduct.real(), dotProduct.imag());
    return dotProduct;
}

//
// Compute dot product for single-device configuration
//
ComplexType computeDotProductSingleDevice(ExStateVector sv1, ExStateVector sv2, int numWires)
{
    output("Compute dot product for single-device state vector");

    // Extract resources from both state vectors
    int32_t deviceId1, deviceId2;
    void* devicePtr1;
    void* devicePtr2;
    cudaStream_t stream1, stream2;
    custatevecHandle_t handle1, handle2;

    // Retrieve computing resource from state vectors
    // Single device state vector has one sub state vector.
    // The sub state vector index is 0.
    ERRCHK(custatevecExStateVectorGetResourcesFromDeviceSubSV(sv1, /*subSVIndex=*/0, &deviceId1,
                                                              &devicePtr1, &stream1, &handle1));
    ERRCHK(custatevecExStateVectorGetResourcesFromDeviceSubSV(sv2, /*subSVIndex=*/0, &deviceId2,
                                                              &devicePtr2, &stream2, &handle2));
    size_t numElements = 1LL << numWires;
    // Use the cuBLAS to compute the dot product
    // deviceId1 == deviceId2
    auto dotProduct =
        computeDotProductCublas(deviceId1, devicePtr1, stream1, devicePtr2, stream2, numElements);

    output("Single-device dot product: (%.6f, %.6f)\n", dotProduct.real(), dotProduct.imag());
    return dotProduct;
}

//
// Compute dot product for multi-device configuration
//
ComplexType computeDotProductMultiDevice(ExStateVector sv1, ExStateVector sv2, int numWires)
{
    // Multi-device state vector has multiple sub state vectors corresponding the number of
    // devices where the state vector is allocated.
    // Get the number of device sub-state vectors
    int32_t numDeviceSubSVs;
    ERRCHK(custatevecExStateVectorGetProperty(sv1, CUSTATEVEC_EX_SV_PROP_NUM_DEVICE_SUBSVS,
                                              &numDeviceSubSVs, sizeof(numDeviceSubSVs)));
    output("Number of device sub-state vectors: %d\n", numDeviceSubSVs);

    // Get the indices of device sub-state vectors
    std::vector<int32_t> deviceSubSVIndices(numDeviceSubSVs);
    ERRCHK(custatevecExStateVectorGetProperty(sv1, CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES,
                                              deviceSubSVIndices.data(),
                                              numDeviceSubSVs * sizeof(int32_t)));

    // Initialize the total dot product
    ComplexType totalDotProduct(0.0f, 0.0f);

    // Compute partial dot products for each device sub-state vector
    for (int i = 0; i < numDeviceSubSVs; ++i)
    {
        int32_t subSVIndex = deviceSubSVIndices[i];

        // Extract resources from both state vectors for this sub-SV
        int32_t deviceId1, deviceId2;
        void* devicePtr1;
        void* devicePtr2;
        cudaStream_t stream1, stream2;
        custatevecHandle_t handle1, handle2;

        ERRCHK(custatevecExStateVectorGetResourcesFromDeviceSubSV(sv1, subSVIndex, &deviceId1,
                                                                  &devicePtr1, &stream1, &handle1));
        ERRCHK(custatevecExStateVectorGetResourcesFromDeviceSubSV(sv2, subSVIndex, &deviceId2,
                                                                  &devicePtr2, &stream2, &handle2));

        // Calculate number of elements for this sub-state vector
        // Each device holds a portion of the full state vector
        size_t numLocalElements = (1LL << numWires) / numDeviceSubSVs;

        output("Sub-SV %d: deviceId1=%d, deviceId2=%d, numLocalElements=%zu\n", subSVIndex,
               deviceId1, deviceId2, numLocalElements);

        // Compute partial dot product using cuBLAS
        ComplexType partialDotProduct = computeDotProductCublas(
            deviceId1, devicePtr1, stream1, devicePtr2, stream2, numLocalElements);
        output("Partial dot product %d: (%.6f, %.6f)\n", i, partialDotProduct.real(),
               partialDotProduct.imag());
        // Add to total dot product
        totalDotProduct += partialDotProduct;
    }

    output("Multi-device dot product: (%.6f, %.6f)\n", totalDotProduct.real(),
           totalDotProduct.imag());
    return totalDotProduct;
}

//
// Compute dot product for multi-process configuration
//
ComplexType computeDotProductMultiProcess(ExStateVector sv1, ExStateVector sv2, int /*numWires*/)
{
    // Get the number of device sub-state vectors (should be 1 for multi-process)
    int32_t numDeviceSubSVs;
    ERRCHK(custatevecExStateVectorGetProperty(sv1, CUSTATEVEC_EX_SV_PROP_NUM_DEVICE_SUBSVS,
                                              &numDeviceSubSVs, sizeof(numDeviceSubSVs)));

    // Get the sub-state vector index for this process
    std::vector<int32_t> deviceSubSVIndices(numDeviceSubSVs);
    ERRCHK(custatevecExStateVectorGetProperty(sv1, CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES,
                                              deviceSubSVIndices.data(),
                                              numDeviceSubSVs * sizeof(int32_t)));

    // For multi-process, each process should have exactly one sub-state vector
    int32_t subSVIndex = deviceSubSVIndices[0];

    // Extract resources from both state vectors for this process's sub-SV
    int32_t deviceId1, deviceId2;
    void* devicePtr1;
    void* devicePtr2;
    cudaStream_t stream1, stream2;
    custatevecHandle_t handle1, handle2;

    ERRCHK(custatevecExStateVectorGetResourcesFromDeviceSubSV(sv1, subSVIndex, &deviceId1,
                                                              &devicePtr1, &stream1, &handle1));
    ERRCHK(custatevecExStateVectorGetResourcesFromDeviceSubSV(sv2, subSVIndex, &deviceId2,
                                                              &devicePtr2, &stream2, &handle2));

    // Get number of local wires to calculate local elements
    int32_t numLocalWires;
    ERRCHK(custatevecExStateVectorGetProperty(sv1, CUSTATEVEC_EX_SV_PROP_NUM_LOCAL_WIRES,
                                              &numLocalWires, sizeof(numLocalWires)));
    output("Process sub-SV index: %d, deviceId1=%d, deviceId2=%d, numLocalWires=%d\n", subSVIndex,
           deviceId1, deviceId2, numLocalWires);

    // Compute local dot product using cuBLAS
    size_t numLocalElements = 1LL << numLocalWires;
    ComplexType localDotProduct = computeDotProductCublas(deviceId1, devicePtr1, stream1,
                                                          devicePtr2, stream2, numLocalElements);
    // Get the communicator and perform allreduce to sum across all processes
    custatevecExCommunicatorDescriptor_t exCommunicator = getMultiProcessCommunicator();

    // Sum all local dot products across processes
    ComplexType totalDotProduct;
    ERRCHK_EXCOMM(exCommunicator->intf->allreduce(exCommunicator, &localDotProduct,
                                                  &totalDotProduct, 1, CUDA_C_32F));
    output("Multi-process dot product: (%.6f, %.6f)\n", totalDotProduct.real(),
           totalDotProduct.imag());
    return totalDotProduct;
}

//
// Compute dot product with configuration-based dispatching
//
ComplexType computeDotProduct(ExStateVector sv1, ExStateVector sv2, int numWires)
{
    // Step 1: Get current wire orderings from both state vectors
    std::vector<int32_t> wireOrdering1(numWires);
    std::vector<int32_t> wireOrdering2(numWires);

    ERRCHK(custatevecExStateVectorGetProperty(sv1, CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING,
                                              wireOrdering1.data(), numWires * sizeof(int32_t)));
    ERRCHK(custatevecExStateVectorGetProperty(sv2, CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING,
                                              wireOrdering2.data(), numWires * sizeof(int32_t)));

    auto dumpVector = [](const char* label, const std::vector<int32_t>& wireOrdering)
    {
        output("%s: [", label);
        for (size_t i = 0; i < wireOrdering.size(); ++i)
            output("%d%s", wireOrdering[i], (i < wireOrdering.size() - 1) ? " " : "");
        output("]\n");
    };

    output("Current wire orderings\n");
    dumpVector("sv1", wireOrdering1);
    dumpVector("sv2", wireOrdering2);

    // Step 2: Permute wires of sv2 to match the wire ordering of sv1
    std::vector<int> permutation(numWires);
    for (int idx = 0; idx < numWires; ++idx)
    {
        auto wire2 = wireOrdering2[idx];
        auto pos = std::find(wireOrdering1.begin(), wireOrdering1.end(), wire2);
        permutation[idx] = static_cast<int>(pos - wireOrdering1.begin());
    }
    ERRCHK(custatevecExStateVectorPermuteIndexBits(sv2, permutation.data(), numWires,
                                                   CUSTATEVEC_EX_PERMUTATION_SCATTER));
    dumpVector("permutation", permutation);

    // Validate wire orderings are identical.
    ERRCHK(custatevecExStateVectorGetProperty(sv1, CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING,
                                              wireOrdering1.data(), numWires * sizeof(int32_t)));
    ERRCHK(custatevecExStateVectorGetProperty(sv2, CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING,
                                              wireOrdering2.data(), numWires * sizeof(int32_t)));

    output("Permuted wire orderings\n");
    dumpVector("sv1", wireOrdering1);
    dumpVector("sv2", wireOrdering2);

    // Get the state vector distribution type
    custatevecExStateVectorDistributionType_t distributionType;
    ERRCHK(custatevecExStateVectorGetProperty(sv1, CUSTATEVEC_EX_SV_PROP_DISTRIBUTION_TYPE,
                                              &distributionType, sizeof(distributionType)));

    // Dispatch to appropriate dot product computation
    switch (distributionType)
    {
    case CUSTATEVEC_EX_SV_DISTRIBUTION_SINGLE_DEVICE:
        return computeDotProductSingleDevice(sv1, sv2, numWires);

    case CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_DEVICE:
        return computeDotProductMultiDevice(sv1, sv2, numWires);

    case CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS:
        return computeDotProductMultiProcess(sv1, sv2, numWires);
    default:
        std::exit(EXIT_FAILURE);
    }
}

//
// Validate against analytical result
//
bool validateDotProduct(const ComplexType& dotProduct, int numWires, float theta)
{
    // Validate: <++++|Rz(θ)^⊗n|++++> = cos^n(θ/2) for Rz gate
    float expected = powf(cosf(theta / 2.0f), numWires);
    float computed = std::abs(dotProduct);
    float err = std::abs(computed - expected);

    auto thetaInDeg = theta * 180.0f / M_PI;
    output("theta=%.1f°, dot_product=%.6f, expected=%.6f, err=%.2e\n", thetaInDeg, computed,
           expected, err);

    constexpr float ep = 5e-4f; // Tolerance for single precision
    return err <= ep;
}

int main(int argc, char* argv[])
{
    const int numWires = 20;

    // Bootstrap multi-process environment
    bootstrapMultiProcessEnvironment(&argc, &argv);

    output("Interoperability example: Dot Product with Z-Rotations\n");
    output("Number of wires: %d\n", numWires);

    // Configure and create state vectors
    auto svConfig = configureStateVector(argc, argv, numWires);

    // Check if double precision is requested (not supported in this sample)
    cudaDataType_t configDataType = getStateVectorDataType();
    if (configDataType == CUDA_C_64F)
    {
        output("Error: Double precision (c128) is not supported in this sample\n");
        ERRCHK(custatevecExDictionaryDestroy(svConfig));
        exit(EXIT_FAILURE);
    }

    // Create state vectors by using the same config
    auto stateVector1 = createStateVector(svConfig);
    auto stateVector2 = createStateVector(svConfig);
    ERRCHK(custatevecExDictionaryDestroy(svConfig));

    // Set random wire orderings
    // Wire ordering can be permuted during operations, such as gate application
    // on global wires.  Mimic this effect by reassigning random wire ordering.
    output("Applying random wire orderings to state vectors\n");

    // Generate the default wire orderings
    std::vector<int32_t> wireOrdering(numWires);
    std::iota(wireOrdering.begin(), wireOrdering.end(), 0);
    // Randomly shuffle wire ordering
    std::mt19937 gen(10202025);
    std::shuffle(wireOrdering.begin(), wireOrdering.end(), gen);
    ERRCHK(
        custatevecExStateVectorReassignWireOrdering(stateVector1, wireOrdering.data(), numWires));
    std::shuffle(wireOrdering.begin(), wireOrdering.end(), gen);
    ERRCHK(
        custatevecExStateVectorReassignWireOrdering(stateVector2, wireOrdering.data(), numWires));
    // Set |00..0>
    ERRCHK(custatevecExStateVectorSetZeroState(stateVector1));
    ERRCHK(custatevecExStateVectorSetZeroState(stateVector2));

    bool pass = true;

    // Prepare |++++> state by applying H gates for all wires
    applyH(stateVector1, numWires);
    applyH(stateVector2, numWires);

    // Dot product computation

    // Compute the dot product and validate result: <++++|++++> = 1
    auto dotProduct = computeDotProduct(stateVector1, stateVector2, numWires);
    pass &= validateDotProduct(dotProduct, numWires, 0.);

    const float theta = (15.0f / 180.0f) * M_PI; // 15 degrees in radian
    // Apply RZ(θ) for all wires in stateVector2
    applyRotationByZ(stateVector2, numWires, theta);
    // Compute the dot product and validate result: <++++|RZ(θ)^⊗n|++++> = cos^n(θ)
    dotProduct = computeDotProduct(stateVector1, stateVector2, numWires);
    pass &= validateDotProduct(dotProduct, numWires, theta);

    // Cleanup
    ERRCHK(custatevecExStateVectorDestroy(stateVector1));
    ERRCHK(custatevecExStateVectorDestroy(stateVector2));

    // Finalize multi-process environment
    finalizeMultiProcessEnvironment();

    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
