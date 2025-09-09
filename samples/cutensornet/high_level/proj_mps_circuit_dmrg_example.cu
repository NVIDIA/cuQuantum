/* Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: Projection MPS Circuit DMRG #1

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <complex>
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>

#include <cuda_runtime.h>
#include <cutensornet.h>
#include <cublas_v2.h>

#define HANDLE_CUDA_ERROR(x)                                                         \
    {                                                                                \
        const auto err = x;                                                          \
        if (err != cudaSuccess)                                                      \
        {                                                                            \
            printf("CUDA error %s in line %d\n", cudaGetErrorString(err), __LINE__); \
            fflush(stdout);                                                          \
            std::abort();                                                            \
        }                                                                            \
    };

#define HANDLE_CUTN_ERROR(x)                                                                       \
    {                                                                                              \
        const auto err = x;                                                                        \
        if (err != CUTENSORNET_STATUS_SUCCESS)                                                     \
        {                                                                                          \
            printf("cuTensorNet error %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
            fflush(stdout);                                                                        \
            std::abort();                                                                          \
        }                                                                                          \
    };

#define HANDLE_CUBLAS_ERROR(x)                                                                     \
    {                                                                                              \
        const auto err = x;                                                                        \
        if (err != CUBLAS_STATUS_SUCCESS)                                                          \
        {                                                                                          \
            printf("cuBLAS error (status code %d) in line %d\n", static_cast<int>(err), __LINE__); \
            fflush(stdout);                                                                        \
            std::abort();                                                                          \
        }                                                                                          \
    };

// Helper function to compute maximum bond extents
std::vector<int64_t> getMaxBondExtents(const std::vector<int64_t>& stateModeExtents, int64_t maxdim = -1)
{
    const int32_t numQubits = stateModeExtents.size();
    std::vector<int64_t> cumprodLeft(numQubits), cumprodRight(numQubits);

    // Compute cumulative products from left and right
    cumprodLeft[0] = stateModeExtents[0];
    for (int32_t i = 1; i < numQubits; ++i)
    {
        cumprodLeft[i] = cumprodLeft[i - 1] * stateModeExtents[i];
    }

    cumprodRight[numQubits - 1] = stateModeExtents[numQubits - 1];
    for (int32_t i = numQubits - 2; i >= 0; --i)
    {
        cumprodRight[i] = cumprodRight[i + 1] * stateModeExtents[i];
    }

    std::vector<int64_t> maxBondExtents(numQubits - 1);
    for (int32_t i = 0; i < numQubits - 1; ++i)
    {
        int64_t minVal    = std::min(cumprodLeft[i], cumprodRight[i + 1]);
        maxBondExtents[i] = (maxdim > 0) ? std::min(minVal, maxdim) : minVal;
    }

    return maxBondExtents;
}

int main()
{
    static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");

    constexpr std::size_t fp64size = sizeof(double);

    // Sphinx: Projection MPS Circuit DMRG #2

    // Quantum state configuration
    const int32_t numQubits = 16;
    const std::vector<int64_t> qubitDims(numQubits, 2);
    std::cout << "DMRG quantum circuit with " << numQubits << " qubits\n";

    // Sphinx: Projection MPS Circuit DMRG #3

    // Initialize the cuTensorNet library
    HANDLE_CUDA_ERROR(cudaSetDevice(0));
    cutensornetHandle_t cutnHandle;
    HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
    std::cout << "Initialized cuTensorNet library on GPU 0\n";

    cublasHandle_t cublasHandle;
    HANDLE_CUBLAS_ERROR(cublasCreate(&cublasHandle));
    
    // Sphinx: Projection MPS Circuit DMRG #4

    // Define necessary quantum gate tensors in Host memory
    const double invsq2 = 1.0 / std::sqrt(2.0);
    const double pi     = 3.14159265358979323846;
    using GateData      = std::vector<std::complex<double>>;

    // CR(k) gate generator
    auto cRGate = [&pi](int32_t k)
    {
        const double phi = pi / std::pow(2.0, k);
        const GateData cr{{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                          {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                          {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
                          {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {std::cos(phi), std::sin(phi)}};
        return cr;
    };

    // Hadamard gate
    const GateData h_gateH{{invsq2, 0.0}, {invsq2, 0.0}, {invsq2, 0.0}, {-invsq2, 0.0}};

    // CR(k) gates (controlled rotations)
    std::vector<GateData> h_gateCR(numQubits);
    for (int32_t k = 0; k < numQubits; ++k)
    {
        h_gateCR[k] = cRGate(k + 1);
    }

    // Copy quantum gates into Device memory
    void* d_gateH{nullptr};
    std::vector<void*> d_gateCR(numQubits, nullptr);
    HANDLE_CUDA_ERROR(cudaMalloc(&d_gateH, 4 * (2 * fp64size)));
    for (int32_t k = 0; k < numQubits; ++k)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&(d_gateCR[k]), 16 * (2 * fp64size)));
    }

    HANDLE_CUDA_ERROR(cudaMemcpy(d_gateH, h_gateH.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
    for (int32_t k = 0; k < numQubits; ++k)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy(d_gateCR[k], h_gateCR[k].data(), 16 * (2 * fp64size), cudaMemcpyHostToDevice));
    }
    std::cout << "Copied quantum gates into GPU memory\n";

    // Sphinx: Projection MPS Circuit DMRG #5

    // Query free memory on Device and allocate a scratch buffer
    std::size_t freeSize{0}, totalSize{0};
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
    const std::size_t scratchSize = (freeSize - (freeSize % 4096)) / 2;
    void* d_scratch{nullptr};
    HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
    std::cout << "Allocated " << scratchSize << " bytes of scratch memory on GPU\n";

    // Create the initial quantum state
    cutensornetState_t quantumState;
    HANDLE_CUTN_ERROR(cutensornetCreateState(cutnHandle, CUTENSORNET_STATE_PURITY_PURE, numQubits, qubitDims.data(),
                                             CUDA_C_64F, &quantumState));
    std::cout << "Created the initial quantum state\n";

    // Construct the quantum circuit state (apply quantum gates)
    int64_t id;
    for (int32_t i = 0; i < numQubits; ++i)
    {
        HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
            cutnHandle, quantumState, 1, std::vector<int32_t>{{i}}.data(), d_gateH, nullptr, 1, 0, 1, &id));
        for (int32_t j = (i + 1); j < numQubits; ++j)
        {
            HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 2,
                                                                  std::vector<int32_t>{{j, i}}.data(), d_gateCR[j - i],
                                                                  nullptr, 1, 0, 1, &id));
        }
    }
    std::cout << "Applied quantum gates\n";

    // Sphinx: Projection MPS Circuit DMRG #6

    // Define the MPS representation and allocate memory buffers for the MPS tensors
    const int64_t maxExtent = 8;
    std::vector<int64_t> dimExtents(numQubits, 2);
    std::vector<int64_t> maxBondExtents = getMaxBondExtents(dimExtents, maxExtent);

    std::vector<std::vector<int64_t>> projMPSTensorExtents;
    std::vector<const int64_t*> projMPSTensorExtentsPtr(numQubits);
    std::vector<void*> d_projMPSTensors(numQubits, nullptr);
    std::vector<int64_t>  numElements(numQubits);
    std::vector<void*> d_envTensorsExtract(numQubits, nullptr);
    std::vector<void*> d_envTensorsInsert(numQubits, nullptr);


    // Sphinx: Projection MPS Circuit DMRG #7

    for (int32_t i = 0; i < numQubits; ++i)
    {
        if (i == 0)
        {
            // left boundary MPS tensor
            auto extent2 = std::min(maxBondExtents[i], maxExtent);
            projMPSTensorExtents.push_back({2, extent2});
            numElements[i] = 2 * extent2;
            HANDLE_CUDA_ERROR(cudaMalloc(&d_projMPSTensors[i], numElements[i] * (2 * fp64size)));
            //projMPSTensorStrides.push_back({1, 2});
        }
        else if (i == numQubits - 1)
        {
            // right boundary MPS tensor
            auto extent1 = std::min(maxBondExtents[i - 1], maxExtent);
            projMPSTensorExtents.push_back({extent1, 2});
            numElements[i] = extent1 * 2;
            HANDLE_CUDA_ERROR(cudaMalloc(&d_projMPSTensors[i], numElements[i] * (2 * fp64size)));
        }
        else
        {
            // middle MPS tensors
            auto extent1 = std::min(maxBondExtents[i - 1], maxExtent);
            auto extent3 = std::min(maxBondExtents[i], maxExtent);
            projMPSTensorExtents.push_back({extent1, 2, extent3});
            numElements[i] = extent1 * 2 * extent3;
            HANDLE_CUDA_ERROR(cudaMalloc(&d_projMPSTensors[i], numElements[i] * (2 * fp64size)));
        }
        projMPSTensorExtentsPtr[i] = projMPSTensorExtents[i].data();

        HANDLE_CUDA_ERROR(cudaMalloc(&d_envTensorsExtract[i], numElements[i] * (2 * fp64size)));
        HANDLE_CUDA_ERROR(cudaMalloc(&d_envTensorsInsert[i], numElements[i] * (2 * fp64size)));
    }

    // Initialize the vacuum state
    for (int32_t i = 0; i < numQubits; ++i)
    {
        std::vector<std::complex<double>> hostTensor(numElements[i]);
        for (int64_t j = 0; j < numElements[i]; ++j)
        {
            hostTensor[j] = {0.0, 0.0};
        }
        hostTensor[0] = {1.0, 0.0};
        HANDLE_CUDA_ERROR(
            cudaMemcpy(d_projMPSTensors[i], hostTensor.data(), numElements[i] * (2 * fp64size), cudaMemcpyHostToDevice));
    }
    std::cout << "Allocated GPU memory for projection MPS tensors\n";

    // Sphinx: Projection MPS Circuit DMRG #8

    // Prepare state projection MPS
    std::vector<cutensornetState_t> states    = {quantumState};
    std::vector<cuDoubleComplex> coefficients = {{1.0, 0.0}};

    // Environment specifications
    std::vector<cutensornetMPSEnvBounds_t> envSpecs;
    for (int32_t i = 0; i < numQubits; ++i)
    {
        cutensornetMPSEnvBounds_t spec;
        spec.lowerBound = i - 1;
        spec.upperBound = i + 1;
        envSpecs.push_back(spec);
    }

    cutensornetMPSEnvBounds_t initialOrthoSpec;
    initialOrthoSpec.lowerBound = -1;
    initialOrthoSpec.upperBound = numQubits;

    // Sphinx: Projection MPS Circuit DMRG #9

    // Create state projection MPS
    cutensornetStateProjectionMPS_t projectionMps;
    HANDLE_CUTN_ERROR(cutensornetCreateStateProjectionMPS(
        cutnHandle, states.size(), states.data(), coefficients.data(), false, envSpecs.size(), envSpecs.data(),
        CUTENSORNET_BOUNDARY_CONDITION_OPEN, numQubits, 0, projMPSTensorExtentsPtr.data(), 0,
        d_projMPSTensors.data(), &initialOrthoSpec, &projectionMps));
    std::cout << "Created state projection MPS\n";

    // Prepare the state projection MPS computation
    cutensornetWorkspaceDescriptor_t workDesc;
    HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
    HANDLE_CUTN_ERROR(cutensornetStateProjectionMPSPrepare(cutnHandle, projectionMps, scratchSize, workDesc, 0x0));

    int64_t worksize{0};
    HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                        CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH,
                                                        &worksize));
    std::cout << "Workspace size for MPS projection = " << worksize << " bytes\n";

    if (worksize <= scratchSize)
    {
        HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                                                        CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
    }
    else
    {
        std::cout << "ERROR: Insufficient workspace size on Device!\n";

        // Cleanup
        HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
        HANDLE_CUTN_ERROR(cutensornetDestroyStateProjectionMPS(projectionMps));
        HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));

        // Free GPU buffers
        for (int32_t i = 0; i < numQubits; i++)
        {
            HANDLE_CUDA_ERROR(cudaFree(d_projMPSTensors[i]));
            HANDLE_CUDA_ERROR(cudaFree(d_envTensorsExtract[i]));
            HANDLE_CUDA_ERROR(cudaFree(d_envTensorsInsert[i]));
        }
        HANDLE_CUDA_ERROR(cudaFree(d_scratch));
        for (auto ptr : d_gateCR) HANDLE_CUDA_ERROR(cudaFree(ptr));
        HANDLE_CUDA_ERROR(cudaFree(d_gateH));
        std::cout << "Freed memory on GPU\n";

        // Finalize the cuTensorNet library
        HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));

        std::abort();
    }

    // Sphinx: Projection MPS Circuit DMRG #10

    // DMRG iterations
    const int32_t numIterations = 5;
    std::cout << "Starting DMRG iterations\n";

    for (int32_t iter = 0; iter < numIterations; ++iter)
    {
        std::cout << "DMRG iteration " << iter + 1 << "/" << numIterations << std::endl;

        // Forward sweep
        for (int32_t i = 0; i < numQubits; ++i)
        {
            // Environment bounds for site i
            cutensornetMPSEnvBounds_t envBounds;
            envBounds.lowerBound = i - 1;
            envBounds.upperBound = i + 1;

            // Extract site tensor
            HANDLE_CUTN_ERROR(cutensornetStateProjectionMPSExtractTensor(
                cutnHandle, projectionMps, &envBounds, nullptr, d_envTensorsExtract[i], workDesc, 0x0));

            // Compute environment tensor
            HANDLE_CUTN_ERROR(cutensornetStateProjectionMPSComputeTensorEnv(cutnHandle, projectionMps, &envBounds, 0, 0,
                                                                            0, d_envTensorsInsert[i], 0,
                                                                            0, workDesc, 0x0));

            // Apply partial-fidelity scaling factor to the environment tensor
            double f_tau_sqrt;
            HANDLE_CUBLAS_ERROR(cublasDznrm2(cublasHandle, numElements[i], 
                        static_cast<const cuDoubleComplex*>(d_envTensorsInsert[i]), 1,
                        &f_tau_sqrt));
            HANDLE_CUDA_ERROR(cudaStreamSynchronize(0));

            if (f_tau_sqrt < std::sqrt(std::numeric_limits<double>::epsilon()))
            {
                std::cout << "ERROR: Scaling factor is zero!\n";
                std::abort();
            }
            cuDoubleComplex scaling_factor = {1.0 / f_tau_sqrt, 0.0};

            HANDLE_CUBLAS_ERROR(cublasZscal(cublasHandle, numElements[i], 
                        &scaling_factor,
                        static_cast<cuDoubleComplex*>(d_envTensorsInsert[i]), 1));

            // Insert updated tensor
            cutensornetMPSEnvBounds_t orthoSpec = envBounds;
            HANDLE_CUTN_ERROR(cutensornetStateProjectionMPSInsertTensor(cutnHandle, projectionMps, &envBounds,
                                                                        &orthoSpec, 0,
                                                                        d_envTensorsInsert[i], workDesc, 0x0));
        }

        // Backward sweep
        for (int32_t i = numQubits - 1; i >= 0; --i)
        {
            // Environment bounds for site i
            cutensornetMPSEnvBounds_t envBounds;
            envBounds.lowerBound = i - 1;
            envBounds.upperBound = i + 1;

            // Extract site tensor
            HANDLE_CUTN_ERROR(cutensornetStateProjectionMPSExtractTensor(
                cutnHandle, projectionMps, &envBounds, 0, d_envTensorsExtract[i], workDesc, 0x0));

            // Compute environment tensor
            HANDLE_CUTN_ERROR(cutensornetStateProjectionMPSComputeTensorEnv(cutnHandle, projectionMps, &envBounds, 0, 0,
                                                                            0, d_envTensorsInsert[i], 0,
                                                                            0, workDesc, 0x0));
            
            // Apply partial-fidelity scaling factor to the environment tensor
            double f_tau_sqrt;
            HANDLE_CUBLAS_ERROR(cublasDznrm2(cublasHandle, numElements[i], 
                        static_cast<const cuDoubleComplex*>(d_envTensorsInsert[i]), 1,
                        &f_tau_sqrt));
            HANDLE_CUDA_ERROR(cudaStreamSynchronize(0));

            if (f_tau_sqrt < std::sqrt(std::numeric_limits<double>::epsilon()))
            {
                std::cout << "ERROR: Scaling factor is zero!\n";
                std::abort();
            }
            cuDoubleComplex scaling_factor = {1.0 / f_tau_sqrt, 0.0};

            HANDLE_CUBLAS_ERROR(cublasZscal(cublasHandle, numElements[i], 
                        &scaling_factor,
                        static_cast<cuDoubleComplex*>(d_envTensorsInsert[i]), 1));

            // Insert updated tensor
            cutensornetMPSEnvBounds_t orthoSpec = envBounds;
            HANDLE_CUTN_ERROR(cutensornetStateProjectionMPSInsertTensor(cutnHandle, projectionMps, &envBounds,
                                                                        &orthoSpec, 0,
                                                                        d_envTensorsInsert[i], workDesc, 0x0));
        }

        std::cout << "Completed DMRG iteration " << iter + 1 << std::endl;
    }

    std::cout << "DMRG optimization completed\n";

    // Sphinx: Projection MPS Circuit DMRG #11

    // Cleanup
    HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
    HANDLE_CUTN_ERROR(cutensornetDestroyStateProjectionMPS(projectionMps));
    HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));

    // Free GPU buffers
    for (int32_t i = 0; i < numQubits; i++)
    {
        HANDLE_CUDA_ERROR(cudaFree(d_projMPSTensors[i]));
        HANDLE_CUDA_ERROR(cudaFree(d_envTensorsExtract[i]));
        HANDLE_CUDA_ERROR(cudaFree(d_envTensorsInsert[i]));
    }
    HANDLE_CUDA_ERROR(cudaFree(d_scratch));
    for (auto ptr : d_gateCR) HANDLE_CUDA_ERROR(cudaFree(ptr));
    HANDLE_CUDA_ERROR(cudaFree(d_gateH));
    std::cout << "Freed memory on GPU\n";

    HANDLE_CUBLAS_ERROR(cublasDestroy(cublasHandle));

    // Finalize the cuTensorNet library
    HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));
    std::cout << "Finalized the cuTensorNet library\n";

    return 0;
}
