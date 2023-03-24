/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <string.h>           // strcpy
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

#define SUPPORTS_MEMORY_POOL ( __CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
#if SUPPORTS_MEMORY_POOL

// upon success, this function should return 0, otherwise a nonzero value
int myMemPoolAlloc(void* ctx, void** ptr, size_t size, cudaStream_t stream) {
    cudaMemPool_t& pool = *static_cast<cudaMemPool_t*>(ctx);
    cudaError_t status = cudaMallocFromPoolAsync(ptr, size, pool, stream);
    return (int)status;
}

// upon success, this function should return 0, otherwise a nonzero value
int myMemPoolFree(void*, void* ptr, size_t, cudaStream_t stream) {
    cudaError_t status = cudaFreeAsync(ptr, stream);
    return (int)status;
}

int main(void) {
    // state vector
    const int nIndexBits   = 3;
    const int nSvSize      = (1 << nIndexBits);

    cuDoubleComplex h_sv[] = {{ 0.48, 0.0}, { 0.36, 0.0}, { 0.64, 0.0}, { 0.48, 0.0}, 
                              { 0.0,  0.0}, { 0.0,  0.0}, { 0.0,  0.0}, { 0.0,  0.0}};

    //----------------------------------------------------------------------------------------------
    // gates
    const int adjoint = 0;
    const custatevecMatrixLayout_t layout = CUSTATEVEC_MATRIX_LAYOUT_ROW;

    // Hadamard gate
    const int hTargets[] = {2};
    const uint32_t hNTargets = 1;
    const double Rsqrt2 = 1. / std::sqrt(2.);
    cuDoubleComplex hGate[] = {{Rsqrt2, 0.0}, {Rsqrt2, 0.0},
                               {Rsqrt2, 0.0}, {-Rsqrt2, 0.0}};

    // control-SWAP gate
    const int swapTargets[] = {0, 1};
    const uint32_t swapNTargets = 2;
    const int swapControls[] = {2};
    const uint32_t swapNControls = 1;
    cuDoubleComplex swapGate[] = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                  {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
                                  {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                  {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};

    // observable
    const int basisBits[] = {2};
    const uint32_t nBasisBits = 1;
    cuDoubleComplex observable[] = {{1.0, 0.0}, {0.0, 0.0},
                                    {0.0, 0.0}, {0.0, 0.0}};

    //----------------------------------------------------------------------------------------------
    // device configuration
    int deviceId;
    HANDLE_CUDA_ERROR( cudaGetDevice(&deviceId) );

    cudaError_t status;
    int isMemPoolSupported;
    status = cudaDeviceGetAttribute(&isMemPoolSupported, cudaDevAttrMemoryPoolsSupported, deviceId);
    if (status != cudaSuccess || !isMemPoolSupported) {
        printf("memory handler example WAIVED: CUDA Memory pools is not supported.\n");
        return EXIT_SUCCESS;
    }

    cudaMemPool_t memPool;
    HANDLE_CUDA_ERROR( cudaDeviceGetDefaultMemPool(&memPool, deviceId) );

    // avoid shrinking the pool 
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold);

    cudaStream_t stream;
    HANDLE_CUDA_ERROR( cudaStreamCreate(&stream) );

    //----------------------------------------------------------------------------------------------
    // data transfer of state vector
    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR( cudaMallocAsync((void**)&d_sv, nSvSize * sizeof(cuDoubleComplex), stream) );

    HANDLE_CUDA_ERROR( cudaMemcpyAsync(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex), 
                                       cudaMemcpyHostToDevice, stream) );

    //----------------------------------------------------------------------------------------------
    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );
    HANDLE_ERROR( custatevecSetStream(handle, stream) );

    // device memory handler
    custatevecDeviceMemHandler_t handler;
    handler.ctx = &memPool;
    handler.device_alloc = myMemPoolAlloc;
    handler.device_free = myMemPoolFree;
    strcpy(handler.name, "mempool");
    HANDLE_ERROR( custatevecSetDeviceMemHandler(handle, &handler) );

    // apply Hadamard gate
    HANDLE_ERROR( custatevecApplyMatrix(
                  handle, d_sv, CUDA_C_64F, nIndexBits, hGate, CUDA_C_64F,
                  layout, adjoint, hTargets, hNTargets, nullptr, nullptr, 0, 
                  CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0) );

    // apply control-SWAP gate
    HANDLE_ERROR( custatevecApplyMatrix(
                  handle, d_sv, CUDA_C_64F, nIndexBits, swapGate, CUDA_C_64F,
                  layout, adjoint, swapTargets, swapNTargets, swapControls, nullptr, swapNControls, 
                  CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0) );

    // apply Hadamard gate
    HANDLE_ERROR( custatevecApplyMatrix(
                  handle, d_sv, CUDA_C_64F, nIndexBits, hGate, CUDA_C_64F,
                  layout, adjoint, hTargets, hNTargets, nullptr, nullptr, 0, 
                  CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0) );

    // compute expectation
    double expectationValue;
    HANDLE_ERROR( custatevecComputeExpectation(
                  handle, d_sv, CUDA_C_64F, nIndexBits, &expectationValue, CUDA_R_64F, nullptr,
                  observable, CUDA_C_64F, layout, basisBits, nBasisBits,
                  CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0) );

    HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    // release device memory and stream
    HANDLE_CUDA_ERROR( cudaFreeAsync(d_sv, stream) );
    HANDLE_CUDA_ERROR( cudaStreamDestroy(stream) );

    double expectationValueResult = 0.9608;
    bool correct = almost_equal(expectationValue, expectationValueResult);
    if (correct) {
        printf("memory_handler example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("memory_handler example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
}

#else
int main(void) {
    printf("memory_handler example WAIVED : This example uses CUDA's built-in stream-ordered memory allocator, which requires CUDA 11.2+.\n");
    return EXIT_SUCCESS;
}
#endif