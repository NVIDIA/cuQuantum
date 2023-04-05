/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecSampler
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

double* binarySearch(double* first, double* last, const double value) {
    double* it;
    int count = last - first;
    while (count > 0) {
        it = first;
        int step = count / 2;
        it += step;
        if (*it < value) {
            first = ++it;
            count -= step + 1;
        }
        else {
            count = step;
        }
    }
    return first;
}

int main(int argc, char** argv) {

    const int nGlobalBits = 2;
    const int nLocalBits  = 2;
    const int nSubSvs     = (1 << nGlobalBits);
    const int subSvSize   = (1 << nLocalBits);
    
    const int nMaxShots  = 5;
    const int nShots     = 5;

    const int bitStringLen  = 4;
    const int bitOrdering[] = {0, 1, 2, 3};

    custatevecIndex_t bitStrings[nShots];
    custatevecIndex_t bitStrings_result[nShots] = {0b0011, 0b0011, 0b0111, 0b1011, 0b1110};

    // In real appliction, random numbers in range [0, 1) will be used.
    double randnums[] = {0.1, 0.2, 0.4, 0.6, 0.8};

    cuDoubleComplex h_sv[][subSvSize] = {{{ 0.000, 0.000}, { 0.000, 0.125}, { 0.000, 0.250}, { 0.000, 0.375}},
                                         {{ 0.000, 0.000}, { 0.000,-0.125}, { 0.000,-0.250}, { 0.000,-0.375}},
                                         {{ 0.125, 0.000}, { 0.125,-0.125}, { 0.125,-0.250}, { 0.125,-0.375}},
                                         {{-0.125, 0.000}, {-0.125,-0.125}, {-0.125,-0.250}, {-0.125,-0.375}}};

    custatevecSamplerDescriptor_t sampler[nSubSvs];

    cuDoubleComplex *d_sv[nSubSvs];

    // device allocation
    int numDevices;
    int devices[nSubSvs];
    if (argc == 1)
    {
        HANDLE_CUDA_ERROR( cudaGetDeviceCount(&numDevices) );
        for (int i = 0; i < nSubSvs; i++) {
            devices[i] = i % numDevices;
        }
    }
    else {
        numDevices = min(argc - 1, nSubSvs);
        for (int i = 0; i < numDevices; i++) {
            const int deviceId = atoi(argv[i + 1]);
            devices[i] = deviceId;
        }
        for (int i = numDevices; i < nSubSvs; i++) {
            devices[i] = devices[i % numDevices];
        }
    }

    printf("The following devices will be used in this sample: \n");
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        printf("  sub-SV #%d : device id %d\n", iSv, devices[iSv]);
    }

    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_sv[iSv], subSvSize * sizeof(cuDoubleComplex)) );
        HANDLE_CUDA_ERROR( cudaMemcpy(d_sv[iSv], h_sv[iSv], subSvSize * sizeof(cuDoubleComplex), 
                           cudaMemcpyHostToDevice) );
    }

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle[nSubSvs];
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_ERROR( custatevecCreate(&handle[iSv]) );
    }

    void* extraWorkspace[nSubSvs];
    size_t extraWorkspaceSizeInBytes[nSubSvs];

    // create sampler and check the size of external workspace
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_ERROR( custatevecSamplerCreate(
                      handle[iSv], d_sv[iSv], CUDA_C_64F, nLocalBits, &sampler[iSv], nMaxShots,
                      &extraWorkspaceSizeInBytes[iSv]) );
    }

    // allocate external workspace if necessary
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        if (extraWorkspaceSizeInBytes[iSv] > 0) {
            HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
            HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace[iSv], extraWorkspaceSizeInBytes[iSv]) );
        }
    }

    // sample preprocess
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_ERROR( custatevecSamplerPreprocess(
                      handle[iSv], sampler[iSv], extraWorkspace[iSv],
                      extraWorkspaceSizeInBytes[iSv]) );
    }

    // get norm of the sub state vectors
    double subNorms[nSubSvs];
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_ERROR( custatevecSamplerGetSquaredNorm(
                      handle[iSv], sampler[iSv], &subNorms[iSv]) );
    }

    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    }

    // get cumulative array
    double cumulativeArray[nSubSvs + 1];
    cumulativeArray[0] = 0.0;
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        cumulativeArray[iSv + 1] = cumulativeArray[iSv] + subNorms[iSv];
    }
    double norm = cumulativeArray[nSubSvs];

    // apply offset and norm
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_ERROR( custatevecSamplerApplySubSVOffset(
                      handle[iSv], sampler[iSv], iSv, nSubSvs, cumulativeArray[iSv], norm) );
    }

    // divide randnum array
    int shotOffsets[nSubSvs + 1];
    shotOffsets[0] = 0;
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        double* pos = binarySearch(randnums, randnums + nShots, cumulativeArray[iSv + 1] / norm);
        if (iSv == nSubSvs - 1) {
            pos = randnums + nShots;
        }
        shotOffsets[iSv + 1] = pos - randnums;
    }

    // sample bit strings
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        int shotOffset = shotOffsets[iSv];
        int nSubShots = shotOffsets[iSv + 1] - shotOffsets[iSv];
        if (nSubShots > 0) {
            HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
            HANDLE_ERROR( custatevecSamplerSample(
                          handle[iSv], sampler[iSv], &bitStrings[shotOffset], bitOrdering,
                          bitStringLen, &randnums[shotOffset], nSubShots,
                          CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER) );
        }
    }

    // destroy sampler descriptor and custatevec handle
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
        HANDLE_ERROR( custatevecSamplerDestroy(sampler[iSv]) );
        HANDLE_ERROR( custatevecDestroy(handle[iSv]) );
    }

    //----------------------------------------------------------------------------------------------

    bool correct = true;
    for (int i = 0; i < nShots; i++) {
        if (bitStrings[i] != bitStrings_result[i]) {
            correct = false;
            break;
        }
    }

    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaFree(d_sv[iSv]) );
        if (extraWorkspaceSizeInBytes[iSv] > 0) {
            HANDLE_CUDA_ERROR( cudaFree(extraWorkspace[iSv]) );
        }
    }

    if (correct) {
        printf("mgpu_sampler example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("mgpu_sampler example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
}
