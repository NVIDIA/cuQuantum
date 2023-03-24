/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecBatchMeasurementWithOffset
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(int argc, char** argv) {

    const int nGlobalBits  = 2;
    const int nLocalBits   = 2;
    const int nSubSvs      = (1 << nGlobalBits);
    const int subSvSize    = (1 << nLocalBits);
    const int bitStringLen = 2;

    const int bitOrdering[] = {1, 0};

    int bitString[bitStringLen];
    const int bitString_result[] = {0, 0};

    // In real appliction, random number in range [0, 1) will be used.
    const double randnum = 0.72; 

    cuDoubleComplex h_sv[][subSvSize]        = {{{ 0.000, 0.000}, { 0.000, 0.125}, { 0.000, 0.250}, { 0.000, 0.375}},
                                                {{ 0.000, 0.000}, { 0.000,-0.125}, { 0.000,-0.250}, { 0.000,-0.375}},
                                                {{ 0.125, 0.000}, { 0.125,-0.125}, { 0.125,-0.250}, { 0.125,-0.375}},
                                                {{-0.125, 0.000}, {-0.125,-0.125}, {-0.125,-0.250}, {-0.125,-0.375}}};
    cuDoubleComplex h_sv_result[][subSvSize] = {{{ 0.0,      0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}}, 
                                                {{ 0.0,      0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}},
                                                {{ 0.707107, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}},
                                                {{-0.707107, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}}};
   
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

    // get abs2sum for each sub state vector
    double abs2SumArray[nSubSvs];
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_ERROR( custatevecAbs2SumArray(
                      handle[iSv], d_sv[iSv], CUDA_C_64F, nLocalBits, &abs2SumArray[iSv], nullptr,
                      0, nullptr, nullptr, 0) );
    }

    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    }

    // get cumulative array
    double cumulativeArray[nSubSvs + 1];
    cumulativeArray[0] = 0.0;
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        cumulativeArray[iSv + 1] = cumulativeArray[iSv] + abs2SumArray[iSv];
    }

    // measurement
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        if (cumulativeArray[iSv] <= randnum && randnum < cumulativeArray[iSv + 1]) {
            double norm = cumulativeArray[nSubSvs];
            double offset = cumulativeArray[iSv];
            HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
            HANDLE_ERROR( custatevecBatchMeasureWithOffset(
                          handle[iSv], d_sv[iSv], CUDA_C_64F, nLocalBits, bitString, bitOrdering,
                          bitStringLen, randnum, CUSTATEVEC_COLLAPSE_NONE, offset, norm) );
        }
    }

    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    }

    // get abs2Sum after collapse
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_ERROR( custatevecAbs2SumArray(
                      handle[iSv], d_sv[iSv], CUDA_C_64F, nLocalBits, &abs2SumArray[iSv], nullptr,
                      0, bitString, bitOrdering, bitStringLen) );
    }

    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    }

    // get norm after collapse
    double norm = 0.0;
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        norm += abs2SumArray[iSv];
    }

    // collapse sub state vectors
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_ERROR( custatevecCollapseByBitString(
                      handle[iSv], d_sv[iSv], CUDA_C_64F, nLocalBits, bitString, bitOrdering,
                      bitStringLen, norm) );
    }

    // destroy handle
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[iSv]) );
        HANDLE_ERROR( custatevecDestroy(handle[iSv]) );
    }

    //----------------------------------------------------------------------------------------------

    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaMemcpy(h_sv[iSv], d_sv[iSv], subSvSize * sizeof(cuDoubleComplex),
                           cudaMemcpyDeviceToHost) );
    }

    bool correct = true;
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        for (int i = 0; i < subSvSize; i++) {
            if (!almost_equal(h_sv[iSv][i], h_sv_result[iSv][i])) {
                correct = false;
                break;
            }
        }
    }

    for (int i = 0; i < bitStringLen; i++) {
        if (bitString[i] != bitString_result[i]) {
            correct = false;
            break;
        }
    }
      
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaFree(d_sv[iSv]) );
    }

    if (correct) {
        printf("mgpu_batch_measure example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("mgpu_batch_measure example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}