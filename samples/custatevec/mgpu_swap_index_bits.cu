/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecMultiDeviceSwapIndexBits
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

// In this example, all the available devices (up to 4 devices) will be used by default.
// ./mgpu_swap_index_bits
//
// When device ids are given as additional inputs, the specified devices will be used.
// ./mgpu_swap_index_bits 0 1

int main(int argc, char** argv) {

    const int nGlobalIndexBits = 2;
    const int nLocalIndexBits  = 1;

    const int nSubSvs   = (1 << nGlobalIndexBits);
    const int subSvSize = (1 << nLocalIndexBits);

    const int nMaxDevices = nSubSvs;

    // specify the type of device network topology to optimize the data transfer sequence.
    // CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH provides better performance for devices connected
    // via NVLink with an NVSwitch or PCIe device network with a single PCIe switch.
    // CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH provides better performance for devices connected
    // by full mesh connection.
    const custatevecDeviceNetworkType_t deviceNetworkType = CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH;

    // swap 0th and 2nd qubits
    const int nIndexBitSwaps = 1;
    const int2 indexBitSwaps[] = {{0, 2}};

    // swap the state vector elements only if 1st qubit is 1
    const int maskLen = 1;
    int maskBitString[] = {1};
    int maskOrdering[] = {1};

    // 0.2|001> + 0.4|011> - 0.4|101> - 0.8|111>
    cuDoubleComplex h_sv[][subSvSize] = {{{ 0.0, 0.0}, { 0.2, 0.0}},
                                         {{ 0.0, 0.0}, { 0.4, 0.0}},
                                         {{ 0.0, 0.0}, {-0.4, 0.0}},
                                         {{ 0.0, 0.0}, {-0.8, 0.0}}};

    // 0.2|001> + 0.4|110> - 0.4|101> - 0.8|111>
    cuDoubleComplex h_sv_result[][subSvSize] = {{{ 0.0, 0.0}, { 0.2, 0.0}},
                                                {{ 0.0, 0.0}, { 0.0, 0.0}}, 
                                                {{ 0.0, 0.0}, {-0.4, 0.0}},
                                                {{ 0.4, 0.0}, {-0.8, 0.0}}};

    cuDoubleComplex *d_sv[nSubSvs];

    // device allocation
    int nDevices;
    int devices[nMaxDevices];

    if (argc == 1) {
        HANDLE_CUDA_ERROR( cudaGetDeviceCount(&nDevices) );
        nDevices = min(nDevices, nMaxDevices);
        for (int i = 0; i < nDevices; i++) {
            devices[i] = i;
        }
    }
    else {
        nDevices = min(argc - 1, nMaxDevices);
        for (int i = 0; i < nDevices; i++) {
            const int deviceId = atoi(argv[i + 1]);
            devices[i] = deviceId;
        }
    }

    // check if device ids do not duplicate
    for (int i0 = 0; i0 < nDevices - 1; i0++) {
        for (int i1 = i0 + 1; i1 < nDevices; i1++) {
            if (devices[i0] == devices[i1]) {
                printf("device id %d is defined more than once.\n", devices[i0]);
                return EXIT_FAILURE;
            }
        }
    }

    // enable P2P access
    for (int i0 = 0; i0 < nDevices; i0++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[i0]) );
        for (int i1 = 0; i1 < nDevices; i1++) {
            if (i0 == i1) {
                continue;
            }
            int canAccessPeer;
            HANDLE_CUDA_ERROR( cudaDeviceCanAccessPeer(&canAccessPeer, devices[i0], devices[i1]) );
            if (canAccessPeer == 0) {
                printf("P2P access between device id %d and %d is unsupported.\n",
                       devices[i0], devices[i1]);
                return EXIT_SUCCESS;
            }
            HANDLE_CUDA_ERROR( cudaDeviceEnablePeerAccess(devices[i1], 0) );
        }
    }

    // define which device stores each sub state vector
    int subSvLayout[nSubSvs];
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        subSvLayout[iSv] = devices[iSv % nDevices];
    }

    printf("The following devices will be used in this sample: \n");
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        printf("  sub-SV #%d : device id %d\n", iSv, subSvLayout[iSv]);
    }

    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(subSvLayout[iSv]) );
        HANDLE_CUDA_ERROR( cudaMalloc(&d_sv[iSv], subSvSize * sizeof(cuDoubleComplex)) );
        HANDLE_CUDA_ERROR( cudaMemcpy(d_sv[iSv], h_sv[iSv], subSvSize * sizeof(cuDoubleComplex), 
                           cudaMemcpyHostToDevice) );
    }

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handles[nMaxDevices];
    for (int i = 0; i < nDevices; i++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[i]) );
        HANDLE_ERROR( custatevecCreate(&handles[i]) );
    }

    // bit swap
    HANDLE_ERROR( custatevecMultiDeviceSwapIndexBits(
                  handles, nDevices, (void**)d_sv, CUDA_C_64F, nGlobalIndexBits, nLocalIndexBits,
                  indexBitSwaps, nIndexBitSwaps, maskBitString, maskOrdering, maskLen,
                  deviceNetworkType) );

    // destroy handle
    for (int i = 0; i < nDevices; i++) {
        HANDLE_CUDA_ERROR( cudaSetDevice(devices[i]) );
        HANDLE_ERROR( custatevecDestroy(handles[i]) );
    }

    HANDLE_CUDA_ERROR( cudaSetDevice(devices[0]) );

    //----------------------------------------------------------------------------------------------

    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaMemcpy(h_sv[iSv], d_sv[iSv], subSvSize * sizeof(cuDoubleComplex),
                           cudaMemcpyDeviceToHost) );
    }

    bool correct = true;
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        for (int j = 0; j < subSvSize; j++) {
            if (!almost_equal(h_sv[iSv][j], h_sv_result[iSv][j])) {
                correct = false;
                break;
            }
        }
    }

    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaFree(d_sv[iSv]) );
    }

    if (correct) {
        printf("mgpu_swap_index_bits example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("mgpu_swap_index_bits example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}
