/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: #1

#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cutensornet.h>

#define HANDLE_ERROR(x)                                                                 \
    do {                                                                                \
        const auto err = x;                                                             \
        if (err != CUTENSORNET_STATUS_SUCCESS)                                          \
        {                                                                               \
            printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
            fflush(stdout);                                                             \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

#define HANDLE_CUDA_ERROR(x)                                                          \
    do {                                                                              \
        const auto err = x;                                                           \
        if (err != cudaSuccess)                                                       \
        {                                                                             \
            printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
            fflush(stdout);                                                           \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

// Usage: DEV_ATTR(cudaDevAttrClockRate, deviceId)
#define DEV_ATTR(ENUMCONST, DID)                                                   \
    ({ int v;                                                                       \
       HANDLE_CUDA_ERROR(cudaDeviceGetAttribute(&v, ENUMCONST, DID));               \
       v; })


struct GPUTimer
{
    GPUTimer(cudaStream_t stream) : stream_(stream)
    {
        HANDLE_CUDA_ERROR(cudaEventCreate(&start_));
        HANDLE_CUDA_ERROR(cudaEventCreate(&stop_));
    }

    ~GPUTimer()
    {
        HANDLE_CUDA_ERROR(cudaEventDestroy(start_));
        HANDLE_CUDA_ERROR(cudaEventDestroy(stop_));
    }

    void start() { HANDLE_CUDA_ERROR(cudaEventRecord(start_, stream_)); }

    float seconds()
    {
        HANDLE_CUDA_ERROR(cudaEventRecord(stop_, stream_));
        HANDLE_CUDA_ERROR(cudaEventSynchronize(stop_));
        float time;
        HANDLE_CUDA_ERROR(cudaEventElapsedTime(&time, start_, stop_));
        return time * 1e-3;
    }

private:
    cudaEvent_t start_, stop_;
    cudaStream_t stream_;
};

int main()
{
    static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");

    bool verbose = true;

    // Check cuTensorNet version
    const size_t cuTensornetVersion = cutensornetGetVersion();
    if (verbose) printf("cuTensorNet version: %ld\n", cuTensornetVersion);

    // Set GPU device
    int numDevices{0};
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
    const int deviceId = 0;
    HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

    if (verbose)
    {
        printf("===== device info ======\n");
        printf("GPU-local-id:%d\n", deviceId);
        printf("GPU-name:%s\n", prop.name);
        printf("GPU-clock:%d\n", DEV_ATTR(cudaDevAttrClockRate, deviceId));
        printf("GPU-memoryClock:%d\n", DEV_ATTR(cudaDevAttrMemoryClockRate, deviceId));
        printf("GPU-nSM:%d\n", prop.multiProcessorCount);
        printf("GPU-major:%d\n", prop.major);
        printf("GPU-minor:%d\n", prop.minor);
        printf("========================\n");
    }

    typedef float floatType;
    cudaDataType_t typeData              = CUDA_R_32F;
    cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F;

    if (verbose) printf("Included headers and defined data types\n");

    // Sphinx: #2
    /**************************************************************************************
     * Computing: R_{k,l} = A_{a,b,c,d,e,f} B_{b,g,h,e,i,j} C_{m,a,g,f,i,k} D_{l,c,h,d,j,m}
     **************************************************************************************/

    constexpr int32_t numInputs = 4;

    // Create vectors of tensor modes
    std::vector<std::vector<int32_t>> tensorModes{ // for input tensors & output tensor
        // input tensors
        {'a', 'b', 'c', 'd', 'e', 'f'}, // tensor A
        {'b', 'g', 'h', 'e', 'i', 'j'}, // tensor B
        {'m', 'a', 'g', 'f', 'i', 'k'}, // tensor C
        {'l', 'c', 'h', 'd', 'j', 'm'}, // tensor D
        // output tensor
        {'k', 'l'}, // tensor R
    };

    // Set mode extents
    int64_t sameExtent = 16; // setting same extent for simplicity. In principle extents can differ.
    std::unordered_map<int32_t, int64_t> extent;
    for (auto& vec : tensorModes)
    {
        for (auto& mode : vec)
        {
            extent[mode] = sameExtent;
        }
    }

    // Create a vector of extents for each tensor
    std::vector<std::vector<int64_t>> tensorExtents; // for input tensors & output tensor
    tensorExtents.resize(numInputs + 1);             // hold inputs + output tensors
    for (int32_t t = 0; t < numInputs + 1; ++t)
    {
        for (auto& mode : tensorModes[t]) tensorExtents[t].push_back(extent[mode]);
    }

    if (verbose) printf("Defined tensor network, modes, and extents\n");

    // Sphinx: #3
    /*****************
     * Allocating data
     *****************/

    std::vector<size_t> tensorElements(numInputs + 1); // for input tensors & output tensor
    std::vector<size_t> tensorSizes(numInputs + 1);    // for input tensors & output tensor
    size_t totalSize = 0;
    for (int32_t t = 0; t < numInputs + 1; ++t)
    {
        size_t numElements = 1;
        for (auto& mode : tensorModes[t]) numElements *= extent[mode];
        tensorElements[t] = numElements;

        tensorSizes[t] = sizeof(floatType) * numElements;
        totalSize += tensorSizes[t];
    }

    if (verbose) printf("Total GPU memory used for tensor storage: %.2f GiB\n", (totalSize) / 1024. / 1024. / 1024);

    void* tensorData_d[numInputs + 1]; // for input tensors & output tensor
    for (int32_t t = 0; t < numInputs + 1; ++t)
    {
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&tensorData_d[t], tensorSizes[t]));
    }

    floatType* tensorData_h[numInputs + 1]; // for input tensors & output tensor
    for (int32_t t = 0; t < numInputs + 1; ++t)
    {
        tensorData_h[t] = (floatType*)malloc(tensorSizes[t]);
        if (tensorData_h[t] == NULL)
        {
            printf("Error: Host memory allocation failed!\n");
            return -1;
        }
    }

    /*****************
     * Initialize data
     *****************/

    // init output tensor to all 0s
    memset(tensorData_h[numInputs], 0, tensorSizes[numInputs]);
    // init input tensors to random values
    for (int32_t t = 0; t < numInputs; ++t)
    {
        for (size_t e = 0; e < tensorElements[t]; ++e) tensorData_h[t][e] = ((floatType)rand()) / RAND_MAX;
    }
    // copy input data to device buffers
    for (int32_t t = 0; t < numInputs; ++t)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy(tensorData_d[t], tensorData_h[t], tensorSizes[t], cudaMemcpyHostToDevice));
    }

    // Sphinx: #4
    /*************
     * cuTensorNet
     *************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    cutensornetHandle_t handle;
    HANDLE_ERROR(cutensornetCreate(&handle));

    if (verbose) printf("Allocated GPU memory for data, initialized data, and created library handle\n");

    /****************
     * Create Network
     ****************/

    // Set up tensor network
    cutensornetNetworkDescriptor_t networkDesc;
    HANDLE_ERROR(cutensornetCreateNetwork(handle, &networkDesc));

    int64_t tensorIDs[numInputs]; // for input tensors
    // attach the input tensors to the network
    for (int32_t t = 0; t < numInputs; ++t)
    {
        HANDLE_ERROR(cutensornetNetworkAppendTensor(handle,
                                                    networkDesc,
                                                    tensorModes[t].size(),
                                                    tensorExtents[t].data(),
                                                    tensorModes[t].data(),
                                                    NULL,
                                                    typeData,
                                                    &tensorIDs[t]));
    }

    // set the output tensor
    HANDLE_ERROR(cutensornetNetworkSetOutputTensor(handle,
                                                   networkDesc,
                                                   tensorModes[numInputs].size(),
                                                   tensorModes[numInputs].data(),
                                                   typeData));

    // set the network compute type
    HANDLE_ERROR(cutensornetNetworkSetAttribute(handle,
                                                networkDesc,
                                                CUTENSORNET_NETWORK_COMPUTE_TYPE,
                                                &typeCompute,
                                                sizeof(typeCompute)));
    if (verbose) printf("Initialized the cuTensorNet library and created a tensor network descriptor\n");

    // Sphinx: #5
    /******************************************************
     * Choose workspace limit based on available resources.
     ******************************************************/

    size_t freeMem, totalMem;
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    uint64_t workspaceLimit = (uint64_t)((double)freeMem * 0.9);
    if (verbose) printf("Workspace limit = %lu\n", workspaceLimit);

    /*******************************
     * Find "optimal" contraction order and slicing
     *******************************/

    cutensornetContractionOptimizerConfig_t optimizerConfig;
    HANDLE_ERROR(cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig));

    // Set the desired number of hyper-samples (defaults to 0)
    int32_t num_hypersamples = 8;
    HANDLE_ERROR(cutensornetContractionOptimizerConfigSetAttribute(handle,
                                                                   optimizerConfig,
                                                                   CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES,
                                                                   &num_hypersamples,
                                                                   sizeof(num_hypersamples)));

    // Create contraction optimizer info and find an optimized contraction path
    cutensornetContractionOptimizerInfo_t optimizerInfo;
    HANDLE_ERROR(cutensornetCreateContractionOptimizerInfo(handle, networkDesc, &optimizerInfo));

    HANDLE_ERROR(cutensornetContractionOptimize(handle,
                                                networkDesc,
                                                optimizerConfig,
                                                workspaceLimit,
                                                optimizerInfo));

    // Query the number of slices the tensor network execution will be split into
    int64_t numSlices = 0;
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                 optimizerInfo,
                                                                 CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                                                                 &numSlices,
                                                                 sizeof(numSlices)));
    assert(numSlices > 0);

    if (verbose) printf("Found an optimized contraction path using cuTensorNet optimizer\n");

    // Sphinx: #6
    /*******************************
     * Create workspace descriptor, allocate workspace, and set it.
     *******************************/

    cutensornetWorkspaceDescriptor_t workDesc;
    HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));

    int64_t requiredWorkspaceSize = 0;
    HANDLE_ERROR(cutensornetWorkspaceComputeContractionSizes(handle,
                                                             networkDesc,
                                                             optimizerInfo,
                                                             workDesc));

    HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(handle,
                                                   workDesc,
                                                   CUTENSORNET_WORKSIZE_PREF_MIN,
                                                   CUTENSORNET_MEMSPACE_DEVICE,
                                                   CUTENSORNET_WORKSPACE_SCRATCH,
                                                   &requiredWorkspaceSize));

    void* work = nullptr;
    HANDLE_CUDA_ERROR(cudaMalloc(&work, requiredWorkspaceSize));

    HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle,
                                               workDesc,
                                               CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH,
                                               work,
                                               requiredWorkspaceSize));

    if (verbose) printf("Allocated and set up the GPU workspace\n");

    // Sphinx: #7
    /**************************
     * Prepare the contraction.
     **************************/

    HANDLE_ERROR(cutensornetNetworkPrepareContraction(handle,
                                                      networkDesc,
                                                      workDesc));

    // set tensor's data buffers and strides
    for (int32_t t = 0; t < numInputs; ++t)
    {
        HANDLE_ERROR(cutensornetNetworkSetInputTensorMemory(handle,
                                                            networkDesc,
                                                            tensorIDs[t],
                                                            tensorData_d[t],
                                                            NULL));
    }
    HANDLE_ERROR(cutensornetNetworkSetOutputTensorMemory(handle,
                                                         networkDesc,
                                                         tensorData_d[numInputs],
                                                         NULL));
    /****************************************************************
     * Optional: Auto-tune the contraction to pick the fastest kernel
     *           for each pairwise tensor contraction.
     ****************************************************************/
    cutensornetNetworkAutotunePreference_t autotunePref;
    HANDLE_ERROR(cutensornetCreateNetworkAutotunePreference(handle,
                                                            &autotunePref));

    const int numAutotuningIterations = 5; // may be 0
    HANDLE_ERROR(cutensornetNetworkAutotunePreferenceSetAttribute(handle,
                                                                  autotunePref,
                                                                  CUTENSORNET_NETWORK_AUTOTUNE_MAX_ITERATIONS,
                                                                  &numAutotuningIterations,
                                                                  sizeof(numAutotuningIterations)));

    // Modify the network again to find the best pair-wise contractions
    HANDLE_ERROR(cutensornetNetworkAutotuneContraction(handle,
                                                       networkDesc,
                                                       workDesc,
                                                       autotunePref,
                                                       stream));

    HANDLE_ERROR(cutensornetDestroyNetworkAutotunePreference(autotunePref));

    if (verbose) printf("Prepared the network contraction for cuTensorNet and optionally auto-tuned it\n");

    // Sphinx: #8
    /****************************************
     * Execute the tensor network contraction
     ****************************************/

    // Create a cutensornetSliceGroup_t object from a range of slice IDs
    cutensornetSliceGroup_t sliceGroup{};
    HANDLE_ERROR(cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup));

    GPUTimer timer{stream};
    double minTimeCUTENSORNET = 1e100;
    const int numRuns         = 3; // number of repeats to get stable performance results
    for (int i = 0; i < numRuns; ++i)
    {
        // reset the output tensor data
        HANDLE_CUDA_ERROR(cudaMemcpy(tensorData_d[numInputs], tensorData_h[numInputs], tensorSizes[numInputs], cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

        /*
         * Contract all slices of the tensor network
         */
        timer.start();

        int32_t accumulateOutput = 0; // output tensor data will be overwritten
        HANDLE_ERROR(cutensornetNetworkContract(handle,
                                                networkDesc,
                                                accumulateOutput,
                                                workDesc,
                                                sliceGroup, // alternatively, NULL can also be used to contract over all slices instead of specifying a sliceGroup object
                                                stream));

        // Synchronize and measure best timing
        auto time          = timer.seconds();
        minTimeCUTENSORNET = (time > minTimeCUTENSORNET) ? minTimeCUTENSORNET : time;
    }

    if (verbose) printf("Contracted the tensor network, each slice used the same prepared contraction\n");

    // Print the 1-norm of the output tensor (verification)
    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
    // restore the output tensor on Host
    HANDLE_CUDA_ERROR(cudaMemcpy(tensorData_h[numInputs], tensorData_d[numInputs], tensorSizes[numInputs], cudaMemcpyDeviceToHost));
    double norm1 = 0.0;
    for (int64_t i = 0; i < tensorElements[numInputs]; ++i)
    {
        norm1 += std::abs(tensorData_h[numInputs][i]);
    }
    if (verbose) printf("Computed the 1-norm of the output tensor: %e\n", norm1);

    /*************************/

    // Query the total Flop count for the tensor network contraction
    double flops{0.0};
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                 optimizerInfo,
                                                                 CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                                                                 &flops,
                                                                 sizeof(flops)));

    if (verbose)
    {
        printf("Number of tensor network slices = %ld\n", numSlices);
        printf("Tensor network contraction time (ms) = %.3f\n", minTimeCUTENSORNET * 1000.f);
    }

    // Sphinx: #9
    /****************
     * Free resources
     ****************/

    // Free cuTensorNet resources
    HANDLE_ERROR(cutensornetDestroySliceGroup(sliceGroup));
    HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig));
    HANDLE_ERROR(cutensornetDestroyNetwork(networkDesc));
    HANDLE_ERROR(cutensornetDestroy(handle));

    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

    // Free Host and GPU memory resources
    for (int32_t t = 0; t < numInputs + 1; ++t)
    {
        if (tensorData_h[t]) free(tensorData_h[t]);
        if (tensorData_d[t]) cudaFree(tensorData_d[t]);
    }
    if (work) cudaFree(work);

    if (verbose) printf("Freed resources and exited\n");

    return 0;
}
