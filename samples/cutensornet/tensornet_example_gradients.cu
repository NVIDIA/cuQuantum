/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: #1

#include <stdlib.h>
#include <stdio.h>

#include <algorithm>
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
    /**********************
     * Computing: O_{a,m} = A_{a,b,c,d} B_{b,c,d,e} C_{e,g,h} D_{g,h,i,j} E_{i,j,k,l} F_{k,l,m}
     * We will execute the contraction and compute the gradients of input tensors A, B, C
     **********************/

    constexpr int32_t numInputs       = 6;
    std::vector<int32_t> gradInputIDs = {0, 1, 2};

    // Create vectors of tensor modes
    std::vector<std::vector<int32_t>> tensorModes{
        {'a', 'b', 'c', 'd'}, // tensor A
        {'b', 'c', 'd', 'e'}, // tensor B
        {'e', 'g', 'h'},      // tensor C
        {'g', 'h', 'i', 'j'}, // tensor D
        {'i', 'j', 'k', 'l'}, // tensor E
        {'k', 'l', 'm'},      // tensor F
        {'a', 'm'}            // tensor O
    };

    // Set mode extents
    int64_t sameExtent = 36; // setting same extent for simplicity. In principle extents can differ.
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
    /**********************
     * Allocating data
     **********************/

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
    void* adjoint_d; // hold data of the adjoint/activation tensor
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&adjoint_d, tensorSizes[numInputs]));

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
    floatType* adjoint_h = (floatType*)malloc(tensorSizes[numInputs]);
    if (adjoint_h == NULL)
    {
        printf("Error: Host memory allocation failed!\n");
        return -1;
    }

    void* gradients_d[numInputs] = {nullptr};
    for (auto i : gradInputIDs)
    {
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&gradients_d[i], tensorSizes[i]));
    }
    void* gradients_h[numInputs] = {nullptr};
    for (auto i : gradInputIDs)
    {
        gradients_h[i] = (floatType*)malloc(tensorSizes[i]);
        if (gradients_h[i] == NULL)
        {
            printf("Error: Host memory allocation failed!\n");
            return -1;
        }
    }

    /*******************
     * Initialize data
     *******************/

    // set output tensor data to all 0s
    memset(tensorData_h[numInputs], 0, tensorSizes[numInputs]);
    // init input tensors data to random values
    for (int32_t t = 0; t < numInputs; ++t)
    {
        for (size_t e = 0; e < tensorElements[t]; ++e) tensorData_h[t][e] = ((floatType)rand()) / RAND_MAX;
    }
    // set activation tensor to all 1s
    for (size_t e = 0; e < tensorElements[numInputs]; ++e) adjoint_h[e] = (floatType)1.0;

    // copy tensors' data to device buffers
    for (int32_t t = 0; t < numInputs; ++t)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy(tensorData_d[t], tensorData_h[t], tensorSizes[t], cudaMemcpyHostToDevice));
    }
    HANDLE_CUDA_ERROR(cudaMemcpy(adjoint_d, adjoint_h, tensorSizes[numInputs], cudaMemcpyHostToDevice));

    /*************************
     * cuTensorNet
     *************************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    cutensornetHandle_t handle;
    HANDLE_ERROR(cutensornetCreate(&handle));

    if (verbose) printf("Allocated GPU memory for data, initialized data, and created library handle\n");

    // Sphinx: #4
    /*******************************
     * Create Network
     *******************************/

    // Set up tensor network
    cutensornetNetworkDescriptor_t networkDesc;
    HANDLE_ERROR(cutensornetCreateNetwork(handle, &networkDesc));

    int64_t tensorIDs[numInputs]; // for input tensors
    // attach the input tensors to the network
    for (int32_t t = 0; t < numInputs; ++t)
    {
        cutensornetTensorQualifiers_t qualifiers{0, 0, 0};
        qualifiers.requiresGradient = gradInputIDs.end() != std::find(gradInputIDs.begin(), gradInputIDs.end(), t);
        HANDLE_ERROR(cutensornetNetworkAppendTensor(handle,
                                                    networkDesc,
                                                    tensorModes[t].size(),
                                                    tensorExtents[t].data(),
                                                    tensorModes[t].data(),
                                                    &qualifiers,
                                                    typeData,
                                                    &tensorIDs[t]));
    }

    // set output tensor of the network
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

    if (verbose) printf("Initialized the cuTensorNet library and created a tensor network\n");

    // Sphinx: #5
    /*******************************
     * Choose workspace limit based on available resources.
     *******************************/

    size_t freeMem, totalMem;
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    uint64_t workspaceLimit = (uint64_t)((double)freeMem * 0.9);
    if (verbose) printf("Workspace limit = %lu\n", workspaceLimit);

    /*******************************
     * Set contraction order
     *******************************/

    // Create contraction optimizer info
    cutensornetContractionOptimizerInfo_t optimizerInfo;
    HANDLE_ERROR(cutensornetCreateContractionOptimizerInfo(handle, networkDesc, &optimizerInfo));

    // set a predetermined contraction path
    std::vector<int32_t> path{0, 1, 0, 4, 0, 3, 0, 2, 0, 1};
    const auto numContractions = numInputs - 1;
    cutensornetContractionPath_t contPath;
    contPath.data            = reinterpret_cast<cutensornetNodePair_t*>(const_cast<int32_t*>(path.data()));
    contPath.numContractions = numContractions;

    // provide user-specified contPath
    HANDLE_ERROR(cutensornetContractionOptimizerInfoSetAttribute(handle,
                                                                 optimizerInfo,
                                                                 CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH,
                                                                 &contPath,
                                                                 sizeof(contPath)));

    // Attach the optimizer info to the network 
    HANDLE_ERROR(cutensornetNetworkSetOptimizerInfo(handle,
                                                    networkDesc,
                                                    optimizerInfo));
    int64_t numSlices = 1;

    if (verbose) printf("Set predetermined contraction path into cuTensorNet optimizer\n");

    // Sphinx: #6
    /*******************************
     * Create workspace descriptor, allocate workspace, and set it.
     *******************************/

    cutensornetWorkspaceDescriptor_t workDesc;
    HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));

    // set SCRATCH workspace, which will be used during each network contraction operation, not needed afterwords
    int64_t requiredWorkspaceSizeScratch = 0;
    HANDLE_ERROR(cutensornetWorkspaceComputeContractionSizes(handle, networkDesc, optimizerInfo, workDesc));

    HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(handle,
                                                   workDesc,
                                                   CUTENSORNET_WORKSIZE_PREF_MIN,
                                                   CUTENSORNET_MEMSPACE_DEVICE,
                                                   CUTENSORNET_WORKSPACE_SCRATCH,
                                                   &requiredWorkspaceSizeScratch));

    void* workScratch = nullptr;
    HANDLE_CUDA_ERROR(cudaMalloc(&workScratch, requiredWorkspaceSizeScratch));

    HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle,
                                               workDesc,
                                               CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH,
                                               workScratch,
                                               requiredWorkspaceSizeScratch));

    // set CACHE workspace, which will be used across network contraction operations
    int64_t requiredWorkspaceSizeCache = 0;
    HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(handle,
                                                   workDesc,
                                                   CUTENSORNET_WORKSIZE_PREF_MIN,
                                                   CUTENSORNET_MEMSPACE_DEVICE,
                                                   CUTENSORNET_WORKSPACE_CACHE,
                                                   &requiredWorkspaceSizeCache));

    void* workCache = nullptr;
    HANDLE_CUDA_ERROR(cudaMalloc(&workCache, requiredWorkspaceSizeCache));

    HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle,
                                               workDesc,
                                               CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_CACHE,
                                               workCache,
                                               requiredWorkspaceSizeCache));

    if (verbose) printf("Allocated and set up the GPU workspace\n");

    // Sphinx: #7
    /**********************************
     * Prepare the network contraction.
     **********************************/

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

    HANDLE_ERROR(cutensornetNetworkPrepareContraction(handle,
                                                      networkDesc,
                                                      workDesc));

    /*******************************
     * Optional: Auto-tune the network's contraction to pick the fastest kernel
     *           for each pairwise tensor contraction.
     *******************************/
    cutensornetNetworkAutotunePreference_t autotunePref;
    HANDLE_ERROR(cutensornetCreateNetworkAutotunePreference(handle, &autotunePref));

    const int numAutotuningIterations = 5; // may be 0
    HANDLE_ERROR(cutensornetNetworkAutotunePreferenceSetAttribute(handle,
                                                                  autotunePref,
                                                                  CUTENSORNET_NETWORK_AUTOTUNE_MAX_ITERATIONS,
                                                                  &numAutotuningIterations,
                                                                  sizeof(numAutotuningIterations)));

    // Autotune the network to find the best pair-wise contractions
    HANDLE_ERROR(cutensornetNetworkAutotuneContraction(handle,
                                                       networkDesc,
                                                       workDesc,
                                                       autotunePref,
                                                       stream));

    HANDLE_ERROR(cutensornetDestroyNetworkAutotunePreference(autotunePref));

    if (verbose) printf("Prepared the network contraction for cuTensorNet and optionally auto-tuned it\n");

    // Sphinx: #8
    /**********************
     * Execute the tensor network contraction
     **********************/

    // Create a cutensornetSliceGroup_t object from a range of slice IDs
    cutensornetSliceGroup_t sliceGroup{};
    HANDLE_ERROR(cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup));

    GPUTimer timer{stream};
    // restore the output tensor on GPU
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

    // Synchronize and measure timing
    auto time = timer.seconds();

    /**********************
     * Prepare the tensor network gradient computation
     **********************/

    HANDLE_ERROR(cutensornetNetworkSetAdjointTensorMemory(handle, networkDesc, adjoint_d, NULL));

    for (auto gid : gradInputIDs) // for only those tensors that require the gradient
    {
        HANDLE_ERROR(cutensornetNetworkSetGradientTensorMemory(handle,
                                                               networkDesc,
                                                               gid,
                                                               gradients_d[gid], NULL));
    }

    HANDLE_ERROR(cutensornetNetworkPrepareGradientsBackward(handle, networkDesc, workDesc));

    /**********************
     * Execute the tensor network gradient computation
     **********************/
    timer.start();

    HANDLE_ERROR(cutensornetNetworkComputeGradientsBackward(handle,
                                                            networkDesc,
                                                            accumulateOutput,
                                                            workDesc,
                                                            sliceGroup, // alternatively, NULL can also be used to contract over all slices instead of specifying a sliceGroup object
                                                            stream));
    // Synchronize and measure timing
    time += timer.seconds();

    if (verbose) printf("Contracted the tensor network and computed gradients\n");

    // restore the output tensor on Host
    HANDLE_CUDA_ERROR(cudaMemcpy(tensorData_h[numInputs], tensorData_d[numInputs], tensorSizes[numInputs], cudaMemcpyDeviceToHost));

    for (auto i : gradInputIDs)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy(gradients_h[i], gradients_d[i], tensorSizes[i], cudaMemcpyDeviceToHost));
    }

    /*************************/

    if (verbose)
    {
        printf("Tensor network contraction and back-propagation time (ms): = %.3f\n", time * 1000.f);
    }

    // Sphinx: #9
    /***************
     * Free resources
     ****************/

    // Free cuTensorNet resources
    HANDLE_ERROR(cutensornetDestroySliceGroup(sliceGroup));
    HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
    HANDLE_ERROR(cutensornetDestroyNetwork(networkDesc));
    HANDLE_ERROR(cutensornetDestroy(handle));

    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

    // Free Host memory resources
    for (int i = 0; i < numInputs; ++i)
    {
        if (tensorData_h[i]) free(tensorData_h[i]);
        if (gradients_h[i]) free(gradients_h[i]);
    }
    if (tensorData_h[numInputs]) free(tensorData_h[numInputs]);
    if (adjoint_h) free(adjoint_h);

    // Free GPU memory resources
    if (workScratch) cudaFree(workScratch);
    if (workCache) cudaFree(workCache);
    if (adjoint_d) cudaFree(adjoint_d);
    for (int i = 0; i < numInputs; ++i)
    {
        if (tensorData_d[i]) cudaFree(tensorData_d[i]);
        if (gradients_d[i]) cudaFree(gradients_d[i]);
    }
    if (tensorData_d[numInputs]) cudaFree(tensorData_d[numInputs]);
    if (verbose) printf("Freed resources and exited\n");

    return 0;
}
