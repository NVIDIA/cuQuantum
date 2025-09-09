/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
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
    /**********************
     * Computing: O_{a,m} = A_{a,b,c,d} B_{b,c,d,e} C_{e,f,g,h} D_{g,h,i,j} E_{i,j,k,l} F_{k,l,m}
     * We will execute the contraction a few times assuming all input tensors being constant except F.
     **********************/

    constexpr int32_t numInputs = 6;

    // Create vectors of tensor modes
    std::vector<std::vector<int32_t>> tensorModes{ // for input tensors & output tensor
        // input tensors
        {'a', 'b', 'c', 'd'}, // tensor A
        {'b', 'c', 'd', 'e'}, // tensor B
        {'e', 'f', 'g', 'h'}, // tensor C
        {'g', 'h', 'i', 'j'}, // tensor D
        {'i', 'j', 'k', 'l'}, // tensor E
        {'k', 'l', 'm'},      // tensor F
        // output tensor
        {'a', 'm'}}; // tensor O

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
    std::vector<std::vector<int64_t>> tensorExtents;
    tensorExtents.resize(numInputs + 1); // hold inputs + output tensors
    for (int32_t t = 0; t < numInputs + 1; ++t)
    {
        for (auto& mode : tensorModes[t]) tensorExtents[t].push_back(extent[mode]);
    }

    if (verbose) printf("Defined tensor network, modes, and extents\n");

    // Sphinx: #3
    /**********************
     * Allocating data
     **********************/

    std::vector<size_t> tensorElements(numInputs + 1);              // hold inputs + output tensors
    std::vector<size_t> tensorSizes(numInputs + 1);                 // hold inputs + output tensors
    std::vector<std::vector<int64_t>> tensorStrides(numInputs + 1); // hold inputs + output tensors
    size_t totalSize = 0;
    for (int32_t t = 0; t < numInputs + 1; ++t)
    {
        size_t numElements = 1;
        for (auto& mode : tensorModes[t])
        {
            tensorStrides[t].push_back(numElements);
            numElements *= extent[mode];
        }
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

    /*******************
     * Initialize data
     *******************/

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
     * Set constant input tensors
     *******************************/

    // specify which input tensors are constant
    std::vector<cutensornetTensorQualifiers_t> qualifiersIn(numInputs, cutensornetTensorQualifiers_t{0, 0, 0});
    for (int i = 0; i < numInputs; ++i)
    {
        qualifiersIn[i].isConstant = i < (numInputs - 1) ? 1 : 0;
    }

    /*******************************
     * Create Network
     *******************************/

    int32_t* modesIn[numInputs];
    int32_t numModesIn[numInputs];
    int64_t* extentsIn[numInputs];
    int64_t* stridesIn[numInputs];
    for (int t = 0; t < numInputs; ++t)
    {
        modesIn[t]    = tensorModes[t].data();
        numModesIn[t] = tensorModes[t].size();
        extentsIn[t]  = tensorExtents[t].data();
        stridesIn[t]  = tensorStrides[t].data();
    }

    // Set up tensor network
    cutensornetNetworkDescriptor_t networkDesc;
    HANDLE_ERROR(cutensornetCreateNetworkDescriptor(handle, numInputs, numModesIn, extentsIn, stridesIn, modesIn,
                                                    qualifiersIn.data(),
                                                    tensorModes[numInputs].size(), tensorExtents[numInputs].data(),
                                                    tensorStrides[numInputs].data(), tensorModes[numInputs].data(),
                                                    typeData, typeCompute, &networkDesc));

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
    /*******************************
     * Initialize the pairwise contraction plan (for cuTENSOR).
     *******************************/

    cutensornetContractionPlan_t plan;
    HANDLE_ERROR(cutensornetCreateContractionPlan(handle, networkDesc, optimizerInfo, workDesc, &plan));

    /*******************************
     * Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
     *           for each pairwise tensor contraction.
     *******************************/
    cutensornetContractionAutotunePreference_t autotunePref;
    HANDLE_ERROR(cutensornetCreateContractionAutotunePreference(handle, &autotunePref));

    const int numAutotuningIterations = 5; // may be 0
    HANDLE_ERROR(cutensornetContractionAutotunePreferenceSetAttribute(handle,
                                                                      autotunePref,
                                                                      CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
                                                                      &numAutotuningIterations,
                                                                      sizeof(numAutotuningIterations)));

    // Modify the plan again to find the best pair-wise contractions
    HANDLE_ERROR(cutensornetContractionAutotune(handle,
                                                plan,
                                                tensorData_d,
                                                tensorData_d[numInputs],
                                                workDesc,
                                                autotunePref,
                                                stream));

    HANDLE_ERROR(cutensornetDestroyContractionAutotunePreference(autotunePref));

    if (verbose) printf("Created a contraction plan for cuTensorNet and optionally auto-tuned it\n");

    // Sphinx: #8
    /**********************
     * Execute the tensor network contraction
     **********************/

    // Create a cutensornetSliceGroup_t object from a range of slice IDs
    cutensornetSliceGroup_t sliceGroup{};
    HANDLE_ERROR(cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup));

    GPUTimer timer{stream};
    double minTimeCUTENSORNET   = 1e100;
    double firstTimeCUTENSORNET = 1e100;
    const int numRuns           = 3; // number of repeats to get stable performance results
    for (int i = 0; i < numRuns; ++i)
    {
        // restore the output tensor on GPU
        HANDLE_CUDA_ERROR(cudaMemcpy(tensorData_d[numInputs], tensorData_h[numInputs], tensorSizes[numInputs], cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

        /*
         * Contract all slices of the tensor network
         */
        timer.start();

        int32_t accumulateOutput = 0; // output tensor data will be overwritten
        HANDLE_ERROR(cutensornetContractSlices(handle,
                                               plan,
                                               tensorData_d,
                                               tensorData_d[numInputs],
                                               accumulateOutput,
                                               workDesc,
                                               sliceGroup, // alternatively, NULL can also be used to contract over
                                                           // all slices instead of specifying a sliceGroup object
                                               stream));

        // Synchronize and measure best timing
        auto time = timer.seconds();
        if (i == 0) firstTimeCUTENSORNET = time;
        minTimeCUTENSORNET = (time > minTimeCUTENSORNET) ? minTimeCUTENSORNET : time;
    }

    if (verbose) printf("Contracted the tensor network, each slice used the same contraction plan\n");

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
        printf("Network contraction flop cost = %e\n", flops);
        printf("Tensor network contraction time (ms):\n");
        printf("\tfirst run (intermediate tensors get cached) = %.3f\n", firstTimeCUTENSORNET * 1000.f);
        printf("\tsubsequent run (cache reused) = %.3f\n", minTimeCUTENSORNET * 1000.f);
    }

    // Sphinx: #9
    /***************
     * Free resources
     ****************/

    // Free cuTensorNet resources
    HANDLE_ERROR(cutensornetDestroySliceGroup(sliceGroup));
    HANDLE_ERROR(cutensornetDestroyContractionPlan(plan));
    HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
    HANDLE_ERROR(cutensornetDestroyNetworkDescriptor(networkDesc));
    HANDLE_ERROR(cutensornetDestroy(handle));

    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

    // Free Host and GPU memory resources
    for (int32_t t = 0; t < numInputs + 1; ++t)
    {
        if (tensorData_h[t]) free(tensorData_h[t]);
        if (tensorData_d[t]) cudaFree(tensorData_d[t]);
    }
    // Free GPU memory resources
    if (workScratch) cudaFree(workScratch);
    if (workCache) cudaFree(workCache);
    if (verbose) printf("Freed resources and exited\n");

    return 0;
}
