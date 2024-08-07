/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
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


#define HANDLE_ERROR(x)                                           \
{ const auto err = x;                                             \
  if( err != CUTENSORNET_STATUS_SUCCESS )                         \
  { printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
  }                                                               \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
  }                                                               \
};


struct GPUTimer
{
    GPUTimer(cudaStream_t stream): stream_(stream)
    {
        HANDLE_CUDA_ERROR(cudaEventCreate(&start_));
        HANDLE_CUDA_ERROR(cudaEventCreate(&stop_));
    }

    ~GPUTimer()
    {
        HANDLE_CUDA_ERROR(cudaEventDestroy(start_));
        HANDLE_CUDA_ERROR(cudaEventDestroy(stop_));
    }

    void start()
    {
        HANDLE_CUDA_ERROR(cudaEventRecord(start_, stream_));
    }

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
    if(verbose)
        printf("cuTensorNet version: %ld\n", cuTensornetVersion);

    // Set GPU device
    int numDevices {0};
    HANDLE_CUDA_ERROR( cudaGetDeviceCount(&numDevices) );
    const int deviceId = 0;
    HANDLE_CUDA_ERROR( cudaSetDevice(deviceId) );
    cudaDeviceProp prop;
    HANDLE_CUDA_ERROR( cudaGetDeviceProperties(&prop, deviceId) );

    if(verbose) {
        printf("===== device info ======\n");
        printf("GPU-local-id:%d\n", deviceId);
        printf("GPU-name:%s\n", prop.name);
        printf("GPU-clock:%d\n", prop.clockRate);
        printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
        printf("GPU-nSM:%d\n", prop.multiProcessorCount);
        printf("GPU-major:%d\n", prop.major);
        printf("GPU-minor:%d\n", prop.minor);
        printf("========================\n");
    }

    typedef float floatType;
    cudaDataType_t typeData = CUDA_R_32F;
    cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F;

    if(verbose)
        printf("Included headers and defined data types\n");

    // Sphinx: #2
    /**********************
    * Computing: O_{a,m} = A_{a,b,c,d} B_{b,c,d,e} C_{e,f,g,h} D_{g,h,i,j} E_{i,j,k,l} F_{k,l,m}
    * We will execute the contraction a few times assuming all input tensors being constant except F.
    **********************/

    constexpr int32_t numInputs = 6;

    // Create vectors of tensor modes
    std::vector<std::vector<int32_t>> modesVec {
        {'a','b','c','d'},
        {'b','c','d','e'},
        {'e','f','g','h'},
        {'g','h','i','j'},
        {'i','j','k','l'},
        {'k','l','m'},
        {'a','m'}
    };

    // Set mode extents
    int64_t sameExtent = 36; // setting same extent for simplicity. In principle extents can differ.
    std::unordered_map<int32_t, int64_t> extent;
    for (auto &vec: modesVec)
    {
        for (auto &mode: vec)
        {
            extent[mode] = sameExtent;
        }
    }

    // Create a vector of extents for each tensor
    std::vector<std::vector<int64_t>> extentVec;
    extentVec.resize(numInputs+1); // hold inputs + output tensors
    for (int i = 0; i < numInputs+1; ++i)
    {
        for (auto mode : modesVec[i])
            extentVec[i].push_back(extent[mode]);
    }

    if(verbose)
        printf("Defined tensor network, modes, and extents\n");

    // Sphinx: #3
    /**********************
    * Allocating data
    **********************/

    std::vector<size_t> elementsVec;
    elementsVec.resize(numInputs+1); // hold inputs + output tensors
    for (int i = 0; i < numInputs+1; ++i)
    {
        elementsVec[i] = 1;
        for (auto mode : modesVec[i])
            elementsVec[i] *= extent[mode];
    }

    size_t totalSize = 0;
    std::vector<size_t> sizeVec;
    sizeVec.resize(numInputs+1); // hold inputs + output tensors
    for (int i = 0; i < numInputs+1; ++i)
    {
        sizeVec[i] = sizeof(floatType) * elementsVec[i];
        totalSize += sizeVec[i];
    }
    if(verbose)
        printf("Total GPU memory used for tensor storage: %.2f GiB\n",
                (totalSize) / 1024. /1024. / 1024);

    void* rawDataIn_d[numInputs];
    void* O_d;
    for (int i = 0; i < numInputs; ++i)
    {
        HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[i], sizeVec[i]) );
    }
    HANDLE_CUDA_ERROR( cudaMalloc((void**) &O_d, sizeVec[numInputs]));

    floatType* rawDataIn_h[numInputs];
    for (int i = 0; i < numInputs; ++i)
    {
        rawDataIn_h[i] = (floatType*) malloc(sizeof(floatType) * elementsVec[i]);
        if (rawDataIn_h[i] == NULL)
        {
           printf("Error: Host memory allocation failed!\n");
           return -1;
        }
    }
    floatType *O_h = (floatType*) malloc(sizeof(floatType) * elementsVec[numInputs]);
    if (O_h == NULL)
    {
        printf("Error: Host memory allocation failed!\n");
        return -1;
    }

    /*******************
    * Initialize data
    *******************/

    memset(O_h, 0, sizeof(floatType) * elementsVec[numInputs]);
    for (int i = 0; i < numInputs; ++i)
    {
        for (size_t e = 0; e < elementsVec[i]; ++e)
            rawDataIn_h[i][e] = ((floatType) rand()) / RAND_MAX;
    }

    for (int i = 0; i < numInputs; ++i)
    {
        HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[i], rawDataIn_h[i], sizeVec[i], cudaMemcpyHostToDevice) );
    }

    if(verbose)
        printf("Allocated GPU memory for data, initialize data, and create library handle\n");

    /*************************
    * cuTensorNet
    *************************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR( cudaStreamCreate(&stream) );

    cutensornetHandle_t handle;
    HANDLE_ERROR( cutensornetCreate(&handle) );

    // Sphinx: #4
    /*******************************
    * Set constant input tensors
    *******************************/

    // specify which input tensors are constant
    std::vector<cutensornetTensorQualifiers_t> qualifiersIn;
    qualifiersIn.resize(numInputs);
    for (int i = 0; i < numInputs; ++i)
    {
        if (i < 5)
            qualifiersIn[i].isConstant = 1;
        else
            qualifiersIn[i].isConstant = 0;
    }

    /*******************************
    * Create Network Descriptor
    *******************************/

    int32_t* modesIn[numInputs];
    int32_t numModesIn[numInputs];
    int64_t* extentsIn[numInputs];
    int64_t* stridesIn[numInputs];
    
    for (int i = 0; i < numInputs; ++i)
    {
        modesIn[i] = modesVec[i].data();
        numModesIn[i] = modesVec[i].size();
        extentsIn[i] = extentVec[i].data();
        stridesIn[i] = NULL; // strides are optional; if no stride is provided, cuTensorNet assumes a generalized column-major data layout
    }

    // Set up tensor network
    cutensornetNetworkDescriptor_t descNet;
    HANDLE_ERROR( cutensornetCreateNetworkDescriptor(handle,
                        numInputs, numModesIn, extentsIn, stridesIn, modesIn, qualifiersIn.data(),
                        modesVec[numInputs].size(), extentVec[numInputs].data(), /*stridesOut = */NULL, modesVec[numInputs].data(),
                        typeData, typeCompute,
                        &descNet) );

    if(verbose)
        printf("Initialized the cuTensorNet library and created a tensor network descriptor\n");

    // Sphinx: #5
    /*******************************
    * Choose workspace limit based on available resources.
    *******************************/

    size_t freeMem, totalMem;
    HANDLE_CUDA_ERROR( cudaMemGetInfo(&freeMem, &totalMem) );
    uint64_t workspaceLimit = (uint64_t)((double)freeMem * 0.9);
    if(verbose)
        printf("Workspace limit = %lu\n", workspaceLimit);

    /*******************************
    * Set contraction order
    *******************************/

    // Create contraction optimizer info
    cutensornetContractionOptimizerInfo_t optimizerInfo;
    HANDLE_ERROR( cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo) );

    // set a predetermined contraction path
    std::vector<int32_t> path{0,1,0,4,0,3,0,2,0,1};
    const auto numContractions = numInputs - 1;
    cutensornetContractionPath_t contPath;
    contPath.data = reinterpret_cast<cutensornetNodePair_t*>(const_cast<int32_t*>(path.data()));
    contPath.numContractions = numContractions;

    // provide user-specified contPath
    HANDLE_ERROR( cutensornetContractionOptimizerInfoSetAttribute(
                    handle,
                    optimizerInfo,
                    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH,
                    &contPath,
                    sizeof(contPath)));
    int64_t numSlices = 1;

    if(verbose)
        printf("Set predetermined contraction path into cuTensorNet optimizer\n");

    // Sphinx: #6
    /*******************************
    * Create workspace descriptor, allocate workspace, and set it.
    *******************************/

    cutensornetWorkspaceDescriptor_t workDesc;
    HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle, &workDesc) );

    // set SCRATCH workspace, which will be used during each network contraction operation, not needed afterwords
    int64_t requiredWorkspaceSizeScratch = 0;
    HANDLE_ERROR( cutensornetWorkspaceComputeContractionSizes(handle,
                                                            descNet,
                                                            optimizerInfo,
                                                            workDesc) );

    HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle,
                                                    workDesc,
                                                    CUTENSORNET_WORKSIZE_PREF_MIN,
                                                    CUTENSORNET_MEMSPACE_DEVICE,
                                                    CUTENSORNET_WORKSPACE_SCRATCH,
                                                    &requiredWorkspaceSizeScratch) );

    void* workScratch = nullptr;
    HANDLE_CUDA_ERROR( cudaMalloc(&workScratch, requiredWorkspaceSizeScratch) );

    HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle,
                                                workDesc,
                                                CUTENSORNET_MEMSPACE_DEVICE,
                                                CUTENSORNET_WORKSPACE_SCRATCH,
                                                workScratch,
                                                requiredWorkspaceSizeScratch) );

    // set CACHE workspace, which will be used across network contraction operations
    int64_t requiredWorkspaceSizeCache = 0;
    HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle,
                                                    workDesc,
                                                    CUTENSORNET_WORKSIZE_PREF_MIN,
                                                    CUTENSORNET_MEMSPACE_DEVICE,
                                                    CUTENSORNET_WORKSPACE_CACHE,
                                                    &requiredWorkspaceSizeCache) );

    void* workCache = nullptr;
    HANDLE_CUDA_ERROR( cudaMalloc(&workCache, requiredWorkspaceSizeCache) );

    HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle,
                                                workDesc,
                                                CUTENSORNET_MEMSPACE_DEVICE,
                                                CUTENSORNET_WORKSPACE_CACHE,
                                                workCache,
                                                requiredWorkspaceSizeCache) );

    if(verbose)
        printf("Allocated and set up the GPU workspace\n");

    // Sphinx: #7
    /*******************************
    * Initialize the pairwise contraction plan (for cuTENSOR).
    *******************************/

    cutensornetContractionPlan_t plan;
    HANDLE_ERROR( cutensornetCreateContractionPlan(handle,
                                                    descNet,
                                                    optimizerInfo,
                                                    workDesc,
                                                    &plan) );

    /*******************************
    * Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
    *           for each pairwise tensor contraction.
    *******************************/
    cutensornetContractionAutotunePreference_t autotunePref;
    HANDLE_ERROR( cutensornetCreateContractionAutotunePreference(handle,
                                                        &autotunePref) );

    const int numAutotuningIterations = 5; // may be 0
    HANDLE_ERROR( cutensornetContractionAutotunePreferenceSetAttribute(
                            handle,
                            autotunePref,
                            CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
                            &numAutotuningIterations,
                            sizeof(numAutotuningIterations)) );

    // Modify the plan again to find the best pair-wise contractions
    HANDLE_ERROR( cutensornetContractionAutotune(handle,
                                                    plan,
                                                    rawDataIn_d,
                                                    O_d,
                                                    workDesc,
                                                    autotunePref,
                                                    stream) );

    HANDLE_ERROR( cutensornetDestroyContractionAutotunePreference(autotunePref) );

    if(verbose)
        printf("Created a contraction plan for cuTensorNet and optionally auto-tuned it\n");

    // Sphinx: #8
    /**********************
    * Execute the tensor network contraction
    **********************/

    // Create a cutensornetSliceGroup_t object from a range of slice IDs
    cutensornetSliceGroup_t sliceGroup{};
    HANDLE_ERROR( cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup) );

    GPUTimer timer {stream};
    double minTimeCUTENSORNET = 1e100;
    double firstTimeCUTENSORNET = 1e100;
    const int numRuns = 3; // number of repeats to get stable performance results
    for (int i = 0; i < numRuns; ++i)
    {
        HANDLE_CUDA_ERROR( cudaMemcpy(O_d, O_h, sizeVec[numInputs], cudaMemcpyHostToDevice) ); // restore the output tensor on GPU
        HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );

        /*
        * Contract all slices of the tensor network
        */
        timer.start();

        int32_t accumulateOutput = 0; // output tensor data will be overwritten
        HANDLE_ERROR( cutensornetContractSlices(handle,
                        plan,
                        rawDataIn_d,
                        O_d,
                        accumulateOutput,
                        workDesc,
                        sliceGroup, // slternatively, NULL can also be used to contract over all slices instead of specifying a sliceGroup object
                        stream) );

        // Synchronize and measure best timing
        auto time = timer.seconds();
        if (i == 0) 
            firstTimeCUTENSORNET = time;
        minTimeCUTENSORNET = (time > minTimeCUTENSORNET) ? minTimeCUTENSORNET : time;
    }

    if(verbose)
        printf("Contracted the tensor network, each slice used the same contraction plan\n");

    // Print the 1-norm of the output tensor (verification)
    HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );
    HANDLE_CUDA_ERROR( cudaMemcpy(O_h, O_d, sizeVec[numInputs], cudaMemcpyDeviceToHost) ); // restore the output tensor on Host
    double norm1 = 0.0;
    for (int64_t i = 0; i < elementsVec[numInputs]; ++i) {
        norm1 += std::abs(O_h[i]);
    }
    if(verbose)
        printf("Computed the 1-norm of the output tensor: %e\n", norm1);

    /*************************/

    // Query the total Flop count for the tensor network contraction
    double flops {0.0};
    HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
                        handle,
                        optimizerInfo,
                        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                        &flops,
                        sizeof(flops)) );

    if(verbose) {
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
    HANDLE_ERROR( cutensornetDestroySliceGroup(sliceGroup) );
    HANDLE_ERROR( cutensornetDestroyContractionPlan(plan) );
    HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );
    HANDLE_ERROR( cutensornetDestroyContractionOptimizerInfo(optimizerInfo) );
    HANDLE_ERROR( cutensornetDestroyNetworkDescriptor(descNet) );
    HANDLE_ERROR( cutensornetDestroy(handle) );

    // Free Host memory resources
    if (O_h) free(O_h);
    for (int i = 0; i < numInputs; ++i)
    {
        if (rawDataIn_h[i]) 
            free(rawDataIn_h[i]);
    }
    // Free GPU memory resources
    if (workScratch) cudaFree(workScratch);
    if (workCache) cudaFree(workCache);
    if (O_d) cudaFree(O_d);
    for (int i = 0; i < numInputs; ++i)
    {
        if (rawDataIn_d[i]) 
            cudaFree(rawDataIn_d[i]);
    }
    if(verbose)
        printf("Freed resources and exited\n");

    return 0;
}
