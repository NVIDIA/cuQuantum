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

// Sphinx: MPI #1 [begin]

#include <mpi.h>

// Sphinx: MPI #1 [end]

#define HANDLE_ERROR(x)                                                                 \
    do {                                                                                \
        const auto err = x;                                                             \
        if (err != CUTENSORNET_STATUS_SUCCESS)                                          \
        {                                                                               \
            printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
            fflush(stdout);                                                             \
            MPI_Abort(MPI_COMM_WORLD, err);                                             \
        }                                                                               \
    } while (0)

#define HANDLE_CUDA_ERROR(x)                                                          \
    do {                                                                              \
        const auto err = x;                                                           \
        if (err != cudaSuccess)                                                       \
        {                                                                             \
            printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
            fflush(stdout);                                                           \
            MPI_Abort(MPI_COMM_WORLD, err);                                           \
        }                                                                             \
    } while (0)

// Sphinx: MPI #2 [begin]

#define HANDLE_MPI_ERROR(x)                                        \
    do {                                                           \
        const auto err = x;                                        \
        if (err != MPI_SUCCESS)                                    \
        {                                                          \
            char error[MPI_MAX_ERROR_STRING];                      \
            int len;                                               \
            MPI_Error_string(err, error, &len);                    \
            printf("MPI Error: %s in line %d\n", error, __LINE__); \
            fflush(stdout);                                        \
            MPI_Abort(MPI_COMM_WORLD, err);                        \
        }                                                          \
    } while (0)

// Sphinx: MPI #2 [end]

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

int main(int argc, char** argv)
{
    static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");

    // Sphinx: MPI #3 [begin]

    // Initialize MPI
    HANDLE_MPI_ERROR(MPI_Init(&argc, &argv));
    int rank{-1};
    HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int numProcs{0};
    HANDLE_MPI_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &numProcs));

    // Sphinx: MPI #3 [end]

    bool verbose = (rank == 0) ? true : false;
    if (verbose)
    {
        printf("*** Printing is done only from the root MPI process to prevent jumbled messages ***\n");
        printf("The number of MPI processes is %d\n", numProcs);
    }
    if (verbose) printf("Initialized MPI service\n");

    // Check cuTensorNet version
    const size_t cuTensornetVersion = cutensornetGetVersion();
    if (verbose) printf("cuTensorNet version: %ld\n", cuTensornetVersion);

    // Sphinx: MPI #4 [begin]

    // Set GPU device based on ranks and nodes
    int numDevices{0};
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
    const int deviceId = rank % numDevices; // we assume that the processes are mapped to nodes in contiguous chunks
    HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

    // Sphinx: MPI #4 [end]

    if (verbose)
    {
        printf("===== rank 0 device info ======\n");
        printf("GPU-local-id:%d\n", deviceId);
        printf("GPU-name:%s\n", prop.name);
        printf("GPU-clock:%d\n", DEV_ATTR(cudaDevAttrClockRate, deviceId));
        printf("GPU-memoryClock:%d\n", DEV_ATTR(cudaDevAttrMemoryClockRate, deviceId));
        printf("GPU-nSM:%d\n", prop.multiProcessorCount);
        printf("GPU-major:%d\n", prop.major);
        printf("GPU-minor:%d\n", prop.minor);
        printf("CUDA-available-devices:%d\n", numDevices);
        printf("CUDA_VISIBLE_DEVICES:%s\n", getenv("CUDA_VISIBLE_DEVICES") != nullptr ? getenv("CUDA_VISIBLE_DEVICES") : "");
        printf("===============================\n");
    }
    else
    {
        printf("===== rank %d device info ======\nGPU-local-id:%d\n", rank, deviceId);
    }

    typedef float floatType;
    MPI_Datatype floatTypeMPI            = MPI_FLOAT;
    cudaDataType_t typeData              = CUDA_R_32F;
    cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F;

    if (verbose) printf("Included headers and defined data types\n");

    // Sphinx: #2
    /**********************
     * Computing: R_{k,l} = A_{a,b,c,d,e,f} B_{b,g,h,e,i,j} C_{m,a,g,f,i,k} D_{l,c,h,d,j,m}
     **********************/

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

    // Sphinx: MPI #5 [begin]
    /*******************
     * Initialize data
     *******************/

    // init output tensor to all 0s
    memset(tensorData_h[numInputs], 0, tensorSizes[numInputs]);
    if (rank == 0)
    {
        // init input tensors to random values
        for (int32_t t = 0; t < numInputs; ++t)
        {
            for (uint64_t i = 0; i < tensorElements[t]; ++i) tensorData_h[t][i] = ((floatType)rand()) / RAND_MAX;
        }
    }

    // Broadcast input data to all ranks
    for (int32_t t = 0; t < numInputs; ++t)
    {
        HANDLE_MPI_ERROR(MPI_Bcast(tensorData_h[t], tensorElements[t], floatTypeMPI, 0, MPI_COMM_WORLD));
    }

    // copy input data to device buffers
    for (int32_t t = 0; t < numInputs; ++t)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy(tensorData_d[t], tensorData_h[t], tensorSizes[t], cudaMemcpyHostToDevice));
    }
    // Sphinx: MPI #5 [end]

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
    if (verbose) printf("Initialized the cuTensorNet library and created a tensor network\n");

    // Sphinx: #5
    /*******************************
     * Choose workspace limit based on available resources.
     *******************************/

    size_t freeMem, totalMem;
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    uint64_t workspaceLimit = (uint64_t)((double)freeMem * 0.9);
    // Make sure all MPI processes will assume the minimal workspace size among all
    HANDLE_MPI_ERROR(MPI_Allreduce(MPI_IN_PLACE, &workspaceLimit, 1, MPI_INT64_T, MPI_MIN, MPI_COMM_WORLD));
    if (verbose) printf("Workspace limit = %lu\n", workspaceLimit);

    /*******************************
     * Find "optimal" contraction order and slicing (in parallel)
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

    // Create contraction optimizer info
    cutensornetContractionOptimizerInfo_t optimizerInfo;
    HANDLE_ERROR(cutensornetCreateContractionOptimizerInfo(handle, networkDesc, &optimizerInfo));

    // Sphinx: MPI #6 [begin]

    // Compute the path on all ranks so that we can choose the path with the lowest cost. Note that since this is a tiny
    // example with 4 operands, all processes will compute the same globally optimal path. This is not the case for large
    // tensor networks. For large networks, hyper-optimization does become beneficial.

    // Enforce tensor network slicing (for parallelization)
    const int32_t min_slices = numProcs;
    HANDLE_ERROR(cutensornetContractionOptimizerConfigSetAttribute(handle,
                                                                   optimizerConfig,
                                                                   CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES,
                                                                   &min_slices,
                                                                   sizeof(min_slices)));

    // Find an optimized tensor network contraction path on each MPI process
    HANDLE_ERROR(cutensornetContractionOptimize(handle,
                                                networkDesc,
                                                optimizerConfig,
                                                workspaceLimit,
                                                optimizerInfo));

    // Query the obtained Flop count
    double flops{-1.};
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                 optimizerInfo,
                                                                 CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                                                                 &flops,
                                                                 sizeof(flops)));

    // Choose the contraction path with the lowest Flop cost
    struct
    {
        double value;
        int rank;
    } in{flops, rank}, out;

    HANDLE_MPI_ERROR(MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD));
    const int sender = out.rank;
    flops            = out.value;

    if (verbose) printf("Process %d has the path with the lowest FLOP count %lf\n", sender, flops);

    // Get the buffer size for optimizerInfo and broadcast it
    size_t bufSize{0};
    if (rank == sender)
    {
        HANDLE_ERROR(cutensornetContractionOptimizerInfoGetPackedSize(handle, optimizerInfo, &bufSize));
    }
    HANDLE_MPI_ERROR(MPI_Bcast(&bufSize, 1, MPI_INT64_T, sender, MPI_COMM_WORLD));

    // Allocate a buffer
    std::vector<char> buffer(bufSize);

    // Pack optimizerInfo on sender and broadcast it
    if (rank == sender)
    {
        HANDLE_ERROR(cutensornetContractionOptimizerInfoPackData(handle, optimizerInfo, buffer.data(), bufSize));
    }
    HANDLE_MPI_ERROR(MPI_Bcast(buffer.data(), bufSize, MPI_CHAR, sender, MPI_COMM_WORLD));

    // Unpack optimizerInfo from the buffer
    if (rank != sender)
    {
        HANDLE_ERROR(
            cutensornetUpdateContractionOptimizerInfoFromPackedData(handle, buffer.data(), bufSize, optimizerInfo));
    }

    // Update the network with the modified optimizer info
    HANDLE_ERROR(cutensornetNetworkSetOptimizerInfo(handle, networkDesc, optimizerInfo));

    // Query the number of slices the tensor network execution will be split into
    int64_t numSlices = 0;
    HANDLE_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                 optimizerInfo,
                                                                 CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                                                                 &numSlices,
                                                                 sizeof(numSlices)));
    assert(numSlices > 0);

    // Calculate each process's share of the slices
    int64_t procChunk  = numSlices / numProcs;
    int extra          = numSlices % numProcs;
    int procSliceBegin = rank * procChunk + std::min(rank, extra);
    int procSliceEnd   = (rank == numProcs - 1) ? numSlices : (rank + 1) * procChunk + std::min(rank + 1, extra);

    // Sphinx: MPI #6 [end]

    if (verbose) printf("Found an optimized contraction path using cuTensorNet optimizer\n");

    // Sphinx: #6
    /*******************************
     * Create workspace descriptor, allocate workspace, and set it.
     *******************************/

    cutensornetWorkspaceDescriptor_t workDesc;
    HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));

    int64_t requiredWorkspaceSize = 0;
    HANDLE_ERROR(cutensornetWorkspaceComputeContractionSizes(handle, networkDesc, optimizerInfo, workDesc));

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
    /*******************************
     * Prepare the contraction.
     *******************************/

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
    /*******************************
     * Optional: Auto-tune the contraction plan to pick the fastest kernel
     *           for each pairwise tensor contraction.
     *******************************/
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
    /**********************
     * Execute the tensor network contraction (in parallel)
     **********************/

    // Sphinx: MPI #7 [begin]

    // Create a cutensornetSliceGroup_t object from a range of slice IDs
    cutensornetSliceGroup_t sliceGroup{};
    HANDLE_ERROR(cutensornetCreateSliceGroupFromIDRange(handle, procSliceBegin, procSliceEnd, 1, &sliceGroup));

    // Sphinx: MPI #7 [end]

    GPUTimer timer{stream};
    double minTimeCUTENSORNET = 1e100;
    const int numRuns         = 3; // number of repeats to get stable performance results
    for (int i = 0; i < numRuns; ++i)
    {
        cudaDeviceSynchronize();

        /*
         * Contract over the range of slices this process is responsible for.
         */
        timer.start();

        // Don't accumulate into output since we use a one-process-per-gpu model
        int32_t accumulateOutput = 0;

        // Sphinx: MPI #8 [begin]

        HANDLE_ERROR(cutensornetNetworkContract(handle,
                                                networkDesc,
                                                accumulateOutput,
                                                workDesc,
                                                sliceGroup,
                                                stream));

        // Sphinx: MPI #8 [end]

        // Sphinx: MPI #9 [begin]

        // Perform Allreduce operation on the output tensor
        HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
        // restore the output tensor on Host
        HANDLE_CUDA_ERROR(cudaMemcpy(tensorData_h[numInputs], tensorData_d[numInputs], tensorSizes[numInputs], cudaMemcpyDeviceToHost));
        HANDLE_MPI_ERROR(MPI_Allreduce(MPI_IN_PLACE, tensorData_h[numInputs], tensorElements[numInputs], floatTypeMPI, MPI_SUM, MPI_COMM_WORLD));

        // Sphinx: MPI #9 [end]

        // Measure timing
        auto time       = timer.seconds();
        minTimeCUTENSORNET = (time > minTimeCUTENSORNET) ? minTimeCUTENSORNET : time;
    }

    if (verbose)
        printf("Contracted the tensor network, all slices within the same rank used the same contraction plan.\n");

    // Print the 1-norm of the output tensor (verification)
    double norm1 = 0.0;
    for (int64_t i = 0; i < tensorElements[numInputs]; ++i)
    {
        norm1 += std::abs(tensorData_h[numInputs][i]);
    }
    if (verbose) printf("Computed the 1-norm of the output tensor: %e\n", norm1);

    /*************************/

    // Query the total Flop count for the tensor network contraction
    flops = 0.0;
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
    /***************
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

    // Sphinx: MPI #10 [begin]

    // Shut down MPI service
    HANDLE_MPI_ERROR(MPI_Finalize());

    // Sphinx: MPI #10 [end]

    if (verbose) printf("Freed resources and exited\n");

    return 0;
}
