/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.
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

#define HANDLE_ERROR(x)                                           \
{ const auto err = x;                                             \
  if( err != CUTENSORNET_STATUS_SUCCESS )                         \
  { printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
    MPI_Abort(MPI_COMM_WORLD, err);                               \
  }                                                               \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
    MPI_Abort(MPI_COMM_WORLD, err) ;                              \
  }                                                               \
};

// Sphinx: MPI #2 [begin]

#define HANDLE_MPI_ERROR(x)                                       \
{ const auto err = x;                                             \
  if( err != MPI_SUCCESS )                                        \
  { char error[MPI_MAX_ERROR_STRING]; int len;                    \
    MPI_Error_string(err, error, &len);                           \
    printf("MPI Error: %s in line %d\n", error, __LINE__);        \
    fflush(stdout);                                               \
    MPI_Abort(MPI_COMM_WORLD, err);                               \
  }                                                               \
};

// Sphinx: MPI #2 [end]

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


int main(int argc, char **argv)
{
   static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");

   // Sphinx: MPI #3 [begin]

   // Initialize MPI
   HANDLE_MPI_ERROR( MPI_Init(&argc, &argv) );
   int rank {-1};
   HANDLE_MPI_ERROR( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
   int numProcs {0};
   HANDLE_MPI_ERROR( MPI_Comm_size(MPI_COMM_WORLD, &numProcs) );

   // Sphinx: MPI #3 [end]

   bool verbose = (rank == 0) ? true : false;
   if (verbose)
   {
      printf("*** Printing is done only from the root MPI process to prevent jumbled messages ***\n");
      printf("The number of MPI processes is %d\n", numProcs);
   }
   if(verbose)
      printf("Initialized MPI service\n");

   // Check cuTensorNet version
   const size_t cuTensornetVersion = cutensornetGetVersion();
   if(verbose)
      printf("cuTensorNet version: %ld\n", cuTensornetVersion);

   // Sphinx: MPI #4 [begin]

   // Set GPU device based on ranks and nodes
   int numDevices {0};
   HANDLE_CUDA_ERROR( cudaGetDeviceCount(&numDevices) );
   const int deviceId = rank % numDevices; // we assume that the processes are mapped to nodes in contiguous chunks
   HANDLE_CUDA_ERROR( cudaSetDevice(deviceId) );
   cudaDeviceProp prop;
   HANDLE_CUDA_ERROR( cudaGetDeviceProperties(&prop, deviceId) );

   // Sphinx: MPI #4 [end]

   if(verbose) {
      printf("===== rank 0 device info ======\n");
      printf("GPU-local-id:%d\n", deviceId);
      printf("GPU-name:%s\n", prop.name);
      printf("GPU-clock:%d\n", prop.clockRate);
      printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
      printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      printf("GPU-major:%d\n", prop.major);
      printf("GPU-minor:%d\n", prop.minor);
      printf("CUDA-available-devices:%d\n",numDevices);
      printf("CUDA_VISIBLE_DEVICES:%s\n",getenv("CUDA_VISIBLE_DEVICES") != nullptr ? getenv("CUDA_VISIBLE_DEVICES") : "");
      printf("===============================\n");
   } else {
      printf("===== rank %d device info ======\nGPU-local-id:%d\n", rank, deviceId);
   }

   typedef float floatType;
   MPI_Datatype floatTypeMPI = MPI_FLOAT;
   cudaDataType_t typeData = CUDA_R_32F;
   cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F;

   if(verbose)
      printf("Included headers and defined data types\n");

   // Sphinx: #2
   /**********************
   * Computing: R_{k,l} = A_{a,b,c,d,e,f} B_{b,g,h,e,i,j} C_{m,a,g,f,i,k} D_{l,c,h,d,j,m}
   **********************/

   constexpr int32_t numInputs = 4;

   // Create vectors of tensor modes
   std::vector<int32_t> modesA{'a','b','c','d','e','f'};
   std::vector<int32_t> modesB{'b','g','h','e','i','j'};
   std::vector<int32_t> modesC{'m','a','g','f','i','k'};
   std::vector<int32_t> modesD{'l','c','h','d','j','m'};
   std::vector<int32_t> modesR{'k','l'};

   // Set mode extents
   std::unordered_map<int32_t, int64_t> extent;
   extent['a'] = 16;
   extent['b'] = 16;
   extent['c'] = 16;
   extent['d'] = 16;
   extent['e'] = 16;
   extent['f'] = 16;
   extent['g'] = 16;
   extent['h'] = 16;
   extent['i'] = 16;
   extent['j'] = 16;
   extent['k'] = 16;
   extent['l'] = 16;
   extent['m'] = 16;

   // Create a vector of extents for each tensor
   std::vector<int64_t> extentA;
   for (auto mode : modesA)
      extentA.push_back(extent[mode]);
   std::vector<int64_t> extentB;
   for (auto mode : modesB)
      extentB.push_back(extent[mode]);
   std::vector<int64_t> extentC;
   for (auto mode : modesC)
      extentC.push_back(extent[mode]);
   std::vector<int64_t> extentD;
   for (auto mode : modesD)
      extentD.push_back(extent[mode]);
   std::vector<int64_t> extentR;
   for (auto mode : modesR)
      extentR.push_back(extent[mode]);

   if(verbose)
      printf("Defined tensor network, modes, and extents\n");

   // Sphinx: #3
   /**********************
   * Allocating data
   **********************/

   size_t elementsA = 1;
   for (auto mode : modesA)
      elementsA *= extent[mode];
   size_t elementsB = 1;
   for (auto mode : modesB)
      elementsB *= extent[mode];
   size_t elementsC = 1;
   for (auto mode : modesC)
      elementsC *= extent[mode];
   size_t elementsD = 1;
   for (auto mode : modesD)
      elementsD *= extent[mode];
   size_t elementsR = 1;
   for (auto mode : modesR)
      elementsR *= extent[mode];

   size_t sizeA = sizeof(floatType) * elementsA;
   size_t sizeB = sizeof(floatType) * elementsB;
   size_t sizeC = sizeof(floatType) * elementsC;
   size_t sizeD = sizeof(floatType) * elementsD;
   size_t sizeR = sizeof(floatType) * elementsR;
   if(verbose)
      printf("Total GPU memory used for tensor storage: %.2f GiB\n",
             (sizeA + sizeB + sizeC + sizeD + sizeR) / 1024. /1024. / 1024);

   void* rawDataIn_d[numInputs];
   void* R_d;
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[0], sizeA) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[1], sizeB) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[2], sizeC) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[3], sizeD) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &R_d, sizeR));

   floatType *A = (floatType*) malloc(sizeof(floatType) * elementsA);
   floatType *B = (floatType*) malloc(sizeof(floatType) * elementsB);
   floatType *C = (floatType*) malloc(sizeof(floatType) * elementsC);
   floatType *D = (floatType*) malloc(sizeof(floatType) * elementsD);
   floatType *R = (floatType*) malloc(sizeof(floatType) * elementsR);

   if (A == NULL || B == NULL || C == NULL || D == NULL || R == NULL)
   {
      printf("Error: Host memory allocation failed!\n");
      return -1;
   }

   // Sphinx: MPI #5 [begin]

   /*******************
   * Initialize data
   *******************/

   memset(R, 0, sizeof(floatType) * elementsR);
   if(rank == 0)
   {
      for (uint64_t i = 0; i < elementsA; i++)
         A[i] = ((floatType) rand()) / RAND_MAX;
      for (uint64_t i = 0; i < elementsB; i++)
         B[i] = ((floatType) rand()) / RAND_MAX;
      for (uint64_t i = 0; i < elementsC; i++)
         C[i] = ((floatType) rand()) / RAND_MAX;
      for (uint64_t i = 0; i < elementsD; i++)
         D[i] = ((floatType) rand()) / RAND_MAX;
   }

   // Broadcast input data to all ranks
   HANDLE_MPI_ERROR( MPI_Bcast(A, elementsA, floatTypeMPI, 0, MPI_COMM_WORLD) );
   HANDLE_MPI_ERROR( MPI_Bcast(B, elementsB, floatTypeMPI, 0, MPI_COMM_WORLD) );
   HANDLE_MPI_ERROR( MPI_Bcast(C, elementsC, floatTypeMPI, 0, MPI_COMM_WORLD) );
   HANDLE_MPI_ERROR( MPI_Bcast(D, elementsD, floatTypeMPI, 0, MPI_COMM_WORLD) );

   // Copy data to GPU
   HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[0], A, sizeA, cudaMemcpyHostToDevice) );
   HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[1], B, sizeB, cudaMemcpyHostToDevice) );
   HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[2], C, sizeC, cudaMemcpyHostToDevice) );
   HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[3], D, sizeD, cudaMemcpyHostToDevice) );

   if(verbose)
      printf("Allocated GPU memory for data, and initialize data\n");

   // Sphinx: MPI #5 [end]

   // Sphinx: #4
   /*************************
   * cuTensorNet
   *************************/

   cudaStream_t stream;
   HANDLE_CUDA_ERROR( cudaStreamCreate(&stream) );

   cutensornetHandle_t handle;
   HANDLE_ERROR( cutensornetCreate(&handle) );

   const int32_t nmodeA = modesA.size();
   const int32_t nmodeB = modesB.size();
   const int32_t nmodeC = modesC.size();
   const int32_t nmodeD = modesD.size();
   const int32_t nmodeR = modesR.size();

   /*******************************
   * Create Network Descriptor
   *******************************/

   const int32_t* modesIn[] = {modesA.data(), modesB.data(), modesC.data(), modesD.data()};
   int32_t const numModesIn[] = {nmodeA, nmodeB, nmodeC, nmodeD};
   const int64_t* extentsIn[] = {extentA.data(), extentB.data(), extentC.data(), extentD.data()};
   const int64_t* stridesIn[] = {NULL, NULL, NULL, NULL}; // strides are optional; if no stride is provided, cuTensorNet assumes a generalized column-major data layout

   // Set up tensor network
   cutensornetNetworkDescriptor_t descNet;
   HANDLE_ERROR( cutensornetCreateNetworkDescriptor(handle,
                     numInputs, numModesIn, extentsIn, stridesIn, modesIn, NULL,
                     nmodeR, extentR.data(), /*stridesOut = */NULL, modesR.data(),
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
   // Make sure all MPI processes will assume the minimal workspace size among all
   HANDLE_MPI_ERROR( MPI_Allreduce(MPI_IN_PLACE, &workspaceLimit, 1, MPI_INT64_T, MPI_MIN, MPI_COMM_WORLD) );
   if(verbose)
      printf("Workspace limit = %lu\n", workspaceLimit);

   /*******************************
   * Find "optimal" contraction order and slicing (in parallel)
   *******************************/

   cutensornetContractionOptimizerConfig_t optimizerConfig;
   HANDLE_ERROR( cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig) );

   // Set the desired number of hyper-samples (defaults to 0)
   int32_t num_hypersamples = 8;
   HANDLE_ERROR( cutensornetContractionOptimizerConfigSetAttribute(handle,
                     optimizerConfig,
                     CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES,
                     &num_hypersamples,
                     sizeof(num_hypersamples)) );

   // Create contraction optimizer info
   cutensornetContractionOptimizerInfo_t optimizerInfo;
   HANDLE_ERROR( cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo) );

   // Sphinx: MPI #6 [begin]

   // Compute the path on all ranks so that we can choose the path with the lowest cost. Note that since this is a tiny
   // example with 4 operands, all processes will compute the same globally optimal path. This is not the case for large
   // tensor networks. For large networks, hyper-optimization does become beneficial.

   // Enforce tensor network slicing (for parallelization)
   const int32_t min_slices = numProcs;
   HANDLE_ERROR( cutensornetContractionOptimizerConfigSetAttribute(handle,
                  optimizerConfig,
                  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES,
                  &min_slices,
                  sizeof(min_slices)) );

   // Find an optimized tensor network contraction path on each MPI process
   HANDLE_ERROR( cutensornetContractionOptimize(handle,
                                       descNet,
                                       optimizerConfig,
                                       workspaceLimit,
                                       optimizerInfo) );

   // Query the obtained Flop count
   double flops{-1.};
   HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(handle,
                     optimizerInfo,
                     CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                     &flops,
                     sizeof(flops)) );

   // Choose the contraction path with the lowest Flop cost
   struct {
      double value;
      int rank;
   } in{flops, rank}, out;
   HANDLE_MPI_ERROR( MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD) );
   const int sender = out.rank;
   flops = out.value;

   if (verbose)
      printf("Process %d has the path with the lowest FLOP count %lf\n", sender, flops);

   // Get the buffer size for optimizerInfo and broadcast it
   size_t bufSize {0};
   if (rank == sender)
   {
       HANDLE_ERROR( cutensornetContractionOptimizerInfoGetPackedSize(handle, optimizerInfo, &bufSize) );
   }
   HANDLE_MPI_ERROR( MPI_Bcast(&bufSize, 1, MPI_INT64_T, sender, MPI_COMM_WORLD) );

   // Allocate a buffer
   std::vector<char> buffer(bufSize);

   // Pack optimizerInfo on sender and broadcast it
   if (rank == sender)
   {
       HANDLE_ERROR( cutensornetContractionOptimizerInfoPackData(handle, optimizerInfo, buffer.data(), bufSize) );
   }
   HANDLE_MPI_ERROR( MPI_Bcast(buffer.data(), bufSize, MPI_CHAR, sender, MPI_COMM_WORLD) );

   // Unpack optimizerInfo from the buffer
   if (rank != sender)
   {
       HANDLE_ERROR( cutensornetUpdateContractionOptimizerInfoFromPackedData(handle, buffer.data(), bufSize, optimizerInfo) );
   }

   // Query the number of slices the tensor network execution will be split into
   int64_t numSlices = 0;
   HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
                  handle,
                  optimizerInfo,
                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                  &numSlices,
                  sizeof(numSlices)) );
   assert(numSlices > 0);

   // Calculate each process's share of the slices
   int64_t procChunk = numSlices / numProcs;
   int extra = numSlices % numProcs;
   int procSliceBegin = rank * procChunk + std::min(rank, extra);
   int procSliceEnd = (rank == numProcs - 1) ? numSlices : (rank + 1) * procChunk + std::min(rank + 1, extra);

   // Sphinx: MPI #6 [end]

   if(verbose)
      printf("Found an optimized contraction path using cuTensorNet optimizer\n");

   // Sphinx: #6
   /*******************************
   * Create workspace descriptor, allocate workspace, and set it.
   *******************************/

   cutensornetWorkspaceDescriptor_t workDesc;
   HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle, &workDesc) );

   int64_t requiredWorkspaceSize = 0;
   HANDLE_ERROR( cutensornetWorkspaceComputeContractionSizes(handle,
                                                         descNet,
                                                         optimizerInfo,
                                                         workDesc) );

   HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle,
                                                   workDesc,
                                                   CUTENSORNET_WORKSIZE_PREF_MIN,
                                                   CUTENSORNET_MEMSPACE_DEVICE,
                                                   CUTENSORNET_WORKSPACE_SCRATCH,
                                                   &requiredWorkspaceSize) );

   void* work = nullptr;
   HANDLE_CUDA_ERROR( cudaMalloc(&work, requiredWorkspaceSize) );

   HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle,
                                               workDesc,
                                               CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH,
                                               work,
                                               requiredWorkspaceSize) );

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
                                                R_d,
                                                workDesc,
                                                autotunePref,
                                                stream) );

   HANDLE_ERROR( cutensornetDestroyContractionAutotunePreference(autotunePref) );

   if(verbose)
      printf("Created a contraction plan for cuTensorNet and optionally auto-tuned it\n");

   // Sphinx: #8
   /**********************
   * Execute the tensor network contraction (in parallel)
   **********************/

   // Sphinx: MPI #7 [begin]

   // Create a cutensornetSliceGroup_t object from a range of slice IDs
   cutensornetSliceGroup_t sliceGroup{};
   HANDLE_ERROR( cutensornetCreateSliceGroupFromIDRange(handle, procSliceBegin, procSliceEnd, 1, &sliceGroup) );

   // Sphinx: MPI #7 [end]

   GPUTimer timer{stream};
   double minTimeCUTENSOR = 1e100;
   const int numRuns = 3; // to get stable performance results
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

      HANDLE_ERROR( cutensornetContractSlices(handle,
                                 plan,
                                 rawDataIn_d,
                                 R_d,
                                 accumulateOutput,
                                 workDesc,
                                 sliceGroup,
                                 stream) );

      // Sphinx: MPI #8 [end]

      // Sphinx: MPI #9 [begin]

      // Perform Allreduce operation on the output tensor
      HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );
      HANDLE_CUDA_ERROR( cudaMemcpy(R, R_d, sizeR, cudaMemcpyDeviceToHost) ); // restore the output tensor on Host
      HANDLE_MPI_ERROR( MPI_Allreduce(MPI_IN_PLACE, R, elementsR, floatTypeMPI, MPI_SUM, MPI_COMM_WORLD) );

      // Sphinx: MPI #9 [end]

      // Measure timing
      auto time = timer.seconds();
      minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
   }

   if (verbose)
      printf("Contracted the tensor network, all slices within the same rank used the same contraction plan.\n");

   // Print the 1-norm of the output tensor (verification)
   double norm1 = 0.0;
   for (int64_t i = 0; i < elementsR; ++i) {
      norm1 += std::abs(R[i]);
   }
   if(verbose)
      printf("Computed the 1-norm of the output tensor: %e\n", norm1);

   /*************************/

   // Query the total Flop count for the tensor network contraction
   flops  = 0.0;
   HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
                     handle,
                     optimizerInfo,
                     CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                     &flops,
                     sizeof(flops)) );

   if(verbose) {
      printf("Number of tensor network slices = %ld\n", numSlices);
      printf("Tensor network contraction time (ms) = %.3f\n", minTimeCUTENSOR * 1000.f);
   }

   // Free cuTensorNet resources
   HANDLE_ERROR( cutensornetDestroySliceGroup(sliceGroup) );
   HANDLE_ERROR( cutensornetDestroyContractionPlan(plan) );
   HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );
   HANDLE_ERROR( cutensornetDestroyContractionOptimizerInfo(optimizerInfo) );
   HANDLE_ERROR( cutensornetDestroyContractionOptimizerConfig(optimizerConfig) );
   HANDLE_ERROR( cutensornetDestroyNetworkDescriptor(descNet) );
   HANDLE_ERROR( cutensornetDestroy(handle) );

   // Free Host memory resources
   if (R) free(R);
   if (D) free(D);
   if (C) free(C);
   if (B) free(B);
   if (A) free(A);

   // Free GPU memory resources
   if (work) cudaFree(work);
   if (R_d) cudaFree(R_d);
   if (rawDataIn_d[0]) cudaFree(rawDataIn_d[0]);
   if (rawDataIn_d[1]) cudaFree(rawDataIn_d[1]);
   if (rawDataIn_d[2]) cudaFree(rawDataIn_d[2]);
   if (rawDataIn_d[3]) cudaFree(rawDataIn_d[3]);

   // Sphinx: MPI #10 [begin]

   // Shut down MPI service
   HANDLE_MPI_ERROR( MPI_Finalize() );

   // Sphinx: MPI #10 [end]

   if(verbose)
      printf("Freed resources and exited\n");

   return 0;
}
