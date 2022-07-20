/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: #1

// Sphinx: MPI #1 [begin]
#include <mpi.h>
// Sphinx: MPI #1 [end]

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
  { printf("[Process %d] Error: %s in line %d\n", rank, cutensornetGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
    MPI_Abort(MPI_COMM_WORLD, err);                               \
  }                                                               \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("[Process %d] CUDA Error: %s in line %d\n", rank, cudaGetErrorString(err), __LINE__);  \
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
    printf("[Process %d] MPI Error: %s in line %d\n", rank, error, __LINE__); \
    fflush(stdout);                                               \
    MPI_Abort(MPI_COMM_WORLD, err);                               \
  }                                                               \
};

// Sphinx: MPI #2 [end]


struct GPUTimer
{
   GPUTimer(cudaStream_t stream): stream_(stream)
   {
      cudaEventCreate(&start_);
      cudaEventCreate(&stop_);
   }

   ~GPUTimer()
   {
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
   }

   void start()
   {
      cudaEventRecord(start_, stream_);
   }

   float seconds()
   {
      cudaEventRecord(stop_, stream_);
      cudaEventSynchronize(stop_);
      float time;
      cudaEventElapsedTime(&time, start_, stop_);
      return time * 1e-3;
   }

   private:
   cudaEvent_t start_, stop_;
   cudaStream_t stream_;
};


int main(int argc, char *argv[])
{
   static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture.");

   // Sphinx: MPI #3 [begin]

   // Initialize MPI.
   int errorCode = MPI_Init(&argc, &argv);
   if (errorCode != MPI_SUCCESS)
   {
      printf("Error initializing MPI.\n");
      MPI_Abort(MPI_COMM_WORLD, errorCode);
   }

   const int root{0};
   int rank{};
   HANDLE_MPI_ERROR( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );

   int numProcs{};
   HANDLE_MPI_ERROR( MPI_Comm_size(MPI_COMM_WORLD, &numProcs) );

   // Sphinx: MPI #3 [end]

   if (rank == root)
   {
      printf("*** Printing is done only from the root process to prevent jumbled messages ***\n");
      printf("The number of processes is %d.\n", numProcs);
   }

   // Get cuTensornet version and device properties.
   const size_t cuTensornetVersion = cutensornetGetVersion();
   if (rank == root)
      printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);

   int numDevices;
   HANDLE_CUDA_ERROR( cudaGetDeviceCount(&numDevices) );

   cudaDeviceProp prop;

   // Sphinx: MPI #4 [begin]

   // Set deviceId based on ranks and nodes.
   int deviceId = rank % numDevices;    // We assume that the processes are mapped to nodes in contiguous chunks.
   HANDLE_CUDA_ERROR( cudaSetDevice(deviceId) );
   HANDLE_CUDA_ERROR( cudaGetDeviceProperties(&prop, deviceId) );

   // Sphinx: MPI #4 [end]

   if (rank == root)
   {
      printf("===== root process device info ======\n");
      printf("GPU-name:%s\n", prop.name);
      printf("GPU-clock:%d\n", prop.clockRate);
      printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
      printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      printf("GPU-major:%d\n", prop.major);
      printf("GPU-minor:%d\n", prop.minor);
      printf("========================\n");
   }

   typedef float floatType;
   MPI_Datatype floatTypeMPI = MPI_FLOAT;
   cudaDataType_t typeData = CUDA_R_32F;
   cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F;
   auto Absolute = fabsf;

   if (rank == root)
      printf("Include headers and define data types\n");

   // Sphinx: #2
   /**********************
   * Computing: D_{m,x,n,y} = A_{m,h,k,n} B_{u,k,h} C_{x,u,y}
   **********************/

   constexpr int32_t numInputs = 3;

   // Create vector of modes
   std::vector<int32_t> modesA{'m','h','k','n'};
   std::vector<int32_t> modesB{'u','k','h'};
   std::vector<int32_t> modesC{'x','u','y'};
   std::vector<int32_t> modesD{'m','x','n','y'};

   // Extents
   std::unordered_map<int32_t, int64_t> extent;
   extent['m'] = 96;
   extent['n'] = 96;
   extent['u'] = 96;
   extent['h'] = 64;
   extent['k'] = 64;
   extent['x'] = 64;
   extent['y'] = 64;

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

   if (rank == root)
      printf("Define network, modes, and extents\n");

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

   size_t sizeA = sizeof(floatType) * elementsA;
   size_t sizeB = sizeof(floatType) * elementsB;
   size_t sizeC = sizeof(floatType) * elementsC;
   size_t sizeD = sizeof(floatType) * elementsD;
   if (rank == root)
      printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC + sizeD)/1024./1024./1024);

   void* rawDataIn_d[numInputs];
   void* D_d;
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[0], sizeA) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[1], sizeB) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[2], sizeC) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_d, sizeD));

   floatType *A = (floatType*) malloc(sizeof(floatType) * elementsA);
   floatType *B = (floatType*) malloc(sizeof(floatType) * elementsB);
   floatType *C = (floatType*) malloc(sizeof(floatType) * elementsC);
   floatType *D = (floatType*) malloc(sizeof(floatType) * elementsD);

   if (A == NULL || B == NULL || C == NULL || D == NULL)
   {
      printf("Process %d: Error: Host allocation of A, B, C, or D.\n", rank);
      MPI_Abort(MPI_COMM_WORLD, -1);

   }

   // Sphinx: MPI #5 [begin]

   /*******************
   * Initialize data
   *******************/

   // Rank root creates the tensor data.
   if (rank == root)
   {
      for (uint64_t i = 0; i < elementsA; i++)
         A[i] = ((floatType) rand())/RAND_MAX;
      for (uint64_t i = 0; i < elementsB; i++)
         B[i] = ((floatType) rand())/RAND_MAX;
      for (uint64_t i = 0; i < elementsC; i++)
         C[i] = ((floatType) rand())/RAND_MAX;
   }

   // Broadcast data to all ranks.
   HANDLE_MPI_ERROR( MPI_Bcast(A, elementsA, floatTypeMPI, root, MPI_COMM_WORLD) );
   HANDLE_MPI_ERROR( MPI_Bcast(B, elementsB, floatTypeMPI, root, MPI_COMM_WORLD) );
   HANDLE_MPI_ERROR( MPI_Bcast(C, elementsC, floatTypeMPI, root, MPI_COMM_WORLD) );

   // Copy data onto the device on all ranks.
   HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[0], A, sizeA, cudaMemcpyHostToDevice) );
   HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[1], B, sizeB, cudaMemcpyHostToDevice) );
   HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[2], C, sizeC, cudaMemcpyHostToDevice) );

   // Sphinx: MPI #5 [end]

   if (rank == root)
      printf("Allocate memory for data, calculate workspace limit, and initialize data.\n");

   // Sphinx: #4
   /*************************
   * cuTensorNet
   *************************/

   cudaStream_t stream;
   cudaStreamCreate(&stream);
   cutensornetHandle_t handle;
   HANDLE_ERROR( cutensornetCreate(&handle) );

   const int32_t nmodeA = modesA.size();
   const int32_t nmodeB = modesB.size();
   const int32_t nmodeC = modesC.size();
   const int32_t nmodeD = modesD.size();

   /*******************************
   * Create Network Descriptor
   *******************************/

   const int32_t* modesIn[] = {modesA.data(), modesB.data(), modesC.data()};
   int32_t const numModesIn[] = {nmodeA, nmodeB, nmodeC};
   const int64_t* extentsIn[] = {extentA.data(), extentB.data(), extentC.data()};
   const int64_t* stridesIn[] = {NULL, NULL, NULL}; // strides are optional; if no stride is provided, then cuTensorNet assumes a generalized column-major data layout

   // Notice that pointers are allocated via cudaMalloc are aligned to 256 byte
   // boundaries by default; however here we're checking the pointer alignment explicitly
   // to demonstrate how one would check the alginment for arbitrary pointers.

   auto getMaximalPointerAlignment = [](const void* ptr) {
      const uint64_t ptrAddr  = reinterpret_cast<uint64_t>(ptr);
      uint32_t alignment = 1;
      while(ptrAddr % alignment == 0 &&
            alignment < 256) // at the latest we terminate once the alignment reached 256 bytes (we could be going, but any alignment larger or equal to 256 is equally fine)
      {
         alignment *= 2;
      }
      return alignment;
   };
   const uint32_t alignmentsIn[] = {getMaximalPointerAlignment(rawDataIn_d[0]),
                                    getMaximalPointerAlignment(rawDataIn_d[1]),
                                    getMaximalPointerAlignment(rawDataIn_d[2])};
   const uint32_t alignmentOut = getMaximalPointerAlignment(D_d);

   // setup tensor network
   cutensornetNetworkDescriptor_t descNet;
   HANDLE_ERROR( cutensornetCreateNetworkDescriptor(handle,
                                                numInputs, numModesIn, extentsIn, stridesIn, modesIn, alignmentsIn,
                                                nmodeD, extentD.data(), /*stridesOut = */NULL, modesD.data(), alignmentOut,
                                                typeData, typeCompute,
                                                &descNet) );

   if (rank == root)
      printf("Initialize the cuTensorNet library and create a network descriptor.\n");

   // Sphinx: #5
   /*******************************
   * Choose workspace limit based on available resources.
   *******************************/

   size_t freeMem, totalMem;
   HANDLE_CUDA_ERROR( cudaMemGetInfo(&freeMem, &totalMem ) );
   HANDLE_MPI_ERROR( MPI_Allreduce(MPI_IN_PLACE, &totalMem, 1, MPI_INT64_T, MPI_MIN, MPI_COMM_WORLD) );
   uint64_t workspaceLimit = totalMem * 0.9;

   /*******************************
   * Find "optimal" contraction order and slicing
   *******************************/

   cutensornetContractionOptimizerConfig_t optimizerConfig;
   HANDLE_ERROR( cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig) );

   cutensornetContractionOptimizerInfo_t optimizerInfo;
   HANDLE_ERROR(cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo) );

   // Sphinx: MPI #6 [begin]

   // Compute the path on all ranks so that we can choose the path with the lowest cost. Note that since this is a tiny
   //   example with 3 operands, all processes will compute the same globally optimal path. This is not the case for large
   //   tensor networks. For large networks, hyperoptimization is also beneficial and can be enabled by setting the
   //   optimizer config attribute CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES.

   // Force slicing.
   int32_t min_slices = numProcs;
   HANDLE_ERROR( cutensornetContractionOptimizerConfigSetAttribute(
                                                               handle,
                                                               optimizerConfig,
                                                               CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES,
                                                               &min_slices,
                                                               sizeof(min_slices)) );

   HANDLE_ERROR( cutensornetContractionOptimize(handle,
                                             descNet,
                                             optimizerConfig,
                                             workspaceLimit,
                                             optimizerInfo) );

   double flops{-1.};
   HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
                                                               handle,
                                                               optimizerInfo,
                                                               CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                                                               &flops,
                                                               sizeof(flops)) );

   // Choose the path with the lowest cost.
   struct {
       double value;
       int rank;
   } in{flops, rank}, out;

   HANDLE_MPI_ERROR( MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD) );

   int sender = out.rank;
   flops = out.value;
   if (rank == root)
   {
       printf("Process %d has the path with the lowest FLOP count %lf.\n", sender, flops);
   }

   size_t bufSize;

   // Get buffer size for optimizerInfo and broadcast it.
   if (rank == sender)
   {
       HANDLE_ERROR( cutensornetContractionOptimizerInfoGetPackedSize(handle, optimizerInfo, &bufSize) );
   }

   HANDLE_MPI_ERROR( MPI_Bcast(&bufSize, 1, MPI_INT64_T, sender, MPI_COMM_WORLD) );

   // Allocate buffer.
   std::vector<char> buffer(bufSize);

   // Pack optimizerInfo on sender and broadcast it.
   if (rank == sender)
   {
       HANDLE_ERROR( cutensornetContractionOptimizerInfoPackData(handle, optimizerInfo, buffer.data(), bufSize) );
   }

   HANDLE_MPI_ERROR( MPI_Bcast(buffer.data(), bufSize, MPI_CHAR, sender, MPI_COMM_WORLD) );

   // Unpack optimizerInfo from buffer.
   if (rank != sender)
   {
       HANDLE_ERROR( cutensornetUpdateContractionOptimizerInfoFromPackedData(handle, buffer.data(), bufSize, optimizerInfo) );
   }

   int64_t numSlices = 0;
   HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
               handle,
               optimizerInfo,
               CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
               &numSlices,
               sizeof(numSlices)) );

   assert(numSlices > 0);

   // Calculate each process's share of the slices.

   int64_t procChunk = numSlices / numProcs;
   int extra = numSlices % numProcs;
   int procSliceBegin = rank * procChunk + std::min(rank, extra);
   int procSliceEnd =  rank == numProcs - 1 ? numSlices : (rank + 1) * procChunk + std::min(rank + 1, extra);

   // Sphinx: MPI #6 [end]

   if (rank == root)
      printf("Find an optimized contraction path with cuTensorNet optimizer.\n");

   // Sphinx: #6
   /*******************************
   * Create workspace descriptor, allocate workspace, and set it.
   *******************************/

   cutensornetWorkspaceDescriptor_t workDesc;
   HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle, &workDesc) );

   uint64_t requiredWorkspaceSize = 0;
   HANDLE_ERROR( cutensornetWorkspaceComputeSizes(handle,
                                          descNet,
                                          optimizerInfo,
                                          workDesc) );

   HANDLE_ERROR( cutensornetWorkspaceGetSize(handle,
                                         workDesc,
                                         CUTENSORNET_WORKSIZE_PREF_MIN,
                                         CUTENSORNET_MEMSPACE_DEVICE,
                                         &requiredWorkspaceSize) );

   void *work = nullptr;
   HANDLE_CUDA_ERROR( cudaMalloc(&work, requiredWorkspaceSize) );

   HANDLE_ERROR( cutensornetWorkspaceSet(handle,
                                         workDesc,
                                         CUTENSORNET_MEMSPACE_DEVICE,
                                         work,
                                         requiredWorkspaceSize) );

   if (rank == root)
      printf("Allocate workspace.\n");

   // Sphinx: #7
   /*******************************
   * Initialize all pair-wise contraction plans (for cuTENSOR)
   *******************************/

   cutensornetContractionPlan_t plan;

   HANDLE_ERROR( cutensornetCreateContractionPlan(handle,
                                                  descNet,
                                                  optimizerInfo,
                                                  workDesc,
                                                  &plan) );

   /*******************************
   * Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
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

   // modify the plan again to find the best pair-wise contractions
   HANDLE_ERROR( cutensornetContractionAutotune(handle,
                           plan,
                           rawDataIn_d,
                           D_d,
                           workDesc,
                           autotunePref,
                           stream) );

   HANDLE_ERROR( cutensornetDestroyContractionAutotunePreference(autotunePref) );

   if (rank == root)
      printf("Create a contraction plan for cuTensorNet and optionally auto-tune it.\n");

   // Sphinx: #8
   /**********************
   * Run
   **********************/

   // Sphinx: MPI #7 [begin]

   cutensornetSliceGroup_t sliceGroup{};
   // Create a cutensornetSliceGroup_t object from a range of slice IDs.
   HANDLE_ERROR( cutensornetCreateSliceGroupFromIDRange(handle, procSliceBegin, procSliceEnd, 1, &sliceGroup) );

   // Sphinx: MPI #7 [end]

   GPUTimer timer{stream};
   double minTimeCUTENSOR = 1e100;
   const int numRuns = 3; // to get stable perf results
   for (int i=0; i < numRuns; ++i)
   {
      cudaDeviceSynchronize();

      /*
      * Contract over the range of slices this process is responsible for.
      */
      timer.start();

      // Don't accumulate into output since we use a one-process-per-gpu model.
      int32_t accumulateOutput = 0;

      // Sphinx: MPI #8 [begin]

      HANDLE_ERROR( cutensornetContractSlices(handle,
                                 plan,
                                 rawDataIn_d,
                                 D_d,
                                 accumulateOutput,
                                 workDesc,
                                 sliceGroup,
                                 stream) );

      // Sphinx: MPI #8 [end]

      // Synchronize and measure timing
      auto time = timer.seconds();
      minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
   }

   if (rank == root)
      printf("Contract the network, all slices within the same rank use the same contraction plan.\n");

   /*************************/

   if (rank == root)
   {
      printf("numSlices: %ld\n", numSlices);
      int64_t numSlicesProc = procSliceEnd - procSliceBegin;
      printf("numSlices on root process: %ld\n", numSlicesProc);
      if (numSlicesProc > 0)
         printf("%.2f ms / slice\n", minTimeCUTENSOR * 1000.f / numSlicesProc);
   }

   HANDLE_ERROR( cutensornetDestroySliceGroup(sliceGroup) );

   HANDLE_CUDA_ERROR( cudaMemcpy(D, D_d, sizeD, cudaMemcpyDeviceToHost) );

   // Sphinx: MPI #9 [begin]

   // Reduce on root process.
   if (rank == root)
   {
      HANDLE_MPI_ERROR( MPI_Reduce(MPI_IN_PLACE, D, elementsD, floatTypeMPI, MPI_SUM, root, MPI_COMM_WORLD) );
   }
   else
   {
      HANDLE_MPI_ERROR( MPI_Reduce(D, D, elementsD, floatTypeMPI, MPI_SUM, root, MPI_COMM_WORLD) );
   }

   // Sphinx: MPI #9 [end]

   // Compute the reference result.
   if (rank == root)
   {
      floatType *Reference = (floatType*) malloc(sizeof(floatType) * elementsD);
      if (Reference == NULL)
      {
         printf("Error: Host allocation of Reference.\n");
         MPI_Abort(MPI_COMM_WORLD, -1);
      }

      void *Reference_d;
      HANDLE_CUDA_ERROR( cudaMalloc((void**) &Reference_d, sizeD) );

      int32_t accumulateOutput = 0;
      HANDLE_ERROR( cutensornetContractSlices(handle,
                                 plan,
                                 rawDataIn_d,
                                 Reference_d,
                                 accumulateOutput,
                                 workDesc,
                                 NULL,    // Contract over all the slices.
                                 stream) );
      cudaDeviceSynchronize();
      HANDLE_CUDA_ERROR( cudaMemcpy(Reference, Reference_d, sizeD, cudaMemcpyDeviceToHost) );

      // Calculate the error.
      floatType max{}, maxError{};
      for (int i=0; i < elementsD; ++i)
      {
         floatType error = Absolute(D[i] - Reference[i]);
         if (error > maxError)
            maxError = error;
         if (Absolute(Reference[i]) > max)
            max = Absolute(Reference[i]);
      }
      printf("The inf norm of the reference result is %f, the maximum absolute error is %f, and the maximum relative error is %e.\n", max, maxError, maxError/max);

      free(Reference);
      cudaFree(Reference_d);
   }

   HANDLE_ERROR( cutensornetDestroy(handle) );
   HANDLE_ERROR( cutensornetDestroyNetworkDescriptor(descNet) );
   HANDLE_ERROR( cutensornetDestroyContractionPlan(plan) );
   HANDLE_ERROR( cutensornetDestroyContractionOptimizerConfig(optimizerConfig) );
   HANDLE_ERROR( cutensornetDestroyContractionOptimizerInfo(optimizerInfo) );
   HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );

   if (A) free(A);
   if (B) free(B);
   if (C) free(C);
   if (D) free(D);
   if (rawDataIn_d[0]) cudaFree(rawDataIn_d[0]);
   if (rawDataIn_d[1]) cudaFree(rawDataIn_d[1]);
   if (rawDataIn_d[2]) cudaFree(rawDataIn_d[2]);
   if (D_d) cudaFree(D_d);
   if (work) cudaFree(work);

   if (rank == root)
      printf("Free resources and exit.\n");

   // Sphinx: MPI #10 [begin]

   HANDLE_MPI_ERROR( MPI_Finalize() );

   // Sphinx: MPI #10 [end]

   return 0;
}
