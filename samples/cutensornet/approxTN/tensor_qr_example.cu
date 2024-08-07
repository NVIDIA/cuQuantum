/*  
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
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
if( err != CUTENSORNET_STATUS_SUCCESS )                           \
{ printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); return err; } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{  const auto err = x;                                            \
   if( err != cudaSuccess )                                       \
   { printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); return err; } \
};

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

int64_t computeCombinedExtent(const std::unordered_map<int32_t, int64_t> &extentMap, 
                                const std::vector<int32_t> &modes)
{
   int64_t combinedExtent{1};
   for (auto mode: modes)
   {
      auto it = extentMap.find(mode);
      if (it != extentMap.end())
         combinedExtent *= it->second;
   }
   return combinedExtent;
}

int main()
{  
   const size_t cuTensornetVersion = cutensornetGetVersion();
   printf("cuTensorNet-vers:%ld\n",cuTensornetVersion);

   cudaDeviceProp prop;
   int deviceId{-1};
   HANDLE_CUDA_ERROR( cudaGetDevice(&deviceId) );
   HANDLE_CUDA_ERROR( cudaGetDeviceProperties(&prop, deviceId) );

   printf("===== device info ======\n");
   printf("GPU-local-id:%d\n", deviceId);
   printf("GPU-name:%s\n", prop.name);
   printf("GPU-clock:%d\n", prop.clockRate);
   printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
   printf("GPU-nSM:%d\n", prop.multiProcessorCount);
   printf("GPU-major:%d\n", prop.major);
   printf("GPU-minor:%d\n", prop.minor);
   printf("========================\n");

   // Sphinx: #2
   /**********************************************
   * Tensor QR: T_{i,j,m,n} -> Q_{i,x,m} R_{n,x,j}  
   ***********************************************/

   typedef float floatType;
   cudaDataType_t typeData = CUDA_R_32F;

   // Create vector of modes
   int32_t sharedMode = 'x';

   std::vector<int32_t> modesT{'i','j','m','n'}; // input
   std::vector<int32_t> modesQ{'i', sharedMode,'m'};
   std::vector<int32_t> modesR{'n', sharedMode,'j'};  // QR output

   // Extents
   std::unordered_map<int32_t, int64_t> extentMap;
   extentMap['i'] = 16;
   extentMap['j'] = 16;
   extentMap['m'] = 16;
   extentMap['n'] = 16;

   int64_t rowExtent = computeCombinedExtent(extentMap, modesQ);
   int64_t colExtent = computeCombinedExtent(extentMap, modesR);
   
   // cuTensorNet tensor QR operates in reduced mode expecting k = min(m, n)
   extentMap[sharedMode] = rowExtent <= colExtent? rowExtent: colExtent;

   // Create a vector of extents for each tensor
   std::vector<int64_t> extentT;
   for (auto mode : modesT)
      extentT.push_back(extentMap[mode]);
   std::vector<int64_t> extentQ;
   for (auto mode : modesQ)
      extentQ.push_back(extentMap[mode]);
   std::vector<int64_t> extentR;
   for (auto mode : modesR)
      extentR.push_back(extentMap[mode]);

   // Sphinx: #3
   /***********************************
   * Allocating data on host and device
   ************************************/

   size_t elementsT = 1;
   for (auto mode : modesT)
      elementsT *= extentMap[mode];
   size_t elementsQ = 1;
   for (auto mode : modesQ)
      elementsQ *= extentMap[mode];
   size_t elementsR = 1;
   for (auto mode : modesR)
      elementsR *= extentMap[mode];

   size_t sizeT = sizeof(floatType) * elementsT;
   size_t sizeQ = sizeof(floatType) * elementsQ;
   size_t sizeR = sizeof(floatType) * elementsR;

   printf("Total memory: %.2f GiB\n", (sizeT + sizeQ + sizeR)/1024./1024./1024);

   floatType *T = (floatType*) malloc(sizeT);
   floatType *Q = (floatType*) malloc(sizeQ);
   floatType *R = (floatType*) malloc(sizeR);

   if (T == NULL || Q==NULL || R==NULL )
   {
      printf("Error: Host allocation of input T or output Q/R.\n");
      return -1;
   }

   void* D_T;
   void* D_Q;
   void* D_R;
   
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_T, sizeT) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_Q, sizeQ) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_R, sizeR) );

   /****************
   * Initialize data
   *****************/

   for (uint64_t i = 0; i < elementsT; i++)
      T[i] = ((floatType) rand())/RAND_MAX;

   HANDLE_CUDA_ERROR( cudaMemcpy(D_T, T, sizeT, cudaMemcpyHostToDevice) );
   printf("Allocate memory for data, and initialize data.\n");

   // Sphinx: #4
   /******************
   * cuTensorNet
   *******************/

   cudaStream_t stream;
   HANDLE_CUDA_ERROR( cudaStreamCreate(&stream) );

   cutensornetHandle_t handle;
   HANDLE_ERROR( cutensornetCreate(&handle) );

   /***************************
   * Create tensor descriptors
   ****************************/

   cutensornetTensorDescriptor_t descTensorIn;
   cutensornetTensorDescriptor_t descTensorQ;
   cutensornetTensorDescriptor_t descTensorR;

   const int32_t numModesIn = modesT.size();
   const int32_t numModesQ = modesQ.size();
   const int32_t numModesR = modesR.size();

   const int64_t* strides = NULL; // assuming fortran layout for all tensors

   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides, modesT.data(), typeData, &descTensorIn) );
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesQ, extentQ.data(), strides, modesQ.data(), typeData, &descTensorQ) );
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesR, extentR.data(), strides, modesR.data(), typeData, &descTensorR) );
    
   printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

   // Sphinx: #5
   /********************************************
   * Query and allocate required workspace sizes
   *********************************************/

   cutensornetWorkspaceDescriptor_t workDesc;
   HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle, &workDesc) );
   HANDLE_ERROR( cutensornetWorkspaceComputeQRSizes(handle, descTensorIn, descTensorQ, descTensorR, workDesc) );
   int64_t hostWorkspaceSize, deviceWorkspaceSize;

   // for tensor QR, it does not matter which cutensornetWorksizePref_t we pick
   HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle,
                                                   workDesc,
                                                   CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                   CUTENSORNET_MEMSPACE_DEVICE,
                                                   CUTENSORNET_WORKSPACE_SCRATCH,
                                                   &deviceWorkspaceSize) );
   HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle,
                                                   workDesc,
                                                   CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                   CUTENSORNET_MEMSPACE_HOST,
                                                   CUTENSORNET_WORKSPACE_SCRATCH,
                                                   &hostWorkspaceSize) );

   void *devWork = nullptr, *hostWork = nullptr;
   if (deviceWorkspaceSize > 0) {
      HANDLE_CUDA_ERROR( cudaMalloc(&devWork, deviceWorkspaceSize) );
   }
   if (hostWorkspaceSize > 0) {
      hostWork = malloc(hostWorkspaceSize);
   }
   HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle,
                                               workDesc,
                                               CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH,
                                               devWork,
                                               deviceWorkspaceSize) );
   HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle,
                                               workDesc,
                                               CUTENSORNET_MEMSPACE_HOST,
                                               CUTENSORNET_WORKSPACE_SCRATCH,
                                               hostWork,
                                               hostWorkspaceSize) );

   // Sphinx: #6
   /**********
   * Execution
   ***********/
  
   GPUTimer timer{stream};
   double minTimeCUTENSOR = 1e100;
   const int numRuns = 3; // to get stable perf results
   for (int i=0; i < numRuns; ++i)
   {  
      // restore output
      cudaMemsetAsync(D_Q, 0, sizeQ, stream);
      cudaMemsetAsync(D_R, 0, sizeR, stream);
      cudaDeviceSynchronize();

      timer.start();
      HANDLE_ERROR( cutensornetTensorQR(handle, 
                        descTensorIn, D_T, 
                        descTensorQ, D_Q, 
                        descTensorR, D_R, 
                        workDesc,
                        stream) );
      // Synchronize and measure timing
      auto time = timer.seconds();
      minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
   }

   printf("Performing QR\n");

   HANDLE_CUDA_ERROR( cudaMemcpyAsync(Q, D_Q, sizeQ, cudaMemcpyDeviceToHost) );
   HANDLE_CUDA_ERROR( cudaMemcpyAsync(R, D_R, sizeR, cudaMemcpyDeviceToHost) );

   cudaDeviceSynchronize(); // device synchronization.
   printf("%.2f ms\n", minTimeCUTENSOR * 1000.f);

   // Sphinx: #7
   /***************
   * Free resources
   ****************/
   
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorIn) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorQ) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorR) );
   HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );
   HANDLE_ERROR( cutensornetDestroy(handle) );

   if (T) free(T);
   if (Q) free(Q);
   if (R) free(R);
   if (D_T) cudaFree(D_T);
   if (D_Q) cudaFree(D_Q);
   if (D_R) cudaFree(D_R);
   if (devWork) cudaFree(devWork);
   if (hostWork) free(hostWork);

   printf("Free resource and exit.\n");

   return 0;
}
