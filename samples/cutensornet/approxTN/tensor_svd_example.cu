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
   /******************************************************
   * Tensor SVD: T_{i,j,m,n} -> U_{i,x,m} S_{x} V_{n,x,j}  
   *******************************************************/

   typedef float floatType;
   cudaDataType_t typeData = CUDA_R_32F;

   // Create vector of modes
   int32_t sharedMode = 'x';

   std::vector<int32_t> modesT{'i','j','m','n'}; // input
   std::vector<int32_t> modesU{'i', sharedMode,'m'};
   std::vector<int32_t> modesV{'n', sharedMode,'j'};  // SVD output

   // Extents
   std::unordered_map<int32_t, int64_t> extentMap;
   extentMap['i'] = 16;
   extentMap['j'] = 16;
   extentMap['m'] = 16;
   extentMap['n'] = 16;

   int64_t rowExtent = computeCombinedExtent(extentMap, modesU);
   int64_t colExtent = computeCombinedExtent(extentMap, modesV);
   // cuTensorNet tensor SVD operates in reduced mode expecting k <= min(m, n)
   int64_t fullSharedExtent = rowExtent <= colExtent? rowExtent: colExtent;
   const int64_t maxExtent = fullSharedExtent / 2;  //fix extent truncation with half of the singular values trimmed out
   extentMap[sharedMode] = maxExtent;

   // Create a vector of extents for each tensor
   std::vector<int64_t> extentT;
   for (auto mode : modesT)
      extentT.push_back(extentMap[mode]);
   std::vector<int64_t> extentU;
   for (auto mode : modesU)
      extentU.push_back(extentMap[mode]);
   std::vector<int64_t> extentV;
   for (auto mode : modesV)
      extentV.push_back(extentMap[mode]);

   // Sphinx: #3
   /***********************************
   * Allocating data on host and device
   ************************************/

   size_t elementsT = 1;
   for (auto mode : modesT)
      elementsT *= extentMap[mode];
   size_t elementsU = 1;
   for (auto mode : modesU)
      elementsU *= extentMap[mode];
   size_t elementsV = 1;
   for (auto mode : modesV)
      elementsV *= extentMap[mode];

   size_t sizeT = sizeof(floatType) * elementsT;
   size_t sizeU = sizeof(floatType) * elementsU;
   size_t sizeS = sizeof(floatType) * extentMap[sharedMode];
   size_t sizeV = sizeof(floatType) * elementsV;

   printf("Total memory: %.2f GiB\n", (sizeT + sizeU + sizeS + sizeV)/1024./1024./1024);

   floatType *T = (floatType*) malloc(sizeT);
   floatType *U = (floatType*) malloc(sizeU);
   floatType *S = (floatType*) malloc(sizeS);
   floatType *V = (floatType*) malloc(sizeV);

   if (T == NULL || U==NULL || S==NULL || V==NULL)
   {
      printf("Error: Host allocation of input T or output U/S/V.\n");
      return -1;
   }

   void* D_T;
   void* D_U;
   void* D_S;
   void* D_V;
   
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_T, sizeT) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_U, sizeU) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_S, sizeS) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_V, sizeV) );

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

   /**************************
   * Create tensor descriptors
   ***************************/

   cutensornetTensorDescriptor_t descTensorIn;
   cutensornetTensorDescriptor_t descTensorU;
   cutensornetTensorDescriptor_t descTensorV;

   const int32_t numModesIn = modesT.size();
   const int32_t numModesU = modesU.size();
   const int32_t numModesV = modesV.size();

   const int64_t* strides = NULL; // assuming fortran layout for all tensors

   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides, modesT.data(), typeData, &descTensorIn) );
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides, modesU.data(), typeData, &descTensorU) );
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides, modesV.data(), typeData, &descTensorV) );
    
   printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

   // Sphinx: #5
   /**********************************************
   * Setup SVD algorithm and truncation parameters
   ***********************************************/

   cutensornetTensorSVDConfig_t svdConfig;
   HANDLE_ERROR( cutensornetCreateTensorSVDConfig(handle, &svdConfig) );

   // set up truncation parameters
   double absCutoff = 1e-2;
   HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, 
                                          svdConfig, 
                                          CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF, 
                                          &absCutoff, 
                                          sizeof(absCutoff)) );
   double relCutoff = 4e-2;
   HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, 
                                          svdConfig, 
                                          CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF, 
                                          &relCutoff, 
                                          sizeof(relCutoff)) );
   
   // optional: choose gesvdj algorithm with customized parameters. Default is gesvd.
   cutensornetTensorSVDAlgo_t svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
   HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, 
                                          svdConfig, 
                                          CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, 
                                          &svdAlgo, 
                                          sizeof(svdAlgo)) );
   cutensornetGesvdjParams_t gesvdjParams{/*tol=*/1e-12, /*maxSweeps=*/80};
   HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, 
                                          svdConfig, 
                                          CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS, 
                                          &gesvdjParams, 
                                          sizeof(gesvdjParams)) );
   printf("Set up SVDConfig to use GESVDJ algorithm with truncation\n");
   
   /********************************************************
   * Create SVDInfo to record runtime SVD truncation details
   *********************************************************/

   cutensornetTensorSVDInfo_t svdInfo; 
   HANDLE_ERROR( cutensornetCreateTensorSVDInfo(handle, &svdInfo)) ;

   // Sphinx: #6
   /**************************************************************
   * Query the required workspace sizes and allocate memory
   **************************************************************/

   cutensornetWorkspaceDescriptor_t workDesc;
   HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle, &workDesc) );
   HANDLE_ERROR( cutensornetWorkspaceComputeSVDSizes(handle, descTensorIn, descTensorU, descTensorV, svdConfig, workDesc) );
   int64_t hostWorkspaceSize, deviceWorkspaceSize;
   // for tensor SVD, it does not matter which cutensornetWorksizePref_t we pick
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

   // Sphinx: #7
   /**********
   * Execution
   ***********/
  
   GPUTimer timer{stream};
   double minTimeCUTENSOR = 1e100;
   const int numRuns = 3; // to get stable perf results
   for (int i=0; i < numRuns; ++i)
   {  
      // restore output
      cudaMemsetAsync(D_U, 0, sizeU, stream);
      cudaMemsetAsync(D_S, 0, sizeS, stream);
      cudaMemsetAsync(D_V, 0, sizeV, stream);
      cudaDeviceSynchronize();
      
      // With value-based truncation, `cutensornetTensorSVD` can potentially update the shared extent in descTensorU/V.
      // We here restore descTensorU/V to the original problem.
      HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorU) );
      HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorV) );
      HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides, modesU.data(), typeData, &descTensorU) );
      HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides, modesV.data(), typeData, &descTensorV) );

      timer.start();
      HANDLE_ERROR( cutensornetTensorSVD(handle, 
                        descTensorIn, D_T, 
                        descTensorU, D_U, 
                        D_S, 
                        descTensorV, D_V, 
                        svdConfig, 
                        svdInfo,
                        workDesc,
                        stream) );
      // Synchronize and measure timing
      auto time = timer.seconds();
      minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
   }

   printf("Performing SVD\n");

   HANDLE_CUDA_ERROR( cudaMemcpyAsync(U, D_U, sizeU, cudaMemcpyDeviceToHost) );
   HANDLE_CUDA_ERROR( cudaMemcpyAsync(S, D_S, sizeS, cudaMemcpyDeviceToHost) );
   HANDLE_CUDA_ERROR( cudaMemcpyAsync(V, D_V, sizeV, cudaMemcpyDeviceToHost) );

   // Sphinx: #8
   /*************************************
   * Query runtime truncation information
   **************************************/

   double discardedWeight{0};
   int64_t reducedExtent{0};
   cutensornetGesvdjStatus_t gesvdjStatus;
   cudaDeviceSynchronize(); // device synchronization.
   HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &discardedWeight, sizeof(discardedWeight)) );
   HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, &reducedExtent, sizeof(reducedExtent)) );
   HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS, &gesvdjStatus, sizeof(gesvdjStatus)) );

   printf("elapsed time: %.2f ms\n", minTimeCUTENSOR * 1000.f);
   printf("GESVDJ residual: %.4f, runtime sweeps = %d\n", gesvdjStatus.residual, gesvdjStatus.sweeps);
   printf("reduced extent found at runtime: %lu\n", reducedExtent);
   printf("discarded weight: %.2f\n", discardedWeight);

   // Sphinx: #9
   /***************
   * Free resources
   ****************/
   
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorIn) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorU) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorV) );
   HANDLE_ERROR( cutensornetDestroyTensorSVDConfig(svdConfig) );
   HANDLE_ERROR( cutensornetDestroyTensorSVDInfo(svdInfo) );
   HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );
   HANDLE_ERROR( cutensornetDestroy(handle) );

   if (T) free(T);
   if (U) free(U);
   if (S) free(S);
   if (V) free(V);
   if (D_T) cudaFree(D_T);
   if (D_U) cudaFree(D_U);
   if (D_S) cudaFree(D_S);
   if (D_V) cudaFree(D_V);
   if (devWork) cudaFree(devWork);
   if (hostWork) free(hostWork);

   printf("Free resource and exit.\n");

   return 0;
}
