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
   /************************************************************************************
   * Gate Split: A_{i,j,k,l} B_{k,o,p,q} G_{m,n,l,o}-> A'_{i,j,x,m} S_{x} B'_{x,n,p,q}  
   *************************************************************************************/
   typedef float floatType;
   cudaDataType_t typeData = CUDA_R_32F;
   cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F;

   // Create vector of modes
   std::vector<int32_t> modesAIn{'i','j','k','l'};
   std::vector<int32_t> modesBIn{'k','o','p','q'};
   std::vector<int32_t> modesGIn{'m','n','l','o'}; // input, G is the gate operator

   std::vector<int32_t> modesAOut{'i','j','x','m'}; 
   std::vector<int32_t> modesBOut{'x','n','p','q'}; // SVD output

   // Extents
   std::unordered_map<int32_t, int64_t> extent;
   extent['i'] = 16;
   extent['j'] = 16;
   extent['k'] = 16;
   extent['l'] = 2;
   extent['m'] = 2;
   extent['n'] = 2;
   extent['o'] = 2;
   extent['p'] = 16;
   extent['q'] = 16;
   
   const int64_t maxExtent = 16; //truncate to a maximal extent of 16
   extent['x'] = maxExtent;

   // Create a vector of extents for each tensor
   std::vector<int64_t> extentAIn;
   for (auto mode : modesAIn)
      extentAIn.push_back(extent[mode]);
   std::vector<int64_t> extentBIn;
   for (auto mode : modesBIn)
      extentBIn.push_back(extent[mode]);
   std::vector<int64_t> extentGIn;
   for (auto mode : modesGIn)
      extentGIn.push_back(extent[mode]);
   std::vector<int64_t> extentAOut;
   for (auto mode : modesAOut)
      extentAOut.push_back(extent[mode]);
   std::vector<int64_t> extentBOut;
   for (auto mode : modesBOut)
      extentBOut.push_back(extent[mode]);
   
   // Sphinx: #3
   /***********************************
   * Allocating data on host and device
   ************************************/

   size_t elementsAIn = 1;
   for (auto mode : modesAIn)
      elementsAIn *= extent[mode];
   size_t elementsBIn = 1;
   for (auto mode : modesBIn)
      elementsBIn *= extent[mode];
   size_t elementsGIn = 1;
   for (auto mode : modesGIn)
      elementsGIn *= extent[mode];
   size_t elementsAOut = 1;
   for (auto mode : modesAOut)
      elementsAOut *= extent[mode];
   size_t elementsBOut = 1;
   for (auto mode : modesBOut)
      elementsBOut *= extent[mode];

   size_t sizeAIn = sizeof(floatType) * elementsAIn;
   size_t sizeBIn = sizeof(floatType) * elementsBIn;
   size_t sizeGIn = sizeof(floatType) * elementsGIn;
   size_t sizeAOut = sizeof(floatType) * elementsAOut;
   size_t sizeBOut = sizeof(floatType) * elementsBOut;
   size_t sizeS = sizeof(floatType) * extent['x'];
   
   printf("Total memory: %.2f GiB\n", (sizeAIn + sizeBIn + sizeGIn + sizeAOut + sizeBOut + sizeS)/1024./1024./1024);

   void* D_AIn;
   void* D_BIn;
   void* D_GIn;
   void* D_AOut;
   void* D_BOut;
   void* D_S;
   
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_AIn, sizeAIn) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_BIn, sizeBIn) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_GIn, sizeGIn) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_AOut, sizeAOut) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_BOut, sizeBOut) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_S, sizeS) );

   floatType *AIn = (floatType*) malloc(sizeAIn);
   floatType *BIn = (floatType*) malloc(sizeBIn);
   floatType *GIn = (floatType*) malloc(sizeGIn);
   
   if (AIn == NULL || BIn == NULL || GIn == NULL)
   {
      printf("Error: Host allocation of tensor data.\n");
      return -1;
   }

   /**********************
   * Initialize input data
   ***********************/
   for (uint64_t i = 0; i < elementsAIn; i++)
      AIn[i] = ((floatType) rand())/RAND_MAX;
   for (uint64_t i = 0; i < elementsBIn; i++)
      BIn[i] = ((floatType) rand())/RAND_MAX;
   for (uint64_t i = 0; i < elementsGIn; i++)
      GIn[i] = ((floatType) rand())/RAND_MAX;
   
   HANDLE_CUDA_ERROR( cudaMemcpy(D_AIn, AIn, sizeAIn, cudaMemcpyHostToDevice) );
   HANDLE_CUDA_ERROR( cudaMemcpy(D_BIn, BIn, sizeBIn, cudaMemcpyHostToDevice) );
   HANDLE_CUDA_ERROR( cudaMemcpy(D_GIn, GIn, sizeGIn, cudaMemcpyHostToDevice) );

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

   cutensornetTensorDescriptor_t descTensorAIn;
   cutensornetTensorDescriptor_t descTensorBIn;
   cutensornetTensorDescriptor_t descTensorGIn;
   cutensornetTensorDescriptor_t descTensorAOut;
   cutensornetTensorDescriptor_t descTensorBOut;

   const int32_t numModesAIn = modesAIn.size();
   const int32_t numModesBIn = modesBIn.size();
   const int32_t numModesGIn = modesGIn.size();
   const int32_t numModesAOut = modesAOut.size();
   const int32_t numModesBOut = modesBOut.size();

   const int64_t* strides = NULL; // assuming fortran layout for all tensors
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesAIn, extentAIn.data(), strides, modesAIn.data(), typeData, &descTensorAIn) );
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesBIn, extentBIn.data(), strides, modesBIn.data(), typeData, &descTensorBIn) );
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesGIn, extentGIn.data(), strides, modesGIn.data(), typeData, &descTensorGIn) );
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesAOut, extentAOut.data(), strides, modesAOut.data(), typeData, &descTensorAOut) );
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesBOut, extentBOut.data(), strides, modesBOut.data(), typeData, &descTensorBOut) );

   printf("Initialize the cuTensorNet library and create tensor descriptors.\n");

   // Sphinx: #5
   /**************************************************
   * Setup gate split truncation options and algorithm
   ***************************************************/

   cutensornetTensorSVDConfig_t svdConfig;
   HANDLE_ERROR( cutensornetCreateTensorSVDConfig(handle, &svdConfig) );
   double absCutoff = 1e-2;
   HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, 
                                          svdConfig, 
                                          CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF, 
                                          &absCutoff, 
                                          sizeof(absCutoff)) );
   double relCutoff = 1e-2;
   HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, 
                                          svdConfig, 
                                          CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF, 
                                          &relCutoff, 
                                          sizeof(relCutoff)) );
   
   cutensornetGateSplitAlgo_t gateAlgo = CUTENSORNET_GATE_SPLIT_ALGO_REDUCED; 
   /********************************************************
   * Create SVDInfo to record runtime SVD truncation details
   *********************************************************/

   cutensornetTensorSVDInfo_t svdInfo; 
   HANDLE_ERROR( cutensornetCreateTensorSVDInfo(handle, &svdInfo)) ;

   // Sphinx: #6
   /**************************************
   * Query and allocate required workspace
   ***************************************/

   cutensornetWorkspaceDescriptor_t workDesc;
   HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle, &workDesc) );
   
   HANDLE_ERROR( cutensornetWorkspaceComputeGateSplitSizes(handle, 
                                                           descTensorAIn, descTensorBIn, descTensorGIn, 
                                                           descTensorAOut, descTensorBOut, 
                                                           gateAlgo, 
                                                           svdConfig, typeCompute, 
                                                           workDesc) );
   int64_t requiredWorkspaceSize = 0;
   HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle,
                                                   workDesc,
                                                   CUTENSORNET_WORKSIZE_PREF_MIN,
                                                   CUTENSORNET_MEMSPACE_DEVICE,
                                                   CUTENSORNET_WORKSPACE_SCRATCH,
                                                   &requiredWorkspaceSize) );
   void *work = nullptr;
   HANDLE_CUDA_ERROR( cudaMalloc(&work, requiredWorkspaceSize) );

   HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle,
                                               workDesc,
                                               CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH,
                                               work,
                                               requiredWorkspaceSize) );

   printf("Allocate workspace.\n");
   
   // Sphinx: #7
   /**********************
   * Execution
   **********************/

   GPUTimer timer{stream};
   double minTimeCUTENSOR = 1e100;
   const int numRuns = 3; // to get stable perf results
   for (int i=0; i < numRuns; ++i)
   {  
      // restore output
      cudaMemsetAsync(D_AOut, 0, sizeAOut, stream);
      cudaMemsetAsync(D_S, 0, sizeS, stream);
      cudaMemsetAsync(D_BOut, 0, sizeBOut, stream);

      // With value-based truncation, `cutensornetGateSplit` can potentially update the shared extent in descTensorA/BOut.
      // We here restore descTensorA/BOut to the original problem.
      HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorAOut) );
      HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorBOut) );
      HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesAOut, extentAOut.data(), strides, modesAOut.data(), typeData, &descTensorAOut) );
      HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesBOut, extentBOut.data(), strides, modesBOut.data(), typeData, &descTensorBOut) );

      cudaDeviceSynchronize();
      timer.start();
      HANDLE_ERROR( cutensornetGateSplit(handle, 
                                         descTensorAIn, D_AIn,
                                         descTensorBIn, D_BIn,
                                         descTensorGIn, D_GIn,
                                         descTensorAOut, D_AOut,
                                         D_S,
                                         descTensorBOut, D_BOut,
                                         gateAlgo,
                                         svdConfig, typeCompute, svdInfo, 
                                         workDesc, stream) );
      // Synchronize and measure timing
      auto time = timer.seconds();
      minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
   }

   printf("Performing Gate Split\n");

   // Sphinx: #8
   /*************************************
   * Query runtime truncation information
   **************************************/

   double discardedWeight{0};
   int64_t reducedExtent{0};
   cudaDeviceSynchronize(); // device synchronization.
   HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &discardedWeight, sizeof(discardedWeight)) );
   HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, &reducedExtent, sizeof(reducedExtent)) );

   printf("elapsed time: %.2f ms\n", minTimeCUTENSOR * 1000.f);
   printf("reduced extent found at runtime: %lu\n", reducedExtent);
   printf("discarded weight: %.6f\n", discardedWeight);

   // Sphinx: #9
   /***************
   * Free resources
   ****************/

   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorAIn) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorBIn) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorGIn) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorAOut) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorBOut) );
   HANDLE_ERROR( cutensornetDestroyTensorSVDConfig(svdConfig) );
   HANDLE_ERROR( cutensornetDestroyTensorSVDInfo(svdInfo) );
   HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );
   HANDLE_ERROR( cutensornetDestroy(handle) );

   if (AIn) free(AIn);
   if (BIn) free(BIn);
   if (GIn) free(GIn);
   if (D_AIn) cudaFree(D_AIn);
   if (D_BIn) cudaFree(D_BIn);
   if (D_GIn) cudaFree(D_GIn);
   if (D_AOut) cudaFree(D_AOut);
   if (D_BOut) cudaFree(D_BOut);
   if (D_S) cudaFree(D_S);
   if (work) cudaFree(work);

   printf("Free resource and exit.\n");

   return 0;
}
