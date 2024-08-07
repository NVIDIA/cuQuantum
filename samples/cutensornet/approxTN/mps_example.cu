/*  
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 */  

#include <cmath>
#include <cstdlib>
#include <cassert>

#include <algorithm>
#include <complex>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cutensornet.h>

/****************************************************************
 *                   Basic Matrix Product State (MPS) Algorithm
 * 
 *  Input:
 *    1. A-J are MPS tensors
 *    2. XXXXX are rank-4 gate tensors:
 * 
 *     A---B---C---D---E---F---G---H---I---J          MPS tensors
 *     |   |   |   |   |   |   |   |   |   |
 *     XXXXX   XXXXX   XXXXX   XXXXX   XXXXX          gate cycle 0
 *     |   |   |   |   |   |   |   |   |   |
 *     |   XXXXX   XXXXX   XXXXX   XXXXX   |          gate cycle 1
 *     |   |   |   |   |   |   |   |   |   |
 *     XXXXX   XXXXX   XXXXX   XXXXX   XXXXX          gate cycle 2
 *     |   |   |   |   |   |   |   |   |   |
 *     |   XXXXX   XXXXX   XXXXX   XXXXX   |          gate cycle 3
 *     |   |   |   |   |   |   |   |   |   |
 *     XXXXX   XXXXX   XXXXX   XXXXX   XXXXX          gate cycle 4
 *     |   |   |   |   |   |   |   |   |   |
 *     |   XXXXX   XXXXX   XXXXX   XXXXX   |          gate cycle 5
 *     |   |   |   |   |   |   |   |   |   |
 *     XXXXX   XXXXX   XXXXX   XXXXX   XXXXX          gate cycle 6
 *     |   |   |   |   |   |   |   |   |   |
 *     |   XXXXX   XXXXX   XXXXX   XXXXX   |          gate cycle 7
 *     |   |   |   |   |   |   |   |   |   |
 * 
 * 
 *  Output:
 *    1. maximal virtual extent of the bonds (===) is `maxVirtualExtent` (set by user).
 * 
 *     A===B===C===D===E===F===G===H===I===J          MPS tensors
 *     |   |   |   |   |   |   |   |   |   |   
 * 
 *  
 *  Algorithm:
 *    Iterative over the gate cycles, within each cycle, perform gate split operation below for all relevant tensors
 *              ---A---B----          
 *                 |   |       GateSplit     ---A===B---   
 *                 XXXXX       ------->         |   |
 *                 |   | 
******************************************************************/

// Sphinx: #1
#define HANDLE_ERROR(x)                                           \
{ const auto err = x;                                             \
if( err != CUTENSORNET_STATUS_SUCCESS )                           \
{ std::cout << "Error: " <<  cutensornetGetErrorString(err) << " in line " << __LINE__ << std::endl; return err;} \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{  const auto err = x;                                            \
   if( err != cudaSuccess )                                       \
   { std::cout << "Error: " <<  cudaGetErrorString(err) << " in line " << __LINE__ << std::endl; return err; } \
};

// Sphinx: #2
class MPSHelper
{
   public:
      /**
       * \brief Construct an MPSHelper object for gate splitting algorithm.
       *        i       j       k
       *     -------A-------B-------                      i        j        k
       *           p|       |q            ------->     -------A`-------B`-------
       *            GGGGGGGGG                                r|        |s
       *           r|       |s
       * \param[in] numSites The number of sites in the MPS
       * \param[in] physExtent The extent for the physical mode where the gate tensors are acted on. 
       * \param[in] maxVirtualExtent The maximal extent allowed for the virtual mode shared between adjacent MPS tensors. 
       * \param[in] initialVirtualExtents A vector of size \p numSites-1 where the ith element denotes the extent of the shared mode for site i and site i+1 in the beginning of the simulation.
       * \param[in] typeData The data type for all tensors and gates
       * \param[in] typeCompute The compute type for all gate splitting process
       */
      MPSHelper(int32_t numSites, 
                int64_t physExtent,
                int64_t maxVirtualExtent,
                const std::vector<int64_t>& initialVirtualExtents,
                cudaDataType_t typeData, 
                cutensornetComputeType_t typeCompute);
      
      /**
       * \brief Initialize the MPS metadata and cutensornet library.
       */
      cutensornetStatus_t initialize();

      /**
       * \brief Compute the maximal number of elements for each site.
       */
      std::vector<size_t> getMaxTensorElements() const;

      /**
       * \brief Update the SVD truncation setting.
       * \param[in] absCutoff The cutoff value for absolute singular value truncation.
       * \param[in] relCutoff The cutoff value for relative singular value truncation.
       * \param[in] renorm The option for renormalization of the truncated singular values.
       * \param[in] partition The option for partitioning of the singular values. 
       */
      cutensornetStatus_t setSVDConfig(double absCutoff, 
                                       double relCutoff, 
                                       cutensornetTensorSVDNormalization_t renorm,
                                       cutensornetTensorSVDPartition_t partition);

      /**
       * \brief Update the algorithm to use for the gating process.
       * \param[in] gateAlgo The gate algorithm to use for MPS simulation.
       */
      void setGateAlgorithm(cutensornetGateSplitAlgo_t gateAlgo) {gateAlgo_ = gateAlgo;}

      /**
       * \brief Compute the maximal workspace needed for MPS gating algorithm.
       * \param[out] workspaceSize The required workspace size on the device. 
       */
      cutensornetStatus_t computeMaxWorkspaceSizes(int64_t* workspaceSize);

      /**
       * \brief Compute the maximal workspace needed for MPS gating algorithm.
       * \param[in] work Pointer to the allocated workspace.
       * \param[in] workspaceSize The required workspace size on the device. 
       */
      cutensornetStatus_t setWorkspace(void* work, int64_t workspaceSize);

      /**
       * \brief In-place execution of the apply gate algorithm on \p siteA and \p siteB.
       * \param[in] siteA The first site where the gate is applied to.
       * \param[in] siteB The second site where the gate is applied to. Must be adjacent to \p siteA.
       * \param[in,out] dataInA The data for the MPS tensor at \p siteA. The input will be overwritten with output mps tensor data.
       * \param[in,out] dataInB The data for the MPS tensor at \p siteB. The input will be overwritten with output mps tensor data.
       * \param[in] dataInG The input data for the gate tensor. 
       * \param[in] verbose Whether to print out the runtime information regarding truncation. 
       * \param[in] stream The CUDA stream on which the computation is performed.
       */
      cutensornetStatus_t applyGate(uint32_t siteA, 
                                    uint32_t siteB, 
                                    void* dataInA, 
                                    void* dataInB, 
                                    const void* dataInG, 
                                    bool verbose,
                                    cudaStream_t stream);
      
      /**
       * \brief Free all the tensor descriptors in mpsHelper.
       */
      ~MPSHelper()
      {
         if (inited_)
         {
            for (auto& descTensor: descTensors_)
            {
               cutensornetDestroyTensorDescriptor(descTensor);
            }
            cutensornetDestroy(handle_);
            cutensornetDestroyWorkspaceDescriptor(workDesc_);
         }
         if (svdConfig_ != nullptr)
         {
            cutensornetDestroyTensorSVDConfig(svdConfig_);
         }
         if (svdInfo_ != nullptr)
         {
            cutensornetDestroyTensorSVDInfo(svdInfo_);
         }
      }

   private:
      int32_t numSites_; ///< Number of sites in the MPS
      int64_t physExtent_; ///< Extent for the physical index 
      int64_t maxVirtualExtent_{0}; ///< The maximal extent allowed for the virtual dimension
      cudaDataType_t typeData_; 
      cutensornetComputeType_t typeCompute_;
      
      bool inited_{false};
      std::vector<int32_t> physModes_; ///< A vector of length \p numSites_ storing the physical mode of each site.
      std::vector<int32_t> virtualModes_; ///< A vector of length \p numSites_+1; For site i, virtualModes_[i] and virtualModes_[i+1] represents the left and right virtual mode.
      std::vector<int64_t> extentsPerSite_; ///< A vector of length \p numSites_+1; For site i, extentsPerSite_[i] and extentsPerSite_[i+1] represents the left and right virtual extent. 

      cutensornetHandle_t handle_{nullptr};
      std::vector<cutensornetTensorDescriptor_t> descTensors_; /// A vector of length \p numSites_ storing the cutensornetTensorDescriptor_t for each site
      cutensornetWorkspaceDescriptor_t workDesc_{nullptr};
      cutensornetTensorSVDConfig_t svdConfig_{nullptr};
      cutensornetTensorSVDInfo_t svdInfo_{nullptr};
      cutensornetGateSplitAlgo_t gateAlgo_{CUTENSORNET_GATE_SPLIT_ALGO_DIRECT};
      int32_t nextMode_{0}; /// The next mode label to use for labelling site tensors and gates.
};

// Sphinx: #3
MPSHelper::MPSHelper(int32_t numSites, 
                     int64_t physExtent,
                     int64_t maxVirtualExtent,
                     const std::vector<int64_t>& initialVirtualExtents,
                     cudaDataType_t typeData, 
                     cutensornetComputeType_t typeCompute)
                     : numSites_(numSites), 
                       physExtent_(physExtent),
                       typeData_(typeData),
                       typeCompute_(typeCompute)
{
   // initialize vectors to store the modes and extents for physical and virtual bond
   for (int32_t i=0; i<numSites+1; i++)
   {
      int64_t e = (i == 0 || i==numSites) ? 1: initialVirtualExtents.at(i-1);
      extentsPerSite_.push_back(e);
      virtualModes_.push_back(nextMode_++);
      if (i != numSites)
      {
         physModes_.push_back(nextMode_++);
      }
   }
   int64_t untruncatedMaxExtent = (int64_t)std::pow(physExtent_, numSites_/2); // maximal virtual extent for the MPS
   maxVirtualExtent_ = maxVirtualExtent == 0? untruncatedMaxExtent: std::min(maxVirtualExtent, untruncatedMaxExtent);
}

// Sphinx: #4
cutensornetStatus_t MPSHelper::initialize()
{
   // initialize workDesc, svdInfo and input tensor descriptors
   assert (! inited_);
   HANDLE_ERROR( cutensornetCreate(&handle_) );
   HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle_, &workDesc_) );
   for (int32_t i=0; i<numSites_; i++)
   {
      cutensornetTensorDescriptor_t descTensor;
      const int64_t extents[]{extentsPerSite_[i], physExtent_, extentsPerSite_[i+1]};
      const int32_t modes[]{virtualModes_[i], physModes_[i], virtualModes_[i+1]};
      HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle_,
                                                      /*numModes=*/3,
                                                      extents,
                                                      /*strides=*/nullptr, // fortran layout
                                                      modes,
                                                      typeData_,
                                                      &descTensor) );
      descTensors_.push_back(descTensor);
   }
   HANDLE_ERROR( cutensornetCreateTensorSVDConfig(handle_, &svdConfig_) );
   HANDLE_ERROR( cutensornetCreateTensorSVDInfo(handle_, &svdInfo_) );
   inited_ = true;
   
   return CUTENSORNET_STATUS_SUCCESS;
}

// Sphinx: #5
/*************************************
* Compute maximal sizes for allocation
**************************************/
std::vector<size_t> MPSHelper::getMaxTensorElements() const
{
   // compute the maximal tensor sizes for all sites during MPS simulation
   std::vector<size_t> maxTensorElements(numSites_);
   int64_t maxLeftExtent = 1;
   for (int32_t i=0; i<numSites_; i++)
   {
      int64_t maxRightExtent = std::min({(int64_t)std::pow(physExtent_, i+1),
                                         (int64_t)std::pow(physExtent_, numSites_-i-1),
                                         maxVirtualExtent_});
      maxTensorElements[i] = physExtent_ * maxLeftExtent * maxRightExtent;
      maxLeftExtent = maxRightExtent;
   }
   return std::move(maxTensorElements);
}

// Sphinx: #6
/********************************
* Setup SVD truncation parameters
*********************************/
cutensornetStatus_t MPSHelper::setSVDConfig(double absCutoff, 
                                            double relCutoff, 
                                            cutensornetTensorSVDNormalization_t renorm,
                                            cutensornetTensorSVDPartition_t partition)
{
   HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle_, 
                                          svdConfig_, 
                                          CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF, 
                                          &absCutoff, 
                                          sizeof(absCutoff)) );
   
   HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle_, 
                                          svdConfig_, 
                                          CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF, 
                                          &relCutoff, 
                                          sizeof(relCutoff)) );

   HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle_, 
                                          svdConfig_, 
                                          CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION, 
                                          &renorm, 
                                          sizeof(renorm)) );
   
   if (partition != CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL)
   {
      std::cout << "This helper class currently only supports \"parititon=CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL\"" << std::endl;
      exit(-1);
   }
   HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle_, 
                                          svdConfig_, 
                                          CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION, 
                                          &partition, 
                                          sizeof(partition)) );
   return CUTENSORNET_STATUS_SUCCESS;
}

// Sphinx: #7
/*****************************
* Query maximal workspace size
******************************/
cutensornetStatus_t MPSHelper::computeMaxWorkspaceSizes(int64_t* workspaceSize)
{
   cutensornetTensorDescriptor_t descTensorInA;
   cutensornetTensorDescriptor_t descTensorInB;
   cutensornetTensorDescriptor_t descTensorInG;
   cutensornetTensorDescriptor_t descTensorOutA;
   cutensornetTensorDescriptor_t descTensorOutB;

   const int64_t maxExtentsAB[]{maxVirtualExtent_, physExtent_, maxVirtualExtent_};
   const int64_t extentsInG[]{physExtent_, physExtent_, physExtent_, physExtent_};

   const int32_t modesInA[] = {'i','p','j'};
   const int32_t modesInB[] = {'j','q','k'};
   const int32_t modesInG[] = {'p', 'q', 'r', 's'};
   const int32_t modesOutA[] = {'i','r','j'};
   const int32_t modesOutB[] = {'j','s','k'};

   // create tensor descriptors for largest gate split process
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle_,
                                                   /*numModes=*/3,
                                                   maxExtentsAB,
                                                   /*strides=*/nullptr, // fortran layout
                                                   modesInA,
                                                   typeData_,
                                                   &descTensorInA) );
      
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle_,
                                                   /*numModes=*/3,
                                                   maxExtentsAB,
                                                   /*strides=*/nullptr, // fortran layout
                                                   modesInB,
                                                   typeData_,
                                                   &descTensorInB) );
      
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle_,
                                                   /*numModes=*/4,
                                                   extentsInG,
                                                   /*strides=*/nullptr, // fortran layout
                                                   modesInG,
                                                   typeData_,
                                                   &descTensorInG) );
      
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle_,
                                                   /*numModes=*/3,
                                                   maxExtentsAB,
                                                   /*strides=*/nullptr, // fortran layout
                                                   modesOutA,
                                                   typeData_,
                                                   &descTensorOutA) );
      
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle_,
                                                   /*numModes=*/3,
                                                   maxExtentsAB,
                                                   /*strides=*/nullptr, // fortran layout
                                                   modesOutB,
                                                   typeData_,
                                                   &descTensorOutB) );
   // query workspace size
   HANDLE_ERROR( cutensornetWorkspaceComputeGateSplitSizes(handle_, 
                                                           descTensorInA, 
                                                           descTensorInB, 
                                                           descTensorInG,
                                                           descTensorOutA,
                                                           descTensorOutB,
                                                           gateAlgo_,
                                                           svdConfig_,
                                                           typeCompute_,
                                                           workDesc_) );
   
   HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle_,
                                                   workDesc_,
                                                   CUTENSORNET_WORKSIZE_PREF_MIN,
                                                   CUTENSORNET_MEMSPACE_DEVICE,
                                                   CUTENSORNET_WORKSPACE_SCRATCH,
                                                   workspaceSize) );
   // free the tensor descriptors
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorInA) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorInB) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorInG) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorOutA) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorOutB) );
   return CUTENSORNET_STATUS_SUCCESS;
}

// Sphinx: #8
cutensornetStatus_t MPSHelper::setWorkspace(void* work, int64_t workspaceSize)
{
   HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle_,
                                               workDesc_,
                                               CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH,
                                               work,
                                               workspaceSize) );
   return CUTENSORNET_STATUS_SUCCESS;
}

// Sphinx: #9
cutensornetStatus_t MPSHelper::applyGate(uint32_t siteA, 
                                         uint32_t siteB, 
                                         void* dataInA, 
                                         void* dataInB, 
                                         const void* dataInG,
                                         bool verbose,
                                         cudaStream_t stream)
{
   if ((siteB - siteA) != 1)
   {
      std::cout<< "SiteB must be the right site of siteA" << std::endl;
      return CUTENSORNET_STATUS_INVALID_VALUE;
   }
   if (siteB >= numSites_)
   {
      std::cout<< "Site index can not exceed maximal number of sites" << std::endl;
      return CUTENSORNET_STATUS_INVALID_VALUE;
   }

   auto descTensorInA = descTensors_[siteA];
   auto descTensorInB = descTensors_[siteB];
   
   cutensornetTensorDescriptor_t descTensorInG;

   /*********************************
   * Create output tensor descriptors
   **********************************/
   int32_t physModeInA = physModes_[siteA];
   int32_t physModeInB = physModes_[siteB];
   int32_t physModeOutA = nextMode_++;
   int32_t physModeOutB = nextMode_++;
   const int32_t modesG[]{physModeInA, physModeInB, physModeOutA, physModeOutB};
   const int64_t extentG[]{physExtent_, physExtent_, physExtent_, physExtent_};
   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle_,
                                                   /*numModes=*/4,
                                                   extentG,
                                                   /*strides=*/nullptr, // fortran layout
                                                   modesG,
                                                   typeData_,
                                                   &descTensorInG) );

   int64_t leftExtentA = extentsPerSite_[siteA];
   int64_t extentABIn = extentsPerSite_[siteA+1];
   int64_t rightExtentB = extentsPerSite_[siteA+2];
   // Compute the expected shared extent of output tensor A and B.
   int64_t combinedExtentLeft = std::min(leftExtentA, extentABIn*physExtent_) * physExtent_;
   int64_t combinedExtentRight = std::min(rightExtentB, extentABIn*physExtent_) * physExtent_;
   int64_t extentABOut = std::min({combinedExtentLeft, combinedExtentRight, maxVirtualExtent_});

   cutensornetTensorDescriptor_t descTensorOutA;
   cutensornetTensorDescriptor_t descTensorOutB;
   const int32_t modesOutA[]{virtualModes_[siteA], physModeOutA, virtualModes_[siteA+1]};
   const int32_t modesOutB[]{virtualModes_[siteB], physModeOutB, virtualModes_[siteB+1]};
   const int64_t extentOutA[]{leftExtentA, physExtent_, extentABOut};
   const int64_t extentOutB[]{extentABOut, physExtent_, rightExtentB};

   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle_,
                                                   /*numModes=*/3,
                                                   extentOutA,
                                                   /*strides=*/nullptr, // fortran layout
                                                   modesOutA,
                                                   typeData_,
                                                   &descTensorOutA) );

   HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle_,
                                                   /*numModes=*/3,
                                                   extentOutB,
                                                   /*strides=*/nullptr, // fortran layout
                                                   modesOutB,
                                                   typeData_,
                                                   &descTensorOutB) );

   /**********
   * Execution
   ***********/
   HANDLE_ERROR( cutensornetGateSplit(handle_,
                                      descTensorInA, dataInA,
                                      descTensorInB, dataInB,
                                      descTensorInG, dataInG,
                                      descTensorOutA, dataInA, // overwrite in place
                                      /*s=*/nullptr, // we partition s equally onto A and B, therefore s is not needed
                                      descTensorOutB, dataInB, // overwrite in place
                                      gateAlgo_, svdConfig_, typeCompute_,
                                      svdInfo_, workDesc_, stream) );
   
   /**************************
   * Query runtime information
   ***************************/
   if (verbose)
   {
      int64_t fullExtent;
      int64_t reducedExtent;
      double discardedWeight;
      HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle_, svdInfo_, CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT, &fullExtent, sizeof(fullExtent)) );
      HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle_, svdInfo_, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, &reducedExtent, sizeof(reducedExtent)) );
      HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle_, svdInfo_, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &discardedWeight, sizeof(discardedWeight)) );
      std::cout << "virtual bond truncated from " << fullExtent << " to " << reducedExtent << " with a discarded weight " << discardedWeight << std::endl;
   }
   
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorInA) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorInB) );
   HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorInG) );

   // update pointer to the output tensor descriptor and the output shared extent
   physModes_[siteA] = physModeOutA;
   physModes_[siteB] = physModeOutB;
   descTensors_[siteA] = descTensorOutA;
   descTensors_[siteB] = descTensorOutB;

   int32_t numModes = 3;
   std::vector<int64_t> extentAOut(numModes);
   HANDLE_ERROR( cutensornetGetTensorDetails(handle_, descTensorOutA, &numModes, nullptr, nullptr, extentAOut.data(), nullptr) );
   // update the shared extent of output A and B which can potentially get reduced if absCutoff and relCutoff is non-zero.
   extentsPerSite_[siteA+1] = extentAOut[2]; // mode label order is always (left_virtual, physical, right_virtual)
   return CUTENSORNET_STATUS_SUCCESS;
}

// Sphinx: #10
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
   
   // Sphinx: #11
   /***********************************
   * Step 1: basic MPS setup
   ************************************/

   // setup the simulation setting for the MPS
   typedef std::complex<double> complexType;
   cudaDataType_t typeData = CUDA_C_64F;
   cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_64F;
   int32_t numSites = 16;
   int64_t physExtent = 2;
   int64_t maxVirtualExtent = 12;
   const std::vector<int64_t> initialVirtualExtents(numSites-1, 1);  // starting MPS with shared extent of 1;

   // initialize an MPSHelper to dynamically update tensor metadats   
   MPSHelper mpsHelper(numSites, physExtent, maxVirtualExtent, initialVirtualExtents, typeData, typeCompute);
   HANDLE_ERROR( mpsHelper.initialize() );

   // Sphinx: #12
   /***********************************
   * Step 2: data allocation 
   ************************************/

   // query largest tensor sizes for the MPS
   const std::vector<size_t> maxElementsPerSite = mpsHelper.getMaxTensorElements();
   std::vector<void*> tensors_h;
   std::vector<void*> tensors_d;
   for (int32_t i=0; i<numSites; i++)
   {
      size_t maxSize = sizeof(complexType) * maxElementsPerSite.at(i);
      void* data_h = malloc(maxSize);
      memset(data_h, 0, maxSize);
      // initialize state to |0000..0000>
      *(complexType*)(data_h) = complexType(1,0);  
      void* data_d;
      HANDLE_CUDA_ERROR( cudaMalloc(&data_d, maxSize) );
      // data transfer from host to device
      HANDLE_CUDA_ERROR( cudaMemcpy(data_d, data_h, maxSize, cudaMemcpyHostToDevice) );
      tensors_h.push_back(data_h);
      tensors_d.push_back(data_d);
   }

   // initialize 4 random gate tensors on host and copy them to device
   const int32_t numRandomGates = 4;
   const int64_t numGateElements = physExtent * physExtent * physExtent * physExtent;  // shape (2, 2, 2, 2)
   size_t gateSize = sizeof(complexType) * numGateElements;
   complexType* gates_h[numRandomGates];
   void* gates_d[numRandomGates];
   
   for (int i=0; i<numRandomGates; i++)
   {
      gates_h[i] = (complexType*) malloc(gateSize);
      HANDLE_CUDA_ERROR( cudaMalloc((void**) &gates_d[i], gateSize) );
      for (int j=0; j<numGateElements; j++)
      {
         gates_h[i][j] = complexType(((float) rand())/RAND_MAX, ((float) rand())/RAND_MAX);
      }
      HANDLE_CUDA_ERROR( cudaMemcpy(gates_d[i], gates_h[i], gateSize, cudaMemcpyHostToDevice) );
   }
   
   // Sphinx: #13
   /*****************************************
   * Step 3: setup options for gate operation
   ******************************************/

   double absCutoff = 1e-2;
   double relCutoff = 1e-2;
   cutensornetTensorSVDNormalization_t renorm = CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2; // renormalize the L2 norm of truncated singular values to 1. 
   cutensornetTensorSVDPartition_t partition = CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL; // equally partition the singular values onto U and V;
   HANDLE_ERROR( mpsHelper.setSVDConfig(absCutoff, relCutoff, renorm, partition));

   cutensornetGateSplitAlgo_t gateAlgo = CUTENSORNET_GATE_SPLIT_ALGO_REDUCED;
   mpsHelper.setGateAlgorithm(gateAlgo);

   // Sphinx: #14
   /********************************************
   * Step 4: workspace size query and allocation
   *********************************************/

   int64_t workspaceSize;
   HANDLE_ERROR( mpsHelper.computeMaxWorkspaceSizes(&workspaceSize) );

   void *work = nullptr;
   std::cout << "Maximal workspace size required: " << workspaceSize << std::endl;
   HANDLE_CUDA_ERROR( cudaMalloc(&work, workspaceSize) );

   HANDLE_ERROR( mpsHelper.setWorkspace(work, workspaceSize));
   
   // Sphinx: #15
   /***********************************
   * Step 5: execution
   ************************************/

   cudaStream_t stream;
   HANDLE_CUDA_ERROR( cudaStreamCreate(&stream) );
   uint32_t numLayers = 10; // 10 layers of gate
   for (uint32_t i=0; i<numLayers; i++)
   {
      uint32_t start_site = i % 2;
      std::cout << "Cycle " << i << ":" << std::endl;
      bool verbose = (i == numLayers - 1);
      for (uint32_t j=start_site; j<numSites-1; j=j+2)
      {
         uint32_t gateIdx = rand() % numRandomGates; // pick a random gate tensor
         std::cout << "apply gate " << gateIdx << " on " << j << " and " << j+1<< std::endl;
         void *dataA = tensors_d[j];
         void *dataB = tensors_d[j+1];
         void *dataG = gates_d[gateIdx];
         HANDLE_ERROR( mpsHelper.applyGate(j, j+1, dataA, dataB, dataG, verbose, stream) );
      }
   }

   HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );

   // Sphinx: #16
   /***********************************
   * Step 6: free resources
   ************************************/
   
   std::cout << "Free all resources" << std::endl;

   for (int i=0; i<numRandomGates; i++)
   {
      free(gates_h[i]);
      HANDLE_CUDA_ERROR( cudaFree(gates_d[i]) );
   }

   for (int32_t i=0; i<numSites; i++)
   {
      free(tensors_h.at(i));
      HANDLE_CUDA_ERROR( cudaFree(tensors_d.at(i)) );
   }

   HANDLE_CUDA_ERROR( cudaFree(work) );
   // The MPSHelper destructor will free all internal resources when out of scope
   return 0;   
}
