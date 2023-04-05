/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <custatevec.h>            // custatevec
#include <mpi.h>                   // MPI
#include <vector>                  // std::vector<>
#include <sys/types.h>             // struct timeval
#include <sys/time.h>              // gettimeofday()
#include <cstdio>                  // printf()
#include <cmath>                   // std::pow()


static char procname[256];

//
// err handler
//

bool hasFailed(bool res, const char** errmsg)
{
    if (res) return false;
    *errmsg = "res == false";
    return true;
}

bool hasFailed(int res, const char** errmsg)
{
    if (res == 0) return false;
    *errmsg = "res != 0";
    return true;
}

bool hasFailed(cudaError_t cuerr, const char** errmsg)
{
    if (cuerr == cudaSuccess) return false;
    *errmsg = cudaGetErrorName(cuerr);
    return true;
}

bool hasFailed(custatevecStatus_t status, const char** errmsg)
{
    if (status == CUSTATEVEC_STATUS_SUCCESS) return false;
    *errmsg = custatevecGetErrorName(status);
    return true;
}

template<class R>
void errChk(R res, const char* text, const char* file, unsigned long line)
{
    const char* errmsg = nullptr;
    if (hasFailed(res, &errmsg))
    {
        fprintf(stderr, "(%s) %s:%lu: %s %s\n", procname, file, line,
                errmsg, text);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
}

#define ERRCHK(s) errChk((s), #s, __FILE__, __LINE__)


// time measurement

double getTime()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1.e-6;
}


void runDistributedIndexBitSwaps(
        int rank, int size, int nGlobalIndexBits, int nLocalIndexBits,
        const std::vector<int2>& indexBitSwaps,
        const std::vector<int>& maskBitString, const std::vector<int>& maskOrdering)
{
    if (rank == 0)
    {
        printf("index bit swaps [ ");
        for (const auto& bitSwap : indexBitSwaps)
            printf("(%d,%d) ", bitSwap.x, bitSwap.y);
        printf("], mask bit string [ ");
        for (const auto& maskBit : maskBitString)
            printf("%d ", maskBit);
        printf("], mask ordering [ ");
        for (const auto& maskPos : maskOrdering)
            printf("%d ", maskPos);
        printf("]\n");
    }

    // Data type of the state vector, acceptable values are CUDA_C_32F and CUDA_C_64F.
    cudaDataType_t svDataType = CUDA_C_64F;
    // the number of index bits corresponding to sub state vectors accessible via GPUDirect P2P,
    // and it should be adjusted based on the number of GPUs/node, N, participating in the distributed
    // state vector (N=2^nP2PDeviceBits) that supports P2P data transfer
    int nP2PDeviceBits = 0;
    int nSubSVsP2P = 1 << nP2PDeviceBits;

    // use rank and size to map sub state vectors
    // this sample assigns one device to one rank and allocates one sub state vector on the assigned device
    // use the rank as the index of the sub state vector locally allocated in this process
    int orgSubSVIndex = rank;
    // the number of sub state vectors is identical to the number of processes
    int nSubSVs = size;

    // data size
    size_t dataSize = (svDataType == CUDA_C_64F) ? 16 : 8;

    // transfer workspace size
    size_t transferWorkspaceSize = size_t(1) << 26;

    //
    // allocate local sub state vector, stream and event
    //
    void* d_orgSubSV = nullptr;
    cudaStream_t localStream = nullptr;
    cudaEvent_t localEvent = nullptr;

    // bind the device to the process
    // this is based on the assumption of the global rank placement that the
    // processes are mapped to nodes in contiguous chunks (see the comment
    // below)
    int numDevices;
    ERRCHK(cudaGetDeviceCount(&numDevices));
    ERRCHK(numDevices > 0);
    if (nP2PDeviceBits > 0)
    {
        ERRCHK(numDevices >= nSubSVsP2P);
        ERRCHK(cudaSetDevice(rank % nSubSVsP2P));
    }
    else
    {
        ERRCHK(cudaSetDevice(rank % numDevices));
    }

    // allocate local resources
    size_t subSVSize = dataSize * (1LL << nLocalIndexBits);
    ERRCHK(cudaMalloc(&d_orgSubSV, subSVSize));
    ERRCHK(cudaMemset(d_orgSubSV, 0, subSVSize));
    ERRCHK(cudaStreamCreate(&localStream));
    // event should be created with the cudaEventInterprocess flag
    ERRCHK(cudaEventCreateWithFlags(&localEvent, cudaEventInterprocess | cudaEventDisableTiming));

    // create cuStateVec handle
    custatevecHandle_t handle;
    ERRCHK(custatevecCreate(&handle));

    //
    // create communicator
    //

    //
    // Using builtin MPI communicator
    //
    // cuStateVec provides builtin communicators for Open MPI and MPICH
    // By enabling a macro of USE_OPENMPI_COMMUNICATOR or USE_MPICH_COMMUNICATOR,
    // a bultin communicator is created.
    //
    // Builtin communicators dynamically resolve required MPI functions by using dlopen().
    // This sample directly links to libmpi.so, and all required MPI functions are loaded
    // to the application at application startup.  By specifying nullptr to the soname
    // argument to call custatevecSVSwapWorkerCreate(), all required functions
    // will be resolved from the functions loaded at application startup
    //
    // Please use one of the following macros to use builtin communicator
#define USE_OPENMPI_COMMUNICATOR (1)
// #define USE_MPICH_COMMUNICATOR (1)


#if defined(USE_OPENMPI_COMMUNICATOR)
    // use Open MPI communicator
    custatevecCommunicatorType_t communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI;
    const char* soname = nullptr;
#endif
#if defined(USE_MPICH_COMMUNICATOR)
    // use MPICH communicator
    custatevecCommunicatorType_t communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_MPICH;
    const char* soname = nullptr;
#endif

    //
    // Using external communicator
    //
    // External communicator is for libraries that are not ABI compatible with
    // Open MPI and MPICH.
    // It uses a shared library that wraps the MPI-library of preference.
    // soname should be the name to the shared library.
    //

// #define USE_EXTERNAL_COMMUNICATOR (1)

#if defined(USE_EXTERNAL_COMMUNICATOR)
    // External communicator
    // Used if the MPI library being used is not ABI-compatible with Open MPI and MPICH
    custatevecCommunicatorType_t communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL;
    // specify the name of the extension.
    // mpicomm.so is an example external communicator which is provided with this sample
    const char* soname = "./mpicomm.so";
#endif
    // create communicator
    custatevecCommunicatorDescriptor_t communicator = nullptr;
    ERRCHK(custatevecCommunicatorCreate(handle, &communicator, communicatorType, soname));

    //
    // create sv segment swap worker
    //
    custatevecSVSwapWorkerDescriptor_t svSegSwapWorker = nullptr;
    size_t extraWorkspaceSize = 0;
    size_t minTransferWorkspaceSize = 0;
    ERRCHK(custatevecSVSwapWorkerCreate(
                   handle, &svSegSwapWorker, communicator,
                   d_orgSubSV, orgSubSVIndex, localEvent, svDataType,
                   localStream, &extraWorkspaceSize, &minTransferWorkspaceSize));
    // set extra workspace
    void* d_extraWorkspace = nullptr;
    ERRCHK(cudaMalloc(&d_extraWorkspace, extraWorkspaceSize));
    ERRCHK(custatevecSVSwapWorkerSetExtraWorkspace(
                   handle, svSegSwapWorker, d_extraWorkspace, extraWorkspaceSize));
    // set transfer workspace
    // The size should be equal to or larger than minTransferWorkspaceSize
    // Depending on the systems, larger transfer workspace can improve the performance
    transferWorkspaceSize = std::max(minTransferWorkspaceSize, transferWorkspaceSize);
    void* d_transferWorkspace = nullptr;
    ERRCHK(cudaMalloc(&d_transferWorkspace, transferWorkspaceSize));
    ERRCHK(custatevecSVSwapWorkerSetTransferWorkspace(
                   handle, svSegSwapWorker, d_transferWorkspace, transferWorkspaceSize));

    //
    // set remote sub state vectors accessible via GPUDirect P2P
    // events should be also set for synchronization
    //
    std::vector<void*> d_subSVsP2P;
    std::vector<int> subSVIndicesP2P;
    std::vector<cudaEvent_t> remoteEvents;
    if (nP2PDeviceBits > 0)
    {
        // distribute device memory handles
        cudaIpcMemHandle_t ipcMemHandle;
        ERRCHK(cudaIpcGetMemHandle(&ipcMemHandle, d_orgSubSV));
        std::vector<cudaIpcMemHandle_t> ipcMemHandles(nSubSVs);
        ERRCHK(MPI_Allgather(&ipcMemHandle, sizeof(ipcMemHandle), MPI_UINT8_T,
                             ipcMemHandles.data(), sizeof(ipcMemHandle), MPI_UINT8_T, MPI_COMM_WORLD));

        // distribute event handles
        cudaIpcEventHandle_t eventHandle;
        ERRCHK(cudaIpcGetEventHandle(&eventHandle, localEvent));
        std::vector<cudaIpcEventHandle_t> ipcEventHandles(nSubSVs);
        ERRCHK(MPI_Allgather(&eventHandle, sizeof(eventHandle), MPI_UINT8_T,
                             ipcEventHandles.data(), sizeof(eventHandle), MPI_UINT8_T, MPI_COMM_WORLD));

        // get remove device pointers and events
        // this calculation assumes that the global rank placement is done in a round-robin fashion
        // across nodes, so for example if nP2PDeviceBits=2 there are 2^2=4 processes/node (and
        // 1 GPU/progress) and we expect the global MPI ranks to be assigned as
        //   0  1  2  3 -> node 0
        //   4  5  6  7 -> node 1
        //   8  9 10 11 -> node 2
        //             ...
        // if the rank placement scheme is different, you will need to calculate based on local MPI
        // rank/size, as CUDA IPC is only for intra-node, not inter-node, communication.
        int p2pSubSVIndexBegin = (orgSubSVIndex / nSubSVsP2P) * nSubSVsP2P;
        int p2pSubSVIndexEnd = p2pSubSVIndexBegin + nSubSVsP2P;
        for (int p2pSubSVIndex = p2pSubSVIndexBegin; p2pSubSVIndex < p2pSubSVIndexEnd; ++p2pSubSVIndex)
        {
            if (orgSubSVIndex == p2pSubSVIndex)
                continue;  // don't need local sub state vector pointer
            void* d_subSVP2P = nullptr;
            const auto& dstMemHandle = ipcMemHandles[p2pSubSVIndex];
            ERRCHK(cudaIpcOpenMemHandle(&d_subSVP2P, dstMemHandle, cudaIpcMemLazyEnablePeerAccess));
            d_subSVsP2P.push_back(d_subSVP2P);
            cudaEvent_t eventP2P = nullptr;
            ERRCHK(cudaIpcOpenEventHandle(&eventP2P, ipcEventHandles[p2pSubSVIndex]));
            remoteEvents.push_back(eventP2P);
            subSVIndicesP2P.push_back(p2pSubSVIndex);
        }

        // set p2p sub state vectors
        ERRCHK(custatevecSVSwapWorkerSetSubSVsP2P(
                       handle, svSegSwapWorker,
                       d_subSVsP2P.data(), subSVIndicesP2P.data(), remoteEvents.data(),
                       static_cast<int>(d_subSVsP2P.size())));
    }

    //
    // create distributed index bit swap scheduler
    //
    custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler;
    ERRCHK(custatevecDistIndexBitSwapSchedulerCreate(
                   handle, &scheduler, nGlobalIndexBits, nLocalIndexBits));

    // set the index bit swaps to the scheduler
    // nSwapBatches is obtained by the call.  This value specifies the number of loops
    unsigned nSwapBatches = 0;
    ERRCHK(custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(
                   handle, scheduler,
                   indexBitSwaps.data(), static_cast<unsigned>(indexBitSwaps.size()),
                   maskBitString.data(), maskOrdering.data(), 0, &nSwapBatches));

    //
    // the main loop of index bit swaps
    //
    constexpr int nLoops = 2;
    for (int loop = 0; loop < nLoops; ++loop)
    {
        double startTime = getTime();
        for (int swapBatchIndex = 0; swapBatchIndex < static_cast<int>(nSwapBatches); ++swapBatchIndex)
        {
            // get parameters
            custatevecSVSwapParameters_t parameters;
            ERRCHK(custatevecDistIndexBitSwapSchedulerGetParameters(
                           handle, scheduler, swapBatchIndex, orgSubSVIndex, &parameters));

            // the rank of the communication endpoint is parameters.dstSubSVIndex
            // as "rank == subSVIndex" is assumed in the present sample.
            int rank = parameters.dstSubSVIndex;
            // set parameters to the worker
            ERRCHK(custatevecSVSwapWorkerSetParameters(
                           handle, svSegSwapWorker, &parameters, rank));
            // execute swap
            ERRCHK(custatevecSVSwapWorkerExecute(
                           handle, svSegSwapWorker, 0, parameters.transferSize));
            // all internal CUDA calls are serialized on localStream
        }
        // synchronize all operations on device
        ERRCHK(cudaStreamSynchronize(localStream));
        // barrier here for time measurement
        ERRCHK(MPI_Barrier(MPI_COMM_WORLD));
        auto elapsedTime = getTime() - startTime;
        if ((loop == nLoops - 1) && (orgSubSVIndex == 0))
        {
            // output benchmark result
            float elmSize = (svDataType == CUDA_C_64F) ? 16 : 8;
            float fraction = 1.f - std::pow(0.5f, indexBitSwaps.size());
            float transferred = std::pow(2.f, nLocalIndexBits) * fraction * elmSize;
            float bw = transferred / elapsedTime * 1.e-9f;
            printf("%s: BW %g [GB/s]\n", procname, bw);
        }
    }

    // free all resources
    ERRCHK(custatevecDistIndexBitSwapSchedulerDestroy(handle, scheduler));
    ERRCHK(custatevecSVSwapWorkerDestroy(handle, svSegSwapWorker));
    ERRCHK(custatevecCommunicatorDestroy(handle, communicator));
    ERRCHK(custatevecDestroy(handle));
    ERRCHK(cudaFree(d_extraWorkspace));
    ERRCHK(cudaFree(d_transferWorkspace));
    // free IPC pointers and events
    for (auto* d_subSV : d_subSVsP2P)
        ERRCHK(cudaIpcCloseMemHandle(d_subSV));
    for (auto event : remoteEvents)
        ERRCHK(cudaEventDestroy(event));
    ERRCHK(cudaFree(d_orgSubSV));
    ERRCHK(cudaEventDestroy(localEvent));
    ERRCHK(cudaStreamDestroy(localStream));
}


int main(int argc, char* argv[])
{

    ERRCHK(MPI_Init(&argc, &argv));

    // get rank and size
    int rank, size;
    ERRCHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    ERRCHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    // size should be a power of two number
    ERRCHK((size & (size - 1)) == 0);

    snprintf(procname, sizeof(procname), "[%d]", rank);

    // compute nGlobalIndexBits from the size 
    // nGlobalIndexBits = log2(size)
    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < size)
        ++nGlobalIndexBits;
    // the size of local sub state vectors
    int nLocalIndexBits = 26;

    // create index bit swap
    int nIndexBitSwaps = 1;
    std::vector<int2> indexBitSwaps;
    int nIndexBits = nLocalIndexBits + nGlobalIndexBits;
    for (int idx = 0; idx < nIndexBitSwaps; ++idx)
        indexBitSwaps.push_back({nLocalIndexBits - 1 - idx, nIndexBits - idx - 1});
    // empty mask
    std::vector<int> maskBitString, maskOrdering;

    runDistributedIndexBitSwaps(rank, size, nGlobalIndexBits, nLocalIndexBits,
                                indexBitSwaps, maskBitString, maskOrdering);

    ERRCHK(MPI_Finalize());
}
