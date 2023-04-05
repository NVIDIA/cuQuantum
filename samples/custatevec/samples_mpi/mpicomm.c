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

#include <custatevec.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <library_types.h>


#include <stdlib.h>
#include <string.h>


/*
 * External communicator for MPI
 */


/*
 * Communicator function types
 */

/* Destroy */
typedef int32_t (*custatevecFnCommunicatorDestroy)(
        custatevecCommunicatorDescriptor_t comm);

/* Barrier */
typedef int (*custatevecFnCommunicatorBarrier)(
        custatevecCommunicatorDescriptor_t comm);

/* SendAsync */
typedef int32_t (*custatevecFnCommunicatorSendAsync)(
        custatevecCommunicatorDescriptor_t comm, const void* buf, int count,
        cudaDataType_t dataType, int peer, int32_t tag);
/* RecvAsync */
typedef int32_t (*custatevecFnCommunicatorRecvAsync)(
        custatevecCommunicatorDescriptor_t comm, void* buf, int count,
        cudaDataType_t dataType, int peer, int32_t tag);

/* SendRecvAsync */
typedef int32_t (*custatevecFnCommunicatorSendRecvAsync)(
        custatevecCommunicatorDescriptor_t comm, const void* sendbuf, void* recvbuf, int count,
        cudaDataType_t dataType, int peer, int32_t tag);

/* Synchronize */
typedef int32_t (*custatevecFnCommunicatorSynchronize)(
        custatevecCommunicatorDescriptor_t comm);


/*
 * Communicator function table
 */


struct custatevecCommunicatorFunctions_t
{
    int                                   version;
    custatevecFnCommunicatorDestroy       destroy;
    custatevecFnCommunicatorBarrier       barrier;
    custatevecFnCommunicatorSendAsync     sendAsync;
    custatevecFnCommunicatorRecvAsync     recvAsync;
    custatevecFnCommunicatorSendRecvAsync sendRecvAsync;
    custatevecFnCommunicatorSynchronize   synchronize;
};


/*
 * MPI Communicator
 */

typedef struct MPICommunicator
{
    // function table
    const custatevecCommunicatorFunctions_t* functions;
    MPI_Request requests[2];
    int nActiveRequests;
} MPICommunicator;


/*
 * MPI Communicator functions
 */

static MPI_Datatype ConvertDataType(cudaDataType_t dataType)
{
    switch (dataType)
    {
    case CUDA_C_32F:
        return MPI_CXX_FLOAT_COMPLEX;
    case CUDA_C_64F:
        return MPI_CXX_DOUBLE_COMPLEX;
    default:
        return MPI_DATATYPE_NULL;
    }
}

static int MPICommunicator_Destroy(custatevecCommunicator_t* comm)
{
    free(comm);
    return 0;
}

static int MPICommunicator_Barrier(custatevecCommunicator_t*)
{
    return MPI_Barrier(MPI_COMM_WORLD);
}

static int MPICommunicator_SendAsync(
        custatevecCommunicatorDescriptor_t comm, const void* buf, int count,
        cudaDataType_t dataType, int peer, int32_t tag)
{
    MPICommunicator* mpicomm = (MPICommunicator*)comm;
    if (mpicomm->nActiveRequests == 2)
        return -1;
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    MPI_Request* request = &mpicomm->requests[mpicomm->nActiveRequests];
    int res = MPI_Isend(buf, count, mpiDatatype, peer, tag, MPI_COMM_WORLD, request);
    if (res != MPI_SUCCESS)
    {
        MPI_Cancel(request);
        return res;
    }
    ++mpicomm->nActiveRequests;
    return 0;
}

static int MPICommunicator_RecvAsync(
        custatevecCommunicatorDescriptor_t comm, void* buf, int count,
        cudaDataType_t dataType, int peer, int32_t tag)
{
    MPICommunicator* mpicomm = (MPICommunicator*)comm;
    if (mpicomm->nActiveRequests == 2)
        return -1;
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    MPI_Request* request = &mpicomm->requests[mpicomm->nActiveRequests];
    int res = MPI_Irecv(buf, count, mpiDatatype, peer, tag, MPI_COMM_WORLD, request);
    if (res != MPI_SUCCESS)
    {
        MPI_Cancel(request);
        return res;
    }
    ++mpicomm->nActiveRequests;
    return 0;
}

static int MPICommunicator_SendRecvAsync(
        custatevecCommunicatorDescriptor_t comm, const void* sendbuf, void* recvbuf, int count,
        cudaDataType_t dataType, int peer, int32_t tag)
{
    MPICommunicator* mpicomm = (MPICommunicator*)comm;
    if (mpicomm->nActiveRequests != 0)
        return -1;
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    MPI_Request* sendRequest = &mpicomm->requests[0];
    MPI_Request* recvRequest = &mpicomm->requests[1];
    int resSend = MPI_Isend(
            sendbuf, count, mpiDatatype, peer, tag, MPI_COMM_WORLD, sendRequest);
    int resRecv = MPI_Irecv(
            recvbuf, count, mpiDatatype, peer, tag, MPI_COMM_WORLD, recvRequest);
    if ((resSend != MPI_SUCCESS) || (resRecv != MPI_SUCCESS))
    {
        MPI_Cancel(sendRequest);
        MPI_Cancel(recvRequest);
        return resSend != MPI_SUCCESS ? resSend : resRecv;
    }
    mpicomm->nActiveRequests = 2;
    return 0;
}

static int MPICommunicator_Synchronize(custatevecCommunicator_t* comm)
{
    MPICommunicator* mpicomm = (MPICommunicator*)comm;
    MPI_Status statuses[2];
    memset(statuses, 0, sizeof(statuses));
    int res = MPI_Waitall(mpicomm->nActiveRequests, mpicomm->requests, statuses);
    mpicomm->nActiveRequests = 0;
    return res;
}

static const custatevecCommunicatorFunctions_t MPICommunicatorFunctions =
{
    1, /* version */
    MPICommunicator_Destroy,
    MPICommunicator_Barrier,
    MPICommunicator_SendAsync,
    MPICommunicator_RecvAsync,
    MPICommunicator_SendRecvAsync,
    MPICommunicator_Synchronize,
};


/* communicator factory function  */
extern "C"
custatevecCommunicatorDescriptor_t mpicommCommunicatorCreate(void)
{
    MPICommunicator* comm = (MPICommunicator*)malloc(sizeof(MPICommunicator));
    memset(comm, 0, sizeof(MPICommunicator));
    comm->functions = &MPICommunicatorFunctions;
    return (custatevecCommunicatorDescriptor_t)comm;
}
