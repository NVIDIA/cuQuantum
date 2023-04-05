/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <library_types.h>

/*
 * Plugin struct
 */

typedef struct custatevecCommPlugin custatevecCommPlugin_t;
typedef struct custatevecCommPluginFunctions custatevecCommPluginFunctions_t;

/*
 * Extention methods
 */

/* destructor */
typedef void (*custatevecCommPlugin_FnDestroy)(custatevecCommPlugin_t* commPlugin);

/* Init */
typedef int (*custatevecCommPlugin_FnInit)(
        custatevecCommPlugin_t* commPlugin, int *argc, char ***argv);
/* Finalize */
typedef int (*custatevecCommPlugin_FnFinalize)(custatevecCommPlugin_t* commPlugin);
/* Abort */
typedef int (*custatevecCommPlugin_FnAbort)(custatevecCommPlugin_t* commPlugin, int status);

/* GetCommSize */
typedef int (*custatevecCommPlugin_FnGetSize)(
        custatevecCommPlugin_t* commPlugin, int* size);
/* GetCommRank */
typedef int (*custatevecCommPlugin_FnGetRank)(
        custatevecCommPlugin_t* commPlugin, int* rank);
/* Barrier */
typedef int (*custatevecCommPlugin_FnBarrier)(custatevecCommPlugin_t* commPlugin);
/* Broadcast */
typedef int (*custatevecCommPlugin_FnBcast)(
        custatevecCommPlugin_t* commPlugin, void* buffer, int count, cudaDataType_t dataType, int root);
/* Allgather */
typedef int (*custatevecCommPlugin_FnAllgather)(
        custatevecCommPlugin_t* commPlugin, const void* sendbuf, void* recvbuf, int count, cudaDataType_t dataType);
/* Allgatherv */
typedef int (*custatevecCommPlugin_FnAllgatherv)(
        custatevecCommPlugin_t* commPlugin, const void* sendbuf, int sendcount,
        void* recvbuf, const int* recvcounts, const int* displs, cudaDataType_t dataType);
/* SendAsync */
typedef int (*custatevecCommPlugin_FnSendAsync)(
        custatevecCommPlugin_t* commPlugin, const void* buf, int count,
        cudaDataType_t dataType, int peer, int32_t tag);
/* RecvAsync */
typedef int (*custatevecCommPlugin_FnRecvAsync)(
        custatevecCommPlugin_t* commPlugin, void* buf, int count,
        cudaDataType_t dataType, int peer, int32_t tag);
/* SendRecvAsync */
typedef int (*custatevecCommPlugin_FnSendRecvAsync)(
        custatevecCommPlugin_t* commPlugin, const void* sendbuf, void* recvbuf, int count,
        cudaDataType_t dataType, int peer, int32_t tag);
/* Synchronize */
typedef int (*custatevecCommPlugin_FnSynchronize)(custatevecCommPlugin_t* commPlugin);


/*
 * Function table
 */

struct custatevecCommPluginFunctions {
    int version;
    custatevecCommPlugin_FnDestroy destroy;
    custatevecCommPlugin_FnInit init;
    custatevecCommPlugin_FnFinalize finalize;
    custatevecCommPlugin_FnAbort abort;
    custatevecCommPlugin_FnGetSize getSize;
    custatevecCommPlugin_FnGetRank getRank;
    custatevecCommPlugin_FnBarrier barrier;
    custatevecCommPlugin_FnBcast bcast;
    custatevecCommPlugin_FnAllgather allgather;
    custatevecCommPlugin_FnAllgatherv allgatherv;
    custatevecCommPlugin_FnSendAsync sendAsync;
    custatevecCommPlugin_FnRecvAsync recvAsync;
    custatevecCommPlugin_FnSendRecvAsync sendRecvAsync;
    custatevecCommPlugin_FnSynchronize synchronize;
};

typedef struct MPIPlugin
{
    const custatevecCommPluginFunctions_t* functions;
    MPI_Request requests[2];
    int nActiveRequests;
    int initCalled;
} MPIPlugin;


static MPI_Datatype ConvertDataType(cudaDataType_t dataType)
{
    switch (dataType)
    {
    case CUDA_R_8U:
        return MPI_UINT8_T;
    case CUDA_R_64I:
        return MPI_INT64_T;
    case CUDA_R_64F:
        return MPI_DOUBLE;
    case CUDA_C_32F:
        return MPI_CXX_COMPLEX;
    case CUDA_C_64F:
        return MPI_CXX_DOUBLE_COMPLEX;
    default:
        return MPI_DATATYPE_NULL;
    }
}

static void MPIPlugin_Destroy(custatevecCommPlugin_t* plugin)
{
    free(plugin);
}

static int MPIPlugin_Init(custatevecCommPlugin_t* plugin, int *argc, char ***argv)
{
    int flag = 0;
    int res = MPI_Initialized(&flag);
    if (res != MPI_SUCCESS)
        return res;
    if (flag)
        return MPI_SUCCESS;
    res = MPI_Init(argc, argv);
    if (res != MPI_SUCCESS)
        return res;
    ((MPIPlugin*)plugin)->initCalled = 1;
    return MPI_SUCCESS;
}

static int MPIPlugin_Finalize(custatevecCommPlugin_t* plugin)
{
    if (((MPIPlugin*)plugin)->initCalled == 0)
        return MPI_SUCCESS;
    return MPI_Finalize();
}

static int MPIPlugin_Abort(custatevecCommPlugin_t* plugin, int status)
{
    return MPI_Abort(MPI_COMM_WORLD, status);
}

static int MPIPlugin_Size(custatevecCommPlugin_t* plugin, int* size)
{
    return MPI_Comm_size(MPI_COMM_WORLD, size);
}

static int MPIPlugin_Rank(custatevecCommPlugin_t* plugin, int* rank)
{
    return MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

static int MPIPlugin_Barrier(custatevecCommPlugin_t* plugin)
{
    return MPI_Barrier(MPI_COMM_WORLD);
}

static int MPIPlugin_Bcast(
        custatevecCommPlugin_t* plugin, void* buffer, int count, cudaDataType_t datatype, int root)
{
    MPI_Datatype mpiDatatype = ConvertDataType(datatype);
    return MPI_Bcast(buffer, count, mpiDatatype, root, MPI_COMM_WORLD);
}

static int MPIPlugin_Allgather(
        custatevecCommPlugin_t* plugin,
        const void *sendbuf, void *recvbuf, int count, cudaDataType_t dataType)
{
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    return MPI_Allgather(sendbuf, count, mpiDatatype, recvbuf, count, mpiDatatype, MPI_COMM_WORLD);
}

static int MPIPlugin_Allgatherv(
        custatevecCommPlugin_t* plugin, const void* sendbuf, int sendcount,
        void* recvbuf, const int* recvcounts, const int* displs, cudaDataType_t dataType)
{
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    return MPI_Allgatherv(sendbuf, sendcount, mpiDatatype,
                          recvbuf, recvcounts, displs, mpiDatatype, MPI_COMM_WORLD);
}

static int MPIPlugin_SendAsync(
        custatevecCommPlugin_t* plugin,
        const void* buf, int count, cudaDataType_t dataType, int peer, int32_t tag)
{
    MPIPlugin* mpiPlugin = (MPIPlugin*)plugin;
    if (mpiPlugin->nActiveRequests == 2)
        return -1;
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    MPI_Request* request = &mpiPlugin->requests[mpiPlugin->nActiveRequests];
    int res = MPI_Isend(buf, count, mpiDatatype, peer, tag, MPI_COMM_WORLD, request);
    if (res != MPI_SUCCESS)
    {
        MPI_Cancel(request);
        return res;
    }
    ++mpiPlugin->nActiveRequests;
    return 0;
}

static int MPIPlugin_RecvAsync(
        custatevecCommPlugin_t* plugin,
        void* buf, int count, cudaDataType_t dataType, int peer, int32_t tag)
{
    MPIPlugin* mpiPlugin = (MPIPlugin*)plugin;
    if (mpiPlugin->nActiveRequests == 2)
        return -1;
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    MPI_Request* request = &mpiPlugin->requests[mpiPlugin->nActiveRequests];
    int res = MPI_Irecv(buf, count, mpiDatatype, peer, tag, MPI_COMM_WORLD, request);
    if (res != MPI_SUCCESS)
    {
        MPI_Cancel(request);
        return res;
    }
    ++mpiPlugin->nActiveRequests;
    return 0;
}

static int MPIPlugin_SendRecvAsync(
        custatevecCommPlugin_t* plugin, const void* sendbuf, void* recvbuf, int count,
        cudaDataType_t dataType, int peer, int32_t tag)
{
    MPIPlugin* mpiPlugin = (MPIPlugin*)plugin;
    if (mpiPlugin->nActiveRequests != 0)
        return -1;
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    MPI_Request* sendRequest = &mpiPlugin->requests[0];
    MPI_Request* recvRequest = &mpiPlugin->requests[1];
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
    mpiPlugin->nActiveRequests = 2;
    return 0;
}

static int MPIPlugin_Synchronize(custatevecCommPlugin_t* plugin)
{
    MPIPlugin* mpiPlugin = (MPIPlugin*)plugin;
    MPI_Status statuses[2];
    memset(statuses, 0, sizeof(statuses));
    int res = MPI_Waitall(mpiPlugin->nActiveRequests, mpiPlugin->requests, statuses);
    mpiPlugin->nActiveRequests = 0;
    return res;
}

static const custatevecCommPluginFunctions_t mpiPluginFunctionTable =
{
    1, /* version */
    MPIPlugin_Destroy,
    MPIPlugin_Init,
    MPIPlugin_Finalize,
    MPIPlugin_Abort,
    MPIPlugin_Size,
    MPIPlugin_Rank,
    MPIPlugin_Barrier,
    MPIPlugin_Bcast,
    MPIPlugin_Allgather,
    MPIPlugin_Allgatherv,
    MPIPlugin_SendAsync,
    MPIPlugin_RecvAsync,
    MPIPlugin_SendRecvAsync,
    MPIPlugin_Synchronize,
};


custatevecCommPlugin_t* CommPluginCreate(void)
{
    MPIPlugin *mpiPlugin = (MPIPlugin*)malloc(sizeof(MPIPlugin));
    memset(mpiPlugin, 0, sizeof(MPIPlugin));
    mpiPlugin->functions = &mpiPluginFunctionTable;
    return (custatevecCommPlugin_t*)mpiPlugin;
}
