/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * Example: C-based External MPI Communicator Plugin
 *
 * This file demonstrates how to implement a custom external communicator plugin
 * for cuStateVec in pure C. The plugin wraps MPI functions to provide inter-process
 * communication for distributed quantum circuit simulation.
 *
 * Key components:
 * 1. MPICommunicator struct - extends custatevecExCommunicator_t with MPI-specific state
 * 2. Module-level functions - manage MPI library lifecycle (init, finalize, etc.)
 * 3. Instance-level functions - perform communication operations on communicator instances
 * 4. Entry point - custatevecExCommunicatorGetModuleEXT() returns the function table
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <custatevecEx_ext.h>

/*
 * MPI Plugin Implementation
 */

typedef struct MPICommunicator
{
    const custatevecExCommunicatorInterface_t* intf;
    MPI_Request requests[2];
    int nActiveRequests;
} MPICommunicator;

/*
 * Helper function to convert CUDA data types to MPI data types
 */
static MPI_Datatype ConvertDataType(cudaDataType_t dataType)
{
    switch (dataType)
    {
    case CUDA_R_8U:
        return MPI_UINT8_T;
    case CUDA_R_32I:
        return MPI_INT32_T;
    case CUDA_R_64I:
        return MPI_INT64_T;
    case CUDA_R_64F:
        return MPI_DOUBLE;
    case CUDA_C_32F:
        return MPI_C_FLOAT_COMPLEX;
    case CUDA_C_64F:
        return MPI_C_DOUBLE_COMPLEX;
    default:
        return MPI_DATATYPE_NULL;
    }
}

/*
 * Library-level functions (IPC library lifecycle management)
 */

static custatevecExCommunicatorStatus_t MPICommunicator_GetVersion(int32_t* major, int32_t* minor)
{
    // custatevecExCommunicator interface version.
    *major = 0;
    *minor = 0;
    return CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS;
}

static custatevecExCommunicatorStatus_t
MPICommunicator_Init(void* /*moduleHandle*/, int* argc, char*** argv)
{
    return MPI_Init(argc, argv);
}

static custatevecExCommunicatorStatus_t
MPICommunicator_Initialized(void* /*moduleHandle*/, int* flag)
{
    return MPI_Initialized(flag);
}

static custatevecExCommunicatorStatus_t MPICommunicator_Finalize(void* /*moduleHandle*/)
{
    return MPI_Finalize();
}

static custatevecExCommunicatorStatus_t MPICommunicator_Finalized(void* /*moduleHandle*/, int* flag)
{
    return MPI_Finalized(flag);
}

static custatevecExCommunicatorStatus_t
MPICommunicator_GetSizeAndRank(void* /*moduleHandle*/, int32_t* size, int32_t* rank)
{
    int mpi_size, mpi_rank;
    int result1 = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int result2 = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (result1 != MPI_SUCCESS)
        return result1;
    if (result2 != MPI_SUCCESS)
        return result2;

    *size = mpi_size;
    *rank = mpi_rank;
    return CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS;
}

/*
 * Instance-level functions (communicator operations)
 */

static custatevecExCommunicatorStatus_t
MPICommunicator_Abort(custatevecExCommunicator_t* /*exCommunicator*/, int status)
{
    return MPI_Abort(MPI_COMM_WORLD, status);
}

static custatevecExCommunicatorStatus_t
MPICommunicator_GetSize(custatevecExCommunicator_t* /*exCommunicator*/, int* size)
{
    return MPI_Comm_size(MPI_COMM_WORLD, size);
}

static custatevecExCommunicatorStatus_t
MPICommunicator_GetRank(custatevecExCommunicator_t* /*exCommunicator*/, int* rank)
{
    return MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

static custatevecExCommunicatorStatus_t
MPICommunicator_Barrier(custatevecExCommunicator_t* /*exCommunicator*/)
{
    return MPI_Barrier(MPI_COMM_WORLD);
}

static custatevecExCommunicatorStatus_t
MPICommunicator_Bcast(custatevecExCommunicator_t* /*exCommunicator*/, void* buffer, int count,
                      cudaDataType_t dataType, int root)
{
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    if (mpiDatatype == MPI_DATATYPE_NULL)
        return -1;

    return MPI_Bcast(buffer, count, mpiDatatype, root, MPI_COMM_WORLD);
}

static custatevecExCommunicatorStatus_t
MPICommunicator_Allreduce(custatevecExCommunicator_t* /*exCommunicator*/, const void* sendbuf,
                          void* recvbuf, int count, cudaDataType_t dataType)
{
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    if (mpiDatatype == MPI_DATATYPE_NULL)
        return -1;

    return MPI_Allreduce(sendbuf, recvbuf, count, mpiDatatype, MPI_SUM, MPI_COMM_WORLD);
}

static custatevecExCommunicatorStatus_t
MPICommunicator_Allgather(custatevecExCommunicator_t* /*exCommunicator*/, const void* sendbuf,
                          void* recvbuf, int count, cudaDataType_t dataType)
{
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    if (mpiDatatype == MPI_DATATYPE_NULL)
        return -1;

    return MPI_Allgather(sendbuf, count, mpiDatatype, recvbuf, count, mpiDatatype, MPI_COMM_WORLD);
}

static custatevecExCommunicatorStatus_t MPICommunicator_Allgatherv(
    custatevecExCommunicator_t* /*exCommunicator*/, const void* sendbuf, int sendcount,
    void* recvbuf, const int* recvcounts, const int* displs, cudaDataType_t dataType)
{
    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    if (mpiDatatype == MPI_DATATYPE_NULL)
        return -1;

    return MPI_Allgatherv(sendbuf, sendcount, mpiDatatype, recvbuf, recvcounts, displs, mpiDatatype,
                          MPI_COMM_WORLD);
}

static custatevecExCommunicatorStatus_t
MPICommunicator_SendAsync(custatevecExCommunicator_t* exCommunicator, const void* buf, int count,
                          cudaDataType_t dataType, int peer, int32_t tag)
{
    MPICommunicator* mpiComm = (MPICommunicator*)exCommunicator;
    if (mpiComm->nActiveRequests >= 2)
        return -1;

    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    if (mpiDatatype == MPI_DATATYPE_NULL)
        return -1;

    MPI_Request* request = &mpiComm->requests[mpiComm->nActiveRequests];
    int result = MPI_Isend(buf, count, mpiDatatype, peer, tag, MPI_COMM_WORLD, request);
    if (result != MPI_SUCCESS)
    {
        MPI_Cancel(request);
        return result;
    }
    ++mpiComm->nActiveRequests;
    return CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS;
}

static custatevecExCommunicatorStatus_t
MPICommunicator_RecvAsync(custatevecExCommunicator_t* exCommunicator, void* buf, int count,
                          cudaDataType_t dataType, int peer, int32_t tag)
{
    MPICommunicator* mpiComm = (MPICommunicator*)exCommunicator;
    if (mpiComm->nActiveRequests >= 2)
        return -1;

    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    if (mpiDatatype == MPI_DATATYPE_NULL)
        return -1;

    MPI_Request* request = &mpiComm->requests[mpiComm->nActiveRequests];
    int result = MPI_Irecv(buf, count, mpiDatatype, peer, tag, MPI_COMM_WORLD, request);
    if (result != MPI_SUCCESS)
    {
        MPI_Cancel(request);
        return result;
    }
    ++mpiComm->nActiveRequests;
    return CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS;
}

static custatevecExCommunicatorStatus_t MPICommunicator_SendRecvAsync(
    custatevecExCommunicator_t* exCommunicator, const void* sendbuf, void* recvbuf, int count,
    cudaDataType_t dataType, int peer, int32_t tag)
{
    MPICommunicator* mpiComm = (MPICommunicator*)exCommunicator;
    if (mpiComm->nActiveRequests != 0)
        return -1;

    MPI_Datatype mpiDatatype = ConvertDataType(dataType);
    if (mpiDatatype == MPI_DATATYPE_NULL)
        return -1;

    MPI_Request* sendRequest = &mpiComm->requests[0];
    MPI_Request* recvRequest = &mpiComm->requests[1];
    int resSend = MPI_Isend(sendbuf, count, mpiDatatype, peer, tag, MPI_COMM_WORLD, sendRequest);
    int resRecv = MPI_Irecv(recvbuf, count, mpiDatatype, peer, tag, MPI_COMM_WORLD, recvRequest);

    if ((resSend != MPI_SUCCESS) || (resRecv != MPI_SUCCESS))
    {
        MPI_Cancel(sendRequest);
        MPI_Cancel(recvRequest);
        return (custatevecExCommunicatorStatus_t)(resSend != MPI_SUCCESS ? resSend : resRecv);
    }
    mpiComm->nActiveRequests = 2;
    return CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS;
}

static custatevecExCommunicatorStatus_t
MPICommunicator_Synchronize(custatevecExCommunicator_t* exCommunicator)
{
    MPICommunicator* mpiComm = (MPICommunicator*)exCommunicator;
    MPI_Status statuses[2];
    memset(statuses, 0, sizeof(statuses));
    int result = MPI_Waitall(mpiComm->nActiveRequests, mpiComm->requests, statuses);
    mpiComm->nActiveRequests = 0;
    return result;
}

/*
 * Communicator instance creation and destruction
 */

// Static interface table - shared across all communicator instances
static const custatevecExCommunicatorInterface_t staticInterface = {
    .abort = MPICommunicator_Abort,
    .getSize = MPICommunicator_GetSize,
    .getRank = MPICommunicator_GetRank,
    .barrier = MPICommunicator_Barrier,
    .bcast = MPICommunicator_Bcast,
    .allgather = MPICommunicator_Allgather,
    .allgatherv = MPICommunicator_Allgatherv,
    .sendAsync = MPICommunicator_SendAsync,
    .recvAsync = MPICommunicator_RecvAsync,
    .sendRecvAsync = MPICommunicator_SendRecvAsync,
    .synchronize = MPICommunicator_Synchronize,
    .allreduce = MPICommunicator_Allreduce};

static custatevecExCommunicator_t* MPICommunicator_CreateCommunicator(void* libraryHandle)
{
    MPICommunicator* mpiComm = (MPICommunicator*)malloc(sizeof(MPICommunicator));
    if (!mpiComm)
        return NULL;

    memset(mpiComm, 0, sizeof(MPICommunicator));

    // Assign static interface table
    mpiComm->intf = &staticInterface;

    return (custatevecExCommunicator_t*)mpiComm;
}

static void MPICommunicator_DestroyCommunicator(void* /*libraryHandle*/,
                                                custatevecExCommunicator_t* exCommunicator)
{
    if (exCommunicator == NULL)
        return;
    free(exCommunicator);
}

/*
 * Library function table
 */
static const custatevecExCommunicatorModule_t mpiLibraryFunctions = {
    .getVersion = MPICommunicator_GetVersion,
    .init = MPICommunicator_Init,
    .initialized = MPICommunicator_Initialized,
    .finalize = MPICommunicator_Finalize,
    .finalized = MPICommunicator_Finalized,
    .getSizeAndRank = MPICommunicator_GetSizeAndRank,
    .createCommunicator = MPICommunicator_CreateCommunicator,
    .destroyCommunicator = MPICommunicator_DestroyCommunicator};

/*
 * Entry point for external communicator plugin
 *
 * This function is called by custatevecExCommunicatorInitialize() when loading
 * an external communicator library. It must be exported and visible.
 */
custatevecStatus_t
custatevecExCommunicatorGetModuleEXT(const custatevecExCommunicatorModule_t** outModule)
{
    if (outModule == NULL)
        return CUSTATEVEC_STATUS_INVALID_VALUE;

    *outModule = &mpiLibraryFunctions;
    return CUSTATEVEC_STATUS_SUCCESS;
}
