/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <custatevecEx.h>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>

//
// Error checking macro for cuStateVec Ex API
//
#define ERRCHK(s)                                                                                  \
    {                                                                                              \
        auto status = (s);                                                                         \
        if (status != CUSTATEVEC_STATUS_SUCCESS)                                                   \
        {                                                                                          \
            printf("cuStateVec Ex Error: %s at %s:%d\n", custatevecGetErrorString(status),         \
                   __FILE__, __LINE__);                                                            \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

//
// Error checking macro for CUDA API
//
#define ERRCHK_CUDA(s)                                                                             \
    {                                                                                              \
        auto status = (s);                                                                         \
        if (status != cudaSuccess)                                                                 \
        {                                                                                          \
            printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(status), __FILE__, __LINE__);   \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

//
// Error checking macro for MPI Communicator API
// Note: Communicator returns custatevecExCommunicatorStatus_t, not custatevecStatus_t
//
#define ERRCHK_EXCOMM(s)                                                                           \
    {                                                                                              \
        auto commStatus = (s);                                                                     \
        if (commStatus != CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS)                               \
        {                                                                                          \
            printf("cuStateVec Ex Communicator Error at %s:%d (status=%d)\n", __FILE__, __LINE__,  \
                   commStatus);                                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

//
// Error checking macro for cuBLAS API
//
#define ERRCHK_CUBLAS(s)                                                                           \
    {                                                                                              \
        auto status = (s);                                                                         \
        if (status != CUBLAS_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            printf("cuBLAS Error: %d at %s:%d\n", status, __FILE__, __LINE__);                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

//
// Output control function - set to false to suppress output
// Default is true (output enabled)
//
void setOutputEnabled(bool enabled);

//
// Output function declaration - implemented in common.cpp
// Respects setOutputEnabled() for quiet mode
//
void output(const char* format, ...);

//
// Output character function - implemented in common.cpp
// Respects setOutputEnabled() for quiet mode
//
void outputChar(char c);
