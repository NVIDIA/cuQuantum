/* Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <string>

#include <cuda_runtime.h>
#include <cudensitymat.h>
#include <xla/ffi/api/ffi.h>


#define FFI_CUDA_ERROR_CHECK(x) \
{ \
    const cudaError_t err = x; \
    if (err != cudaSuccess) { \
        std::string message = "CUDA Error: " + std::string(cudaGetErrorString(err)) + " in line " + std::to_string(__LINE__); \
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, message); \
    } \
}

#define FFI_CUDM_ERROR_CHECK(x) \
{ \
    const cudensitymatStatus_t status = x;  \
    if (status != CUDENSITYMAT_STATUS_SUCCESS) { \
        std::string message = "cuDensityMat error in line " + std::to_string(__LINE__); \
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, message); \
    } \
}
