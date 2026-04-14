/* Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cstdint>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"


enum class InputType : int64_t {
    ElementaryOperator,
    MatrixOperator,
    OperatorProductBatchedCoeffs,
    OperatorTermBatchedCoeffs,
    NonBatchedCoeffs
};


enum class OutputType : int64_t {
    OperatorTermBatchedCoeffs,
    OperatorProductBatchedCoeffs,
    Gradient
};


XLA_FFI_DECLARE_HANDLER_SYMBOL(OperatorActionHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(OperatorActionBackwardDiffHandler);
