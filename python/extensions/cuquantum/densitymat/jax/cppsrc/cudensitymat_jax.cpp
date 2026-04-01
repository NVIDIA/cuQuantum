/* Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdio>
#include <cstdint>
#include <vector>
#include <set>

#include <cuda_runtime.h>
#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>
#include "cudensitymat.h"

#include "cudensitymat_jax.h"
#include "utils.h"


xla::ffi::Error OperatorActionImpl(cudaStream_t stream,
                                   xla::ffi::RemainingArgs inBufs,
                                   xla::ffi::Result<xla::ffi::AnyBuffer> workspaceBuf,
                                   xla::ffi::RemainingRets otherOutBufs,
                                   xla::ffi::Span<const int64_t> otherInTypes,
                                   xla::ffi::Span<const int64_t> otherInPtrs,
                                   xla::ffi::Span<const int64_t> otherOutTypes,
                                   xla::ffi::Span<const int64_t> otherOutPtrs,
                                   int64_t batchSize,
                                   int64_t numStateComponents,
                                   intptr_t handleIntPtr,
                                   intptr_t operatorIntPtr,
                                   intptr_t stateInIntPtr,
                                   intptr_t stateOutIntPtr)
{
    try {
        // Convert integer pointers to C API opaque pointers.
        const cudensitymatHandle_t handle = reinterpret_cast<const cudensitymatHandle_t>(handleIntPtr);
        const cudensitymatOperator_t superoperator = reinterpret_cast<const cudensitymatOperator_t>(operatorIntPtr);
        const cudensitymatState_t stateIn = reinterpret_cast<const cudensitymatState_t>(stateInIntPtr);
        cudensitymatState_t stateOut = reinterpret_cast<cudensitymatState_t>(stateOutIntPtr);

        // Process other output buffers.
        std::vector<void*> operatorProductBatchedCoeffsPtrs;
        std::vector<void*> operatorProductBatchedCoeffs;
        std::vector<void*> operatorTermBatchedCoeffsPtrs;
        std::vector<void*> operatorTermBatchedCoeffs;

        // Stream synchronization is needed to ensure JAX has put scalar coefficients buffers on GPU,
        // and copying to the intermediate slots is completed.
        bool needStreamSynchronize = false;
        for (int i = 0; i < otherInTypes.size(); ++i) {
            if (otherInTypes[i] == static_cast<int64_t>(InputType::NonBatchedCoeffs)) {
                needStreamSynchronize = true;
                break;
            }
        }

        if (needStreamSynchronize) {
            FFI_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        }

        // Attach storage to input state.
        std::vector<void*> stateInComponentBufs;
        std::vector<size_t> stateInComponentSizes;
        for (int i = 0; i < inBufs.size(); ++i) {
            xla::ffi::AnyBuffer buf = inBufs.get<xla::ffi::AnyBuffer>(i).value();
            int j = i - numStateComponents;

            if (i < numStateComponents) {
                stateInComponentBufs.push_back(buf.untyped_data());
                stateInComponentSizes.push_back(buf.size_bytes());
            } else if (otherInTypes[j] == static_cast<int64_t>(InputType::ElementaryOperator)) {
                cudensitymatElementaryOperator_t elemOp = reinterpret_cast<cudensitymatElementaryOperator_t>(otherInPtrs[j]);
                FFI_CUDM_ERROR_CHECK(cudensitymatElementaryOperatorAttachBuffer(handle,
                                                                                elemOp,
                                                                                buf.untyped_data(),
                                                                                buf.size_bytes()));
            } else if (otherInTypes[j] == static_cast<int64_t>(InputType::MatrixOperator)) {
                cudensitymatMatrixOperator_t matrixOp = reinterpret_cast<cudensitymatMatrixOperator_t>(otherInPtrs[j]);
                FFI_CUDM_ERROR_CHECK(cudensitymatMatrixOperatorDenseLocalAttachBuffer(handle,
                                                                                      matrixOp,
                                                                                      buf.untyped_data(),
                                                                                      buf.size_bytes()));
            } else if (otherInTypes[j] == static_cast<int64_t>(InputType::OperatorProductBatchedCoeffs)) {
                void* coeffsPtrs = reinterpret_cast<void*>(otherInPtrs[j]);
                operatorProductBatchedCoeffsPtrs.push_back(coeffsPtrs);
                operatorProductBatchedCoeffs.push_back(buf.untyped_data());
            } else if (otherInTypes[j] == static_cast<int64_t>(InputType::OperatorTermBatchedCoeffs)) {
                void* coeffsPtrs = reinterpret_cast<void*>(otherInPtrs[j]);
                operatorTermBatchedCoeffsPtrs.push_back(coeffsPtrs);
                operatorTermBatchedCoeffs.push_back(buf.untyped_data());
            } else if (otherInTypes[j] == static_cast<int64_t>(InputType::NonBatchedCoeffs)) {
                // Non-batched coefficients.
                void* coeffsPtr = reinterpret_cast<void*>(otherInPtrs[j]);
                FFI_CUDA_ERROR_CHECK(cudaMemcpyAsync(coeffsPtr,
                                                     buf.untyped_data(),
                                                     buf.size_bytes(),
                                                     cudaMemcpyDeviceToDevice,
                                                     stream));
            } else {
                return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, "Invalid input type.");
            }
        }

        if (needStreamSynchronize) {
            FFI_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        }

        FFI_CUDM_ERROR_CHECK(cudensitymatStateAttachComponentStorage(handle,
                                                                     stateIn,
                                                                     numStateComponents,
                                                                     stateInComponentBufs.data(),
                                                                     stateInComponentSizes.data()));


        // Set workspace memory.
        cudensitymatWorkspaceDescriptor_t workspaceDesc;
        FFI_CUDM_ERROR_CHECK(cudensitymatCreateWorkspace(handle, &workspaceDesc));

        // NOTE: In Python/JAX we added 255 to the required buffer size. Here we clear the lower 8 bits
        // of the buffer address to ensure the buffer is 256-aligned.
        uintptr_t workspaceIntPtr = reinterpret_cast<uintptr_t>(workspaceBuf->untyped_data());
        void* workspacePtrAligned = reinterpret_cast<void*>((workspaceIntPtr + 255) & ~255);
        size_t workspaceSizeAligned = workspaceBuf->size_bytes() - 255;
        FFI_CUDM_ERROR_CHECK(cudensitymatWorkspaceSetMemory(handle,
                                                            workspaceDesc,
                                                            CUDENSITYMAT_MEMSPACE_DEVICE,
                                                            CUDENSITYMAT_WORKSPACE_SCRATCH,
                                                            workspacePtrAligned,
                                                            workspaceSizeAligned));

        // Attach storage to output state.
        std::vector<void*> stateOutComponentBufs;
        std::vector<size_t> stateOutComponentSizes;

        for (int i = 0; i < numStateComponents; ++i) {
            xla::ffi::Result<xla::ffi::AnyBuffer> resBuf = otherOutBufs.get<xla::ffi::AnyBuffer>(i).value();
            stateOutComponentBufs.push_back(resBuf->untyped_data());
            stateOutComponentSizes.push_back(resBuf->size_bytes());
        }

        FFI_CUDM_ERROR_CHECK(cudensitymatStateAttachComponentStorage(
            handle,
            stateOut,
            numStateComponents,
            stateOutComponentBufs.data(),
            stateOutComponentSizes.data())
        );


        // Execute operator action.

        // Attach batched coefficients if needed.
        if (operatorTermBatchedCoeffsPtrs.size() > 0 || operatorProductBatchedCoeffsPtrs.size() > 0) {
            FFI_CUDM_ERROR_CHECK(cudensitymatAttachBatchedCoefficients(
                handle,
                superoperator,
                operatorTermBatchedCoeffsPtrs.size(),
                operatorTermBatchedCoeffsPtrs.data(),
                operatorTermBatchedCoeffs.data(),
                operatorProductBatchedCoeffsPtrs.size(),
                operatorProductBatchedCoeffsPtrs.data(),
                operatorProductBatchedCoeffs.data()));
        }
    
        // Initialize output state to zero.
        FFI_CUDM_ERROR_CHECK(cudensitymatStateInitializeZero(handle, stateOut, stream));

        FFI_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        FFI_CUDM_ERROR_CHECK(cudensitymatOperatorComputeAction(handle,
                                                               superoperator,
                                                               0.0, // time
                                                               batchSize,
                                                               0, // numParams
                                                               nullptr, // params
                                                               stateIn,
                                                               stateOut,
                                                               workspaceDesc,
                                                               stream));

        FFI_CUDM_ERROR_CHECK(cudensitymatDestroyWorkspace(workspaceDesc));

    } catch (const std::exception& e) {
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, e.what());
    }

    return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    OperatorActionHandler,
    OperatorActionImpl,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs() // all input buffers
        .Ret<xla::ffi::AnyBuffer>() // workspace
        .RemainingRets() // other output buffers
        .Attr<xla::ffi::Span<const int64_t>>("other_in_types")
        .Attr<xla::ffi::Span<const int64_t>>("other_in_ptrs")
        .Attr<xla::ffi::Span<const int64_t>>("other_out_types")
        .Attr<xla::ffi::Span<const int64_t>>("other_out_ptrs")
        .Attr<int64_t>("batch_size")
        .Attr<int64_t>("num_state_components")
        .Attr<intptr_t>("handle")
        .Attr<intptr_t>("operator")
        .Attr<intptr_t>("state_in")
        .Attr<intptr_t>("state_out")
);


xla::ffi::Error OperatorActionBackwardDiffImpl(cudaStream_t stream,
                                               xla::ffi::RemainingArgs inBufs,
                                               xla::ffi::Result<xla::ffi::AnyBuffer> workspaceBuf,
                                               xla::ffi::RemainingRets otherOutBufs,
                                               xla::ffi::Span<const int64_t> otherInTypes,
                                               xla::ffi::Span<const int64_t> otherInPtrs,
                                               xla::ffi::Span<const int64_t> otherOutTypes,
                                               xla::ffi::Span<const int64_t> otherOutPtrs,
                                               int64_t batchSize,
                                               int64_t numStateComponents,
                                               intptr_t handleIntPtr,
                                               intptr_t operatorIntPtr,
                                               intptr_t stateInIntPtr,
                                               intptr_t stateOutAdjIntPtr,
                                               intptr_t stateInAdjIntPtr)
{
    try {
        // Convert integer pointers to C API opaque handles.
        const cudensitymatHandle_t handle = reinterpret_cast<const cudensitymatHandle_t>(handleIntPtr);
        const cudensitymatOperator_t superoperator = reinterpret_cast<const cudensitymatOperator_t>(operatorIntPtr);
        const cudensitymatState_t stateIn = reinterpret_cast<const cudensitymatState_t>(stateInIntPtr);
        const cudensitymatState_t stateOutAdj = reinterpret_cast<const cudensitymatState_t>(stateOutAdjIntPtr);
        cudensitymatState_t stateInAdj = reinterpret_cast<cudensitymatState_t>(stateInAdjIntPtr);

        // Process other input buffers and attach storage to states.
        std::vector<void*> operatorProductBatchedCoeffsPtrs;
        std::vector<void*> operatorProductBatchedCoeffs;
        std::vector<void*> operatorTermBatchedCoeffsPtrs;
        std::vector<void*> operatorTermBatchedCoeffs;
        std::vector<void*> stateInComponentBufs;
        std::vector<size_t> stateInComponentSizes;
        std::vector<void*> stateOutAdjComponentBufs;
        std::vector<size_t> stateOutAdjComponentSizes;

        // Stream synchronization is needed to ensure JAX has put scalar coefficients buffers on GPU,
        // and copying to the intermediate slots is completed.
        bool needStreamSynchronize = false;
        for (int i = 0; i < otherInTypes.size(); ++i) {
            if (otherInTypes[i] == static_cast<int64_t>(InputType::NonBatchedCoeffs)) {
                needStreamSynchronize = true;
                break;
            }
        }

        if (needStreamSynchronize) {
            FFI_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        }

        for (int i = 0; i < inBufs.size(); ++i) {
            xla::ffi::AnyBuffer buf = inBufs.get<xla::ffi::AnyBuffer>(i).value();

            int j = i - 2 * numStateComponents;
            
            if (i < numStateComponents) {
                // State input buffer
                stateInComponentBufs.push_back(buf.untyped_data());
                stateInComponentSizes.push_back(buf.size_bytes());
            } else if (i < 2 * numStateComponents) {
                // State output adjoint buffer
                stateOutAdjComponentBufs.push_back(buf.untyped_data());
                stateOutAdjComponentSizes.push_back(buf.size_bytes());
            } else if (otherInTypes[j] == static_cast<int64_t>(InputType::ElementaryOperator)) {
                cudensitymatElementaryOperator_t elemOp = reinterpret_cast<cudensitymatElementaryOperator_t>(otherInPtrs[j]);
                FFI_CUDM_ERROR_CHECK(cudensitymatElementaryOperatorAttachBuffer(handle,
                                                                                elemOp,
                                                                                buf.untyped_data(),
                                                                                buf.size_bytes()));
            } else if (otherInTypes[j] == static_cast<int64_t>(InputType::MatrixOperator)) {
                cudensitymatMatrixOperator_t matrixOp = reinterpret_cast<cudensitymatMatrixOperator_t>(otherInPtrs[j]);
                FFI_CUDM_ERROR_CHECK(cudensitymatMatrixOperatorDenseLocalAttachBuffer(handle,
                                                                                      matrixOp,
                                                                                      buf.untyped_data(),
                                                                                      buf.size_bytes()));
            } else if (otherInTypes[j] == static_cast<int64_t>(InputType::OperatorProductBatchedCoeffs)) {
                void* coeffsPtrs = reinterpret_cast<void*>(otherInPtrs[j]);
                operatorProductBatchedCoeffsPtrs.push_back(coeffsPtrs);
                operatorProductBatchedCoeffs.push_back(buf.untyped_data());
            } else if (otherInTypes[j] == static_cast<int64_t>(InputType::OperatorTermBatchedCoeffs)) {
                void* coeffsPtrs = reinterpret_cast<void*>(otherInPtrs[j]);
                operatorTermBatchedCoeffsPtrs.push_back(coeffsPtrs);
                operatorTermBatchedCoeffs.push_back(buf.untyped_data());
            } else if (otherInTypes[j] == static_cast<int64_t>(InputType::NonBatchedCoeffs)) {
                // Non-batched coefficients.
                void* coeffsPtr = reinterpret_cast<void*>(otherInPtrs[j]);
                FFI_CUDA_ERROR_CHECK(cudaMemcpyAsync(coeffsPtr,
                                                     buf.untyped_data(),
                                                     buf.size_bytes(),
                                                     cudaMemcpyDeviceToDevice,
                                                     stream));
            } else {
                return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, "Invalid other input type.");
            }
        }

        if (needStreamSynchronize) {
            FFI_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        }

        // FIXME: Do we still need to attach storage to input state since they have been attached in forward execution?
        FFI_CUDM_ERROR_CHECK(cudensitymatStateAttachComponentStorage(handle,
                                                                     stateIn,
                                                                     numStateComponents,
                                                                     stateInComponentBufs.data(),
                                                                     stateInComponentSizes.data()));

        FFI_CUDM_ERROR_CHECK(cudensitymatStateAttachComponentStorage(handle,
                                                                     stateOutAdj,
                                                                     numStateComponents,
                                                                     stateOutAdjComponentBufs.data(),
                                                                     stateOutAdjComponentSizes.data()));

        // Attach storage to output state.
        std::vector<void*> stateInAdjComponentBufs;
        std::vector<size_t> stateInAdjComponentSizes;

        for (int i = 0; i < numStateComponents; ++i) {
            xla::ffi::Result<xla::ffi::AnyBuffer> resBuf = otherOutBufs.get<xla::ffi::AnyBuffer>(i).value();
            stateInAdjComponentBufs.push_back(resBuf->untyped_data());
            stateInAdjComponentSizes.push_back(resBuf->size_bytes());
        }

        FFI_CUDM_ERROR_CHECK(cudensitymatStateAttachComponentStorage(
            handle,
            stateInAdj,
            numStateComponents,
            stateInAdjComponentBufs.data(),
            stateInAdjComponentSizes.data())
        );

        // Set workspace memory.
        cudensitymatWorkspaceDescriptor_t workspaceDesc;
        FFI_CUDM_ERROR_CHECK(cudensitymatCreateWorkspace(handle, &workspaceDesc));

        // NOTE: In Python/JAX we added 255 to the required buffer size. Here we clear the lower 8 bits
        // of the buffer address to ensure the buffer is 256-aligned.
        uintptr_t workspaceIntPtr = reinterpret_cast<uintptr_t>(workspaceBuf->untyped_data());
        void* workspacePtrAligned = reinterpret_cast<void*>((workspaceIntPtr + 255) & ~255);
        size_t workspaceSizeAligned = workspaceBuf->size_bytes() - 255;

        FFI_CUDM_ERROR_CHECK(cudensitymatWorkspaceSetMemory(handle,
                                                            workspaceDesc,
                                                            CUDENSITYMAT_MEMSPACE_DEVICE,
                                                            CUDENSITYMAT_WORKSPACE_SCRATCH,
                                                            workspacePtrAligned,
                                                            workspaceSizeAligned));

        // Initialize output state to zero.
        FFI_CUDM_ERROR_CHECK(cudensitymatStateInitializeZero(handle, stateInAdj, stream));

        // Execute operator action backward differentiation.

        // Attach batched coefficients if needed.
        if (operatorTermBatchedCoeffsPtrs.size() > 0 || operatorProductBatchedCoeffsPtrs.size() > 0) {
            FFI_CUDM_ERROR_CHECK(cudensitymatAttachBatchedCoefficients(
                handle,
                superoperator,
                operatorTermBatchedCoeffsPtrs.size(),
                operatorTermBatchedCoeffsPtrs.data(),
                operatorTermBatchedCoeffs.data(),
                operatorProductBatchedCoeffsPtrs.size(),
                operatorProductBatchedCoeffsPtrs.data(),
                operatorProductBatchedCoeffs.data()));
        }

        FFI_CUDM_ERROR_CHECK(cudensitymatOperatorComputeActionBackwardDiff(handle,
                                                                           superoperator,
                                                                           0.0, // time
                                                                           batchSize,
                                                                           0, // numParams
                                                                           nullptr, // params
                                                                           stateIn,
                                                                           stateOutAdj,
                                                                           stateInAdj,
                                                                           nullptr, // paramsGrad
                                                                           workspaceDesc,
                                                                           stream));

        // The Python gradient callbacks (for both scalar and tensor operators) dispatch
        // their += accumulation operations on CuPy's default CUDA stream, which may differ
        // from the XLA FFI stream used here. Synchronize the default stream before copying
        // so that all callback writes are guaranteed to be visible to the memcpy below.
        FFI_CUDA_ERROR_CHECK(cudaStreamSynchronize(cudaStreamDefault));

        // otherOutBufs layout (after numStateComponents state adj bufs): [grad bufs].
        for (int i = 0; i < otherOutTypes.size(); ++i) {
            xla::ffi::Result<xla::ffi::AnyBuffer> resBuf =
                otherOutBufs.get<xla::ffi::AnyBuffer>(i + numStateComponents).value();
            void* coeffsGradPtr = reinterpret_cast<void*>(otherOutPtrs[i]);
            FFI_CUDA_ERROR_CHECK(cudaMemcpyAsync(resBuf->untyped_data(),
                                                 coeffsGradPtr,
                                                 resBuf->size_bytes(),
                                                 cudaMemcpyDeviceToDevice,
                                                 stream));
        }

        FFI_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        FFI_CUDM_ERROR_CHECK(cudensitymatDestroyWorkspace(workspaceDesc));

    } catch (const std::exception& e) {
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, e.what());
    }

    return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    OperatorActionBackwardDiffHandler,
    OperatorActionBackwardDiffImpl,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs() // all input buffers
        .Ret<xla::ffi::AnyBuffer>() // workspace
        .RemainingRets() // other output buffers
        .Attr<xla::ffi::Span<const int64_t>>("other_in_types")
        .Attr<xla::ffi::Span<const int64_t>>("other_in_ptrs")
        .Attr<xla::ffi::Span<const int64_t>>("other_out_types")
        .Attr<xla::ffi::Span<const int64_t>>("other_out_ptrs")
        .Attr<int64_t>("batch_size")
        .Attr<int64_t>("num_state_components")
        .Attr<intptr_t>("handle")
        .Attr<intptr_t>("operator")
        .Attr<intptr_t>("state_in")
        .Attr<intptr_t>("state_out_adj")
        .Attr<intptr_t>("state_in_adj")
);
