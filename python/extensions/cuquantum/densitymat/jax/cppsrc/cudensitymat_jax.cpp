/* Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
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
#include <cudensitymat.h>

#include "cudensitymat_jax.h"
#include "utils.h"


xla::ffi::Error OperatorActionImpl(cudaStream_t stream,
                                   xla::ffi::Buffer<xla::ffi::F64> timeBuf,
                                   xla::ffi::Buffer<xla::ffi::F64> paramsBuf,
                                   xla::ffi::RemainingArgs otherBufs,
                                   xla::ffi::Result<xla::ffi::AnyBuffer> workspaceBuf,
                                   xla::ffi::RemainingRets stateOutBufs,
                                   xla::ffi::Span<const int64_t> baseOpPtrs,
                                   xla::ffi::Span<const int64_t> isElemOp,
                                   int64_t batchSize,
                                   intptr_t handleIntPtr,
                                   intptr_t operatorIntPtr,
                                   intptr_t stateInIntPtr,
                                   intptr_t stateOutIntPtr,
                                   intptr_t workspaceDescIntPtr)
{
    try {
        // Convert integer pointers to C API opaque pointers.
        const cudensitymatHandle_t handle = reinterpret_cast<const cudensitymatHandle_t>(handleIntPtr);
        const cudensitymatOperator_t superoperator = reinterpret_cast<const cudensitymatOperator_t>(operatorIntPtr);
        const cudensitymatState_t stateIn = reinterpret_cast<const cudensitymatState_t>(stateInIntPtr);
        cudensitymatState_t stateOut = reinterpret_cast<cudensitymatState_t>(stateOutIntPtr);
        cudensitymatWorkspaceDescriptor_t workspaceDesc = reinterpret_cast<cudensitymatWorkspaceDescriptor_t>(workspaceDescIntPtr);

        // Calculate number of buffer pointers and number of state components.
        size_t numBaseOps = baseOpPtrs.size();
        size_t numOtherBufs = otherBufs.size();
        size_t numStateComponents = numOtherBufs - numBaseOps;

        // Find unique base operator pointers.
        std::set<int64_t> uniqueBaseOpPtrs;
        std::vector<int> uniqueBaseOpPtrsInds;
        for (int i = 0; i < numBaseOps; ++i) {
            if (uniqueBaseOpPtrs.find(baseOpPtrs[i]) == uniqueBaseOpPtrs.end()) {
                uniqueBaseOpPtrs.insert(baseOpPtrs[i]);
                uniqueBaseOpPtrsInds.push_back(i);
            }
        }

        // Attach storage to base operators.
        for (int i : uniqueBaseOpPtrsInds) {
            xla::ffi::AnyBuffer buf = otherBufs.get<xla::ffi::AnyBuffer>(i).value();
            if (isElemOp[i]) {
                cudensitymatElementaryOperator_t elemOp = reinterpret_cast<cudensitymatElementaryOperator_t>(baseOpPtrs[i]);
                FFI_CUDM_ERROR_CHECK(cudensitymatElementaryOperatorAttachBuffer(handle,
                                                                                elemOp,
                                                                                buf.untyped_data(),
                                                                                buf.size_bytes()));
            } else {
                cudensitymatMatrixOperator_t matrixOp = reinterpret_cast<cudensitymatMatrixOperator_t>(baseOpPtrs[i]);
                FFI_CUDM_ERROR_CHECK(cudensitymatMatrixOperatorDenseLocalAttachBuffer(handle,
                                                                                      matrixOp,
                                                                                      buf.untyped_data(),
                                                                                      buf.size_bytes()));
            }
        }

        // Attach storage to input state.
        std::vector<void*> stateInComponentBufs;
        std::vector<size_t> stateInComponentSizes;
        for (int i = numBaseOps; i < numOtherBufs; ++i) {
            xla::ffi::AnyBuffer buf = otherBufs.get<xla::ffi::AnyBuffer>(i).value();
            stateInComponentBufs.push_back(buf.untyped_data());
            stateInComponentSizes.push_back(buf.size_bytes());
        }
        FFI_CUDM_ERROR_CHECK(cudensitymatStateAttachComponentStorage(handle,
                                                                     stateIn,
                                                                     numStateComponents,
                                                                     stateInComponentBufs.data(),
                                                                     stateInComponentSizes.data()));

        // Attach storage to output state.
        std::vector<void*> stateOutComponentBufs;
        std::vector<size_t> stateOutComponentSizes;
        for (int i = 0; i < numStateComponents; ++i) {
            xla::ffi::Result<xla::ffi::AnyBuffer> resBuf = stateOutBufs.get<xla::ffi::AnyBuffer>(i).value();
            stateOutComponentBufs.push_back(resBuf->untyped_data());
            stateOutComponentSizes.push_back(resBuf->size_bytes());
        }
        FFI_CUDM_ERROR_CHECK(cudensitymatStateAttachComponentStorage(handle,
                                                                     stateOut,
                                                                     numStateComponents,
                                                                     stateOutComponentBufs.data(),
                                                                     stateOutComponentSizes.data()));

        // Set workspace memory.
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
        FFI_CUDM_ERROR_CHECK(cudensitymatStateInitializeZero(handle, stateOut, stream));

        // Execute operator action.
        // TODO: time needs to be copied from device to host. Is there a better way to handle this?
        double time;
        FFI_CUDA_ERROR_CHECK(cudaMemcpy(&time, timeBuf.typed_data(), sizeof(double), cudaMemcpyDeviceToHost));

        FFI_CUDM_ERROR_CHECK(cudensitymatOperatorComputeAction(handle,
                                                               superoperator,
                                                               time,
                                                               batchSize,
                                                               paramsBuf.element_count(),
                                                               paramsBuf.typed_data(),
                                                               stateIn,
                                                               stateOut,
                                                               workspaceDesc,
                                                               stream));

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
        .Arg<xla::ffi::Buffer<xla::ffi::F64>>() // time
        .Arg<xla::ffi::Buffer<xla::ffi::F64>>() // params
        .RemainingArgs() // base operators and input state
        .Ret<xla::ffi::AnyBuffer>() // workspace
        .RemainingRets() // output state
        .Attr<xla::ffi::Span<const int64_t>>("base_op_ptrs")
        .Attr<xla::ffi::Span<const int64_t>>("is_elem_op")
        .Attr<int64_t>("batch_size")
        .Attr<intptr_t>("handle")
        .Attr<intptr_t>("operator")
        .Attr<intptr_t>("state_in")
        .Attr<intptr_t>("state_out")
        .Attr<intptr_t>("workspace_desc")
);


xla::ffi::Error OperatorActionBackwardDiffImpl(cudaStream_t stream,
                                               xla::ffi::Buffer<xla::ffi::F64> timeBuf,
                                               xla::ffi::Buffer<xla::ffi::F64> paramsBuf,
                                               xla::ffi::RemainingArgs otherBufs,
                                               xla::ffi::Result<xla::ffi::AnyBuffer> workspaceBuf,
                                               xla::ffi::Result<xla::ffi::Buffer<xla::ffi::F64>> paramsGradBuf,
                                               xla::ffi::RemainingRets stateInAdjBufs,
                                               xla::ffi::Span<const int64_t> baseOpPtrs,
                                               xla::ffi::Span<const int64_t> isElemOp,
                                               int64_t batchSize,
                                               intptr_t handleIntPtr,
                                               intptr_t operatorIntPtr,
                                               intptr_t stateInIntPtr,
                                               intptr_t stateOutAdjIntPtr,
                                               intptr_t stateInAdjIntPtr,
                                               intptr_t workspaceDescIntPtr)
{
    try {
        // Convert integer pointers to C API opaque handles.
        const cudensitymatHandle_t handle = reinterpret_cast<const cudensitymatHandle_t>(handleIntPtr);
        const cudensitymatOperator_t superoperator = reinterpret_cast<const cudensitymatOperator_t>(operatorIntPtr);
        const cudensitymatState_t stateIn = reinterpret_cast<const cudensitymatState_t>(stateInIntPtr);
        const cudensitymatState_t stateOutAdj = reinterpret_cast<const cudensitymatState_t>(stateOutAdjIntPtr);
        cudensitymatState_t stateInAdj = reinterpret_cast<cudensitymatState_t>(stateInAdjIntPtr);
        cudensitymatWorkspaceDescriptor_t workspaceDesc = reinterpret_cast<cudensitymatWorkspaceDescriptor_t>(workspaceDescIntPtr);

        // Calculate number of buffer pointers and number of state components.
        size_t numBaseOps = baseOpPtrs.size();
        size_t numOtherBufs = otherBufs.size();
        size_t numStateComponents = (numOtherBufs - numBaseOps) / 2;

        // Find unique base operator pointers.
        std::set<int64_t> uniqueBaseOpPtrs;
        std::vector<int> uniqueBaseOpPtrsInds;
        for (int i = 0; i < numBaseOps; ++i) {
            if (uniqueBaseOpPtrs.find(baseOpPtrs[i]) == uniqueBaseOpPtrs.end()) {
                uniqueBaseOpPtrs.insert(baseOpPtrs[i]);
                uniqueBaseOpPtrsInds.push_back(i);
            }
        }

        // Attach storage to elementary operators.
        // NOTE: operator PyTree is flattened and unflattened when calling custom VJP, so here it 
        // will always attach all base operators.
        for (int i : uniqueBaseOpPtrsInds) {
            xla::ffi::AnyBuffer buf = otherBufs.get<xla::ffi::AnyBuffer>(i).value();
            if (isElemOp[i]) {
                cudensitymatElementaryOperator_t elemOp = reinterpret_cast<cudensitymatElementaryOperator_t>(baseOpPtrs[i]);
                FFI_CUDM_ERROR_CHECK(cudensitymatElementaryOperatorAttachBuffer(handle,
                                                                                elemOp,
                                                                                buf.untyped_data(),
                                                                                buf.size_bytes()));
            } else {
                cudensitymatMatrixOperator_t matrixOp = reinterpret_cast<cudensitymatMatrixOperator_t>(baseOpPtrs[i]);
                FFI_CUDM_ERROR_CHECK(cudensitymatMatrixOperatorDenseLocalAttachBuffer(handle,
                                                                                      matrixOp,
                                                                                      buf.untyped_data(),
                                                                                      buf.size_bytes()));
            }
        }

        // Attach storage to input state.
        std::vector<void*> stateInComponentBufs;
        std::vector<size_t> stateInComponentSizes;
        std::vector<void*> stateOutAdjComponentBufs;
        std::vector<size_t> stateOutAdjComponentSizes;
        for (int i = numBaseOps; i < numBaseOps + numStateComponents; ++i) {
            xla::ffi::AnyBuffer stateInBuf = otherBufs.get<xla::ffi::AnyBuffer>(i).value();
            stateInComponentBufs.push_back(stateInBuf.untyped_data());
            stateInComponentSizes.push_back(stateInBuf.size_bytes());

            xla::ffi::AnyBuffer stateOutAdjBuf = otherBufs.get<xla::ffi::AnyBuffer>(i + numStateComponents).value();
            stateOutAdjComponentBufs.push_back(stateOutAdjBuf.untyped_data());
            stateOutAdjComponentSizes.push_back(stateOutAdjBuf.size_bytes());
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
            xla::ffi::Result<xla::ffi::AnyBuffer> buf = stateInAdjBufs.get<xla::ffi::AnyBuffer>(i).value();
            stateInAdjComponentBufs.push_back(buf->untyped_data());
            stateInAdjComponentSizes.push_back(buf->size_bytes());
        }
        FFI_CUDM_ERROR_CHECK(cudensitymatStateAttachComponentStorage(handle,
                                                                     stateInAdj,
                                                                     numStateComponents,
                                                                     stateInAdjComponentBufs.data(),
                                                                     stateInAdjComponentSizes.data()));

        // Set workspace memory.
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

        // Execute operator action.
        // TODO: time needs to be copied from device to host. Is there a better way to handle this?
        double time;
        FFI_CUDA_ERROR_CHECK(cudaMemcpy(&time, timeBuf.typed_data(), sizeof(double), cudaMemcpyDeviceToHost));

        FFI_CUDM_ERROR_CHECK(cudensitymatOperatorComputeActionBackwardDiff(handle,
                                                                           superoperator,
                                                                           time,
                                                                           batchSize,
                                                                           paramsBuf.element_count(),
                                                                           paramsBuf.typed_data(),
                                                                           stateIn,
                                                                           stateOutAdj,
                                                                           stateInAdj,
                                                                           paramsGradBuf->typed_data(),
                                                                           workspaceDesc,
                                                                           stream));

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
        .Arg<xla::ffi::Buffer<xla::ffi::F64>>() // time
        .Arg<xla::ffi::Buffer<xla::ffi::F64>>() // params
        .RemainingArgs() // base operators, input state and output state adjoint
        .Ret<xla::ffi::AnyBuffer>() // workspace
        .Ret<xla::ffi::Buffer<xla::ffi::F64>>() // paramsGrad
        .RemainingRets() // input state adjoint
        .Attr<xla::ffi::Span<const int64_t>>("base_op_ptrs")
        .Attr<xla::ffi::Span<const int64_t>>("is_elem_op")
        .Attr<int64_t>("batch_size")
        .Attr<intptr_t>("handle")
        .Attr<intptr_t>("operator")
        .Attr<intptr_t>("state_in")
        .Attr<intptr_t>("state_out_adj")
        .Attr<intptr_t>("state_in_adj")
        .Attr<intptr_t>("workspace_desc")
);
