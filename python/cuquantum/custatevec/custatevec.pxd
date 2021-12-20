# TODO: Ultimately, everything should be auto-generated using
# the scripts from the CUDA Python team

from libc.stdint cimport intptr_t, int32_t, uint32_t, int64_t


# The C types are prefixed with an underscore because we are not
# yet protected by the module namespaces as done in CUDA Python.
# Once we switch over the names would be prettier (in the Cython
# layer).

cdef extern from '<custatevec.h>' nogil:
    # cuStateVec types
    ctypedef void* _Handle 'custatevecHandle_t'
    ctypedef int64_t _Index 'custatevecIndex_t'
    ctypedef int _Status 'custatevecStatus_t'
    ctypedef struct _SamplerDescriptor 'custatevecSamplerDescriptor_t':
        pass
    ctypedef struct _AccessorDescriptor 'custatevecAccessorDescriptor':
        pass
    ctypedef enum _ComputeType 'custatevecComputeType_t':
        pass
    # ctypedef void(*custatevecLoggerCallback_t)(
    #     int32_t logLevel,
    #     const char* functionName,
    #     const char* message)
    # ctypedef custatevecLoggerCallback_t LoggerCallback

    # cuStateVec enums
    ctypedef enum _Pauli 'custatevecPauli_t':
        CUSTATEVEC_PAULI_I
        CUSTATEVEC_PAULI_X
        CUSTATEVEC_PAULI_Y
        CUSTATEVEC_PAULI_Z

    ctypedef enum _MatrixLayout 'custatevecMatrixLayout_t':
        CUSTATEVEC_MATRIX_LAYOUT_COL
        CUSTATEVEC_MATRIX_LAYOUT_ROW

    ctypedef enum _MatrixType 'custatevecMatrixType_t':
        CUSTATEVEC_MATRIX_TYPE_GENERAL
        CUSTATEVEC_MATRIX_TYPE_UNITARY
        CUSTATEVEC_MATRIX_TYPE_HERMITIAN

    ctypedef enum _CollapseOp 'custatevecCollapseOp_t':
        CUSTATEVEC_COLLAPSE_NONE
        CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO

    ctypedef enum _SamplerOutput 'custatevecSamplerOutput_t':
        CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER
        CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER

    # cuStateVec consts
    int CUSTATEVEC_VER_MAJOR
    int CUSTATEVEC_VER_MINOR
    int CUSTATEVEC_VER_PATCH
    int CUSTATEVEC_VERSION
