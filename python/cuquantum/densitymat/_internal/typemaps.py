import cuquantum.bindings.cudensitymat as cudm

CUDENSITYMAT_COMPUTE_TYPE_MAP = {
    "float64": cudm.ComputeType.COMPUTE_64F,
    "complex128": cudm.ComputeType.COMPUTE_64F,
    "float32": cudm.ComputeType.COMPUTE_32F,
    "complex64": cudm.ComputeType.COMPUTE_32F,
    }