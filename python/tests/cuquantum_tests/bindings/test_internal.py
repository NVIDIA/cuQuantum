# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from cuquantum import ComputeType, cudaDataType
from cuquantum import bindings

from nvmath import ComputeType as nvmath_compute_type
from nvmath import CudaDataType as nvmath_cuda_data_type

# This test ensures that the ComputeType enum values are consistent across all component libraries, 
# as captured by cuquantum.ComputeType.
@pytest.mark.parametrize("lib", [
    pytest.param("cudensitymat", marks=pytest.mark.cudensitymat),
    pytest.param("custatevec", marks=pytest.mark.custatevec),
    pytest.param("cutensornet", marks=pytest.mark.cutensornet),
    pytest.param("custabilizer", marks=pytest.mark.custabilizer),
    pytest.param("cupauliprop", marks=pytest.mark.cupauliprop),
])
def test_compute_type(lib):
    binding_module = getattr(bindings, lib)
    module_compute_type = getattr(binding_module, "ComputeType", None)
    if lib == "custabilizer":
        # custabilizer currently does not have a ComputeType enum
        assert module_compute_type is None
    else:
        assert module_compute_type is not None
        for val in module_compute_type:
            ref_val = ComputeType(val.value)
            assert val == ref_val, f"{val.name} from {lib} has a different value than cuquantum.ComputeType"


# This test ensures that the ComputeType enum values are consistent between cuquantum and nvmath
# currently nvmath.ComputeType is a subset of cuquantum.ComputeType
@pytest.mark.utility
def test_compute_type_alignment_with_nvmath():
    for val in ComputeType:
        try:
            ref_val = nvmath_compute_type(val.value)
            if val.name != ref_val.name:
                pytest.fail(f"{val.name} from cuquantum has a different name than nvmath.ComputeType")
        except ValueError:
            assert val.name == "COMPUTE_3XTF32"

    for val in nvmath_compute_type:
        ref_val = ComputeType(val.value)
        if val.name != ref_val.name:
            raise ValueError(f"{ref_val.name} from cuquantum has a different name than nvmath.ComputeType")

# This test ensures that the CudaDataType enum values are consistent between cuquantum and nvmath
# currently nvmath.CudaDataType is a *superset* of cuquantum.cudaDataType
@pytest.mark.utility
def test_data_type_alignment_with_nvmath():
    for val in cudaDataType:
        ref_val = nvmath_cuda_data_type(val.value)
        if val.name != ref_val.name:
            raise ValueError(f"{val.name} from cuquantum has a different name than nvmath.CudaDataType")
    
    for val in nvmath_cuda_data_type:
        try:
            ref_val = cudaDataType(val.value)
            if val.name != ref_val.name:
                raise ValueError(f"{val.name} from nvmath has a different name than cuquantum.cudaDataType")
        except ValueError:
            # nvmath.CudaDataType has additional values that are not captured by cuquantum.cudaDataType
            assert val.name in {"CUDA_R_8F_E4M3", "CUDA_R_8F_E5M2", "CUDA_R_4F_E2M1"}