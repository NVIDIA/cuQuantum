# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause


export CUDART_LIB=$(     python -c "from search_package_path import get_link_path; print(get_link_path('cuda-runtime', $CUDA_MAJOR))")
export CUSOLVER_LIB=$(   python -c "from search_package_path import get_link_path; print(get_link_path('cusolver',     $CUDA_MAJOR))")
export CUSPARSE_LIB=$(   python -c "from search_package_path import get_link_path; print(get_link_path('cusparse',     $CUDA_MAJOR))")
export CUTENSORNET_LIB=$(python -c "from search_package_path import get_link_path; print(get_link_path('cutensornet',  $CUDA_MAJOR))")

export CUTENSORNET_INCLUDE=$(python -c "from search_package_path import get_include_path; print(get_include_path('cutensornet',  $CUDA_MAJOR))")
