# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

SHELL            := /bin/bash
CUDA_TOOLKIT     := $(shell realpath $$(dirname $$(command -v nvcc))/..)
CUTENSORNET_ROOT ?= ${CUQUANTUM_ROOT}

INCLUDE_DIRS     := -I${CUTENSORNET_ROOT}/include -I${CUTENSOR_ROOT}/include
LIBRARY_DIRS     := -L${CUTENSORNET_ROOT}/lib -L${CUTENSORNET_ROOT}/lib64 -L${CUTENSOR_ROOT}/lib/11
LINKER_FLAGS     := -lcutensornet -lcutensor -lcudart -lstdc++

ARCH_FLAGS_SM70  = -gencode arch=compute_70,code=sm_70
ARCH_FLAGS_SM75  = -gencode arch=compute_75,code=sm_75
ARCH_FLAGS_SM80  = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
ARCH_FLAGS       = $(ARCH_FLAGS_SM70) $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80)

CXX_FLAGS        =-std=c++11 $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(LINKER_FLAGS) $(ARCH_FLAGS)

all: check-env
	nvcc tensornet_example.cu -o tensornet_example ${CXX_FLAGS}

check-env:
	@ echo "" && \
	echo "CUDA_TOOLKIT=${CUDA_TOOLKIT}"; \
	echo "CUTENSOR_ROOT=${CUTENSOR_ROOT}"; \
	echo "CUTENSORNET_ROOT=${CUTENSORNET_ROOT}"; \
	echo ""; \
	if [[ -z "${CUTENSOR_ROOT}" ]]; \
	then \
		echo "" && \
		echo "CUTENSOR_ROOT is not set." && \
		exit 1; \
	fi; \
	if [[ -z "${CUTENSORNET_ROOT}" ]]; \
	then \
		echo "" && \
		echo "Neither CUTENSORNET_ROOT nor CUQUANTUM_ROOT is set." && \
		exit 1; \
	fi

clean:
	rm -f tensornet_example tensornet_example.o
