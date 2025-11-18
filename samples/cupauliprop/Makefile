# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

SHELL              := /bin/bash
CUDA_PATH          := ${CUDA_PATH}
CUDA_MAJOR_VERSION := $(shell "${CUDA_PATH}/bin/"nvcc --version | egrep -o "V[0-9]+.[0-9]+.[0-9]" | cut -d. -f1 | cut -c2-)
CUDA_MINOR_VERSION := $(shell "${CUDA_PATH}/bin/"nvcc --version | egrep -o "V[0-9]+.[0-9]+.[0-9]" | cut -d. -f2)

CUPAULIPROP_ROOT ?= ${CUQUANTUM_ROOT}

INCLUDE_DIRS     := -I${CUPAULIPROP_ROOT}/include
LIBRARY_DIRS     := -L${CUPAULIPROP_ROOT}/lib -L${CUPAULIPROP_ROOT}/lib64 -lcupauliprop
ARCH_FLAGS_SM75  = -gencode arch=compute_75,code=sm_75
ARCH_FLAGS_SM80  = -gencode arch=compute_80,code=sm_80
ARCH_FLAGS_SM86  = -gencode arch=compute_86,code=sm_86
ARCH_FLAGS_SM90  = -gencode arch=compute_90,code=sm_90   -gencode arch=compute_90,code=compute_90
ARCH_FLAGS_SM100 = -gencode arch=compute_100,code=sm_100 -gencode arch=compute_100,code=compute_100
ARCH_FLAGS_SM120 = -gencode arch=compute_120,code=sm_120

ifeq ($(shell [ $(CUDA_MAJOR_VERSION) -lt 12 ] || ( [ $(CUDA_MAJOR_VERSION) -eq 12 ] && [ $(CUDA_MINOR_VERSION) -lt 8 ] ); echo $$?),0)
	ARCH_FLAGS = $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80) $(ARCH_FLAGS_SM86) $(ARCH_FLAGS_SM90)
else
	ARCH_FLAGS = $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80) $(ARCH_FLAGS_SM86) $(ARCH_FLAGS_SM90) \
	             $(ARCH_FLAGS_SM100) $(ARCH_FLAGS_SM120)
endif

CXX_FLAGS = -std=c++17 $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(ARCH_FLAGS)

all: check-env
	${CUDA_PATH}/bin/nvcc kicked_ising_example.cpp ${CXX_FLAGS} -o kicked_ising_example

kicked_ising_example:
	${CUDA_PATH}/bin/nvcc kicked_ising_example.cpp ${CXX_FLAGS} -o kicked_ising_example

check-env:
	@ echo "" && \
	echo "CUDA_PATH=${CUDA_PATH}"; \
	echo "CUPAULIPROP_ROOT=${CUPAULIPROP_ROOT}"; \
	echo ""; \
	if [[ -z "${CUPAULIPROP_ROOT}" ]]; \
	then \
		echo "" && \
		echo "Neither CUPAULIPROP_ROOT nor CUQUANTUM_ROOT is set." && \
		exit 1; \
	fi; \

clean:
	rm -f \
		kicked_ising_example kicked_ising_example.o
