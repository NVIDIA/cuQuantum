# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

SHELL            := /bin/bash
CUDA_PATH        := ${CUDA_PATH}
CUDENSITYMAT_ROOT ?= ${CUQUANTUM_ROOT}
MPI_ROOT         := ${MPI_ROOT}
CUDA_MAJOR_VERSION := $(shell "${CUDA_PATH}/bin/"nvcc --version | egrep -o "V[0-9]+.[0-9]+.[0-9]" | cut -d. -f1 | cut -c2-)
CUDA_MINOR_VERSION := $(shell "${CUDA_PATH}/bin/"nvcc --version | egrep -o "V[0-9]+.[0-9]+.[0-9]" | cut -d. -f2)
INCLUDE_DIRS     := -I${CUDENSITYMAT_ROOT}/include -I${CUTENSOR_ROOT}/include
LIBRARY_DIRS     := -L${CUDENSITYMAT_ROOT}/lib -lcudensitymat -L${CUTENSOR_ROOT}/lib/${CUDA_MAJOR_VERSION} -lcutensor
ARCH_FLAGS_SM70  = -gencode arch=compute_70,code=sm_70
ARCH_FLAGS_SM75  = -gencode arch=compute_75,code=sm_75
ARCH_FLAGS_SM80  = -gencode arch=compute_80,code=sm_80
ARCH_FLAGS_SM86  = -gencode arch=compute_86,code=sm_86
ARCH_FLAGS_SM90  = -gencode arch=compute_90,code=sm_90   -gencode arch=compute_90,code=compute_90
ARCH_FLAGS_SM100 = -gencode arch=compute_100,code=sm_100 -gencode arch=compute_100,code=compute_100
ARCH_FLAGS_SM120 = -gencode arch=compute_120,code=sm_120

ifeq ($(shell [ $(CUDA_MAJOR_VERSION) -lt 12 ] || ( [ $(CUDA_MAJOR_VERSION) -eq 12 ] && [ $(CUDA_MINOR_VERSION) -lt 8 ] ); echo $$?),0)
	ARCH_FLAGS = $(ARCH_FLAGS_SM70) $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80) $(ARCH_FLAGS_SM86) $(ARCH_FLAGS_SM90) 
else
	ARCH_FLAGS = $(ARCH_FLAGS_SM70) $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80) $(ARCH_FLAGS_SM86) $(ARCH_FLAGS_SM90) \
	             $(ARCH_FLAGS_SM100) $(ARCH_FLAGS_SM120)
endif

CXX_FLAGS        = -std=c++17 $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(ARCH_FLAGS)

all: check-env
	${CUDA_PATH}/bin/nvcc operator_action_example.cpp ${CXX_FLAGS} -o operator_action_example
	${CUDA_PATH}/bin/nvcc operator_action_gradient_example.cpp ${CXX_FLAGS} -o operator_action_gradient_example
  ifdef MPI_ROOT
	${CUDA_PATH}/bin/nvcc operator_action_mpi_example.cpp ${CXX_FLAGS} -L${MPI_ROOT}/lib -lmpi -I${MPI_ROOT}/include -o operator_action_mpi_example
  endif

operator_action_example:
	${CUDA_PATH}/bin/nvcc operator_action_example.cpp ${CXX_FLAGS} -o operator_action_example

operator_action_gradient_example:
	${CUDA_PATH}/bin/nvcc operator_action_gradient_example.cpp ${CXX_FLAGS} -o operator_action_gradient_example

ifdef MPI_ROOT
operator_action_mpi_example:
	${CUDA_PATH}/bin/nvcc operator_action_mpi_example.cpp ${CXX_FLAGS} -L${MPI_ROOT}/lib -lmpi -I${MPI_ROOT}/include -o operator_action_mpi_example
endif

check-env:
	@ echo "" && \
	echo "CUDA_PATH=${CUDA_PATH}"; \
	echo "CUDENSITYMAT_ROOT=${CUDENSITYMAT_ROOT}"; \
    echo "MPI_ROOT=${MPI_ROOT}"; \
	echo ""; \
	if [[ -z "${CUDENSITYMAT_ROOT}" ]]; \
	then \
		echo "" && \
		echo "Neither CUDENSITYMAT_ROOT nor CUQUANTUM_ROOT is set." && \
		exit 1; \
	fi; \
	if [[ -z "${MPI_ROOT}" ]]; \
	then \
		echo "" && \
		echo "MPI_ROOT is not set. Please set MPI_ROOT if you would like to build the distributed example(s)." && \
        echo ""; \
	fi

clean:
	rm -f operator_action_example operator_action_example.o operator_action_gradient_example operator_action_gradient_example.o
ifdef MPI_ROOT
	rm -f operator_action_mpi_example operator_action_mpi_example.o
endif
