# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

SHELL            := /bin/bash
CUDA_PATH        := ${CUDA_PATH}
CUTENSOR_ROOT    := ${CUTENSOR_ROOT}
CUTENSORNET_ROOT ?= ${CUQUANTUM_ROOT}
MPI_ROOT         := ${MPI_ROOT}

INCLUDE_DIRS     := -I${CUTENSORNET_ROOT}/include -I${CUTENSOR_ROOT}/include -I${MPI_ROOT}/include
LIBRARY_DIRS     := -L${CUTENSORNET_ROOT}/lib -L${CUTENSORNET_ROOT}/lib64 -L${CUTENSOR_ROOT}/lib/11
LINKER_FLAGS     := -lcutensornet -lcutensor -lcudart -lcusolver

ARCH_FLAGS_SM70  = -gencode arch=compute_70,code=sm_70
ARCH_FLAGS_SM75  = -gencode arch=compute_75,code=sm_75
ARCH_FLAGS_SM80  = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
ARCH_FLAGS_SM86  = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
ARCH_FLAGS_SM90  = -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
ARCH_FLAGS       = $(ARCH_FLAGS_SM70) $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80) $(ARCH_FLAGS_SM86) $(ARCH_FLAGS_SM90) 

CXX_FLAGS        = -std=c++11 $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(LINKER_FLAGS) $(ARCH_FLAGS)

all: check-env
	${CUDA_PATH}/bin/nvcc tensornet_example.cu -o tensornet_example ${CXX_FLAGS}
	${CUDA_PATH}/bin/nvcc tensornet_example_reuse.cu -o tensornet_example_reuse ${CXX_FLAGS}
	${CUDA_PATH}/bin/nvcc approxTN/tensor_svd_example.cu -o tensor_svd_example ${CXX_FLAGS}
	${CUDA_PATH}/bin/nvcc approxTN/tensor_qr_example.cu -o tensor_qr_example ${CXX_FLAGS}
	${CUDA_PATH}/bin/nvcc approxTN/gate_split_example.cu -o gate_split_example ${CXX_FLAGS}
	${CUDA_PATH}/bin/nvcc approxTN/mps_example.cu -o mps_example ${CXX_FLAGS}
    ifdef MPI_ROOT
		${CUDA_PATH}/bin/nvcc tensornet_example_mpi.cu -Xlinker -rpath,${MPI_ROOT}/lib -L${MPI_ROOT}/lib -o tensornet_example_mpi ${CXX_FLAGS} -lmpi
		${CUDA_PATH}/bin/nvcc tensornet_example_mpi_auto.cu -Xlinker -rpath,${MPI_ROOT}/lib -L${MPI_ROOT}/lib -o tensornet_example_mpi_auto ${CXX_FLAGS} -lmpi
    endif

check-env:
	@ echo "" && \
	echo "CUDA_PATH=${CUDA_PATH}"; \
	echo "CUTENSOR_ROOT=${CUTENSOR_ROOT}"; \
	echo "CUTENSORNET_ROOT=${CUTENSORNET_ROOT}"; \
    echo "MPI_ROOT=${MPI_ROOT}"; \
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
	fi; \
	if [[ -z "${MPI_ROOT}" ]]; \
	then \
		echo "" && \
		echo "MPI_ROOT is not set. Please set MPI_ROOT if you would like to build the distributed example(s)." && \
        echo ""; \
	fi

clean:
	rm -f tensornet_example tensornet_example.o tensornet_example_reuse tensornet_example_reuse.o tensornet_example_mpi tensornet_example_mpi.o tensornet_example_mpi_auto tensornet_example_mpi_auto.o
	rm -f tensor_qr_example tensor_qr_example.o tensor_svd_example tensor_svd_example.o
	rm -f gatesplit_example gatesplit_example.o mps_example mps_example.o
