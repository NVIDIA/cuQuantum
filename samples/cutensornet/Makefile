#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
