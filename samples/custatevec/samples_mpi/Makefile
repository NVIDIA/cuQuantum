# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

CUDA_TOOLKIT     := $(shell realpath $$(dirname $$(command -v nvcc))/..)
CUSTATEVEC_ROOT  ?= ${CUQUANTUM_ROOT}

CXXFLAGS=-Wall

has_ompiinfo:=$(shell which ompi_info > /dev/null; echo $$?)
ifeq ($(has_ompiinfo), 0)
	CXX=$(shell mpicxx --showme:command)
	ompi_incdirs := $(shell mpicxx --showme:incdirs)
	MPI_INCLUDES=$(addprefix -I, $(ompi_incdirs))
	MPI_LINKER_FLAGS := $(shell mpicxx --showme:link)
endif

has_mpichversion:=$(shell which mpichversion > /dev/null; echo $$?)
ifeq ($(has_mpichversion), 0)
	CXX := $(shell mpicxx -link_info | cut -c -4)
	MPI_LINKER_FLAGS := $(shell mpicxx -link_info | cut -c 5-)
endif

INCLUDE_DIRS     := $(MPI_INCLUDES) -I$(CUDA_TOOLKIT)/include -I$(CUSTATEVEC_ROOT)/include
LIBRARY_DIRS     := -L$(CUSTATEVEC_ROOT)/lib64 -L$(CUSTATEVEC_ROOT)/lib -L$(CUDA_TOOLKIT)/lib64
LINKER_FLAGS     := $(MPI_LINKER_FLAGS) ${LIBRARY_DIRS} -lcustatevec -lcudart

all: distributedIndexBitSwap mpicomm.so

distributedIndexBitSwap: distributedIndexBitSwap.cpp
	$(CXX) $(INCLUDE_DIRS) $(CXXFLAGS) $< $(LINKER_FLAGS) -o $@

mpicomm.so: mpicomm.c
	$(CXX) $(INCLUDE_DIRS) $(CXXFLAGS) -fPIC -shared $< $(MPI_LINKER_FLAGS) -o $@

.PHONY:

clean:
	rm -f distributedIndexBitSwap mpicomm.so
