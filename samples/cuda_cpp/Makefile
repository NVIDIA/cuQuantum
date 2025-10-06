# CUDA C++ Examples - Makefile

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++14
LIBS = -lcustatevec -lcublas -lcutensornet

# Directories
BASIC_DIR = basic
ADVANCED_DIR = advanced

# Basic examples
BASIC_SOURCES = $(wildcard $(BASIC_DIR)/*.cu)
BASIC_TARGETS = $(BASIC_SOURCES:.cu=)

# Advanced examples
ADVANCED_SOURCES = $(wildcard $(ADVANCED_DIR)/*.cu)
ADVANCED_TARGETS = $(ADVANCED_SOURCES:.cu=)

# All targets
ALL_TARGETS = $(BASIC_TARGETS) $(ADVANCED_TARGETS)

.PHONY: all basic advanced clean help

all: basic advanced

basic: $(BASIC_TARGETS)
	@echo "✓ Basic examples built"

advanced: $(ADVANCED_TARGETS)
	@echo "✓ Advanced examples built"

# Pattern rule for building executables
%: %.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)
	@echo "✓ Built $@"

clean:
	@echo "Cleaning..."
	rm -f $(ALL_TARGETS)
	@echo "✓ Clean complete"

help:
	@echo "cuQuantum CUDA C++ Examples - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make all       - Build all examples"
	@echo "  make basic     - Build basic examples only"
	@echo "  make advanced  - Build advanced examples only"
	@echo "  make clean     - Remove all built executables"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make basic/01_bell_state    - Build specific example"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA Toolkit 12.0+"
	@echo "  - cuQuantum SDK"
	@echo "  - NVIDIA GPU (Compute Capability 7.0+)"
