# CUTLASS Matrix Multiplication Auto-Tuning Makefile
# For NVIDIA RTX 3090 (Ampere, SM86)

# CUDA and CUTLASS paths
CUDA_PATH ?= /usr/local/cuda
CUTLASS_PATH ?= /home/jin/cutlass

# Compiler
NVCC = $(CUDA_PATH)/bin/nvcc

# Compiler flags
NVCC_FLAGS = -std=c++17 \
             -O3 \
             -arch=sm_86 \
             -I$(CUTLASS_PATH)/include \
             -I$(CUTLASS_PATH)/tools/util/include \
             --expt-relaxed-constexpr \
             -Xcompiler=-fPIC

# cuBLAS linking flags
CUBLAS_FLAGS = -lcublas

# Targets
TARGETS = cutlass_matmul_tuning verify_correctness

.PHONY: all clean autotune run help

all: $(TARGETS)

# Basic auto-tuning executable
cutlass_matmul_tuning: cutlass_matmul_tuning.cu
	@echo "Compiling CUTLASS matrix multiplication auto-tuner..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "Build complete: ./$@"

# Correctness verification executable
verify_correctness: verify_correctness.cu
	@echo "Compiling CUTLASS correctness verification..."
	$(NVCC) $(NVCC_FLAGS) $(CUBLAS_FLAGS) $< -o $@
	@echo "Build complete: ./$@"

# Generate and compile extensive auto-tuning
autotune: autotune.py
	@echo "Generating extensive auto-tuning code..."
	python3 autotune.py
	@echo "Compiling generated auto-tuner..."
	$(NVCC) $(NVCC_FLAGS) cutlass_autotune_generated.cu -o cutlass_autotune_generated
	@echo "Build complete: ./cutlass_autotune_generated"

# Run basic tuning
run: cutlass_matmul_tuning
	@echo "Running auto-tuning..."
	./cutlass_matmul_tuning

# Run correctness verification
verify: verify_correctness
	@echo "Running correctness verification..."
	./verify_correctness

# Run extensive tuning and save results
run-autotune: autotune
	@echo "Running extensive auto-tuning (this may take a while)..."
	./cutlass_autotune_generated 2>&1 | tee results.csv
	@echo ""
	@echo "Results saved to results.csv"

# Clean build artifacts
clean:
	rm -f $(TARGETS) cutlass_autotune_generated
	rm -f cutlass_autotune_generated.cu
	rm -f results.csv tuning.log
	@echo "Clean complete"

# Help
help:
	@echo "CUTLASS Matrix Multiplication Auto-Tuning"
	@echo "=========================================="
	@echo ""
	@echo "Targets:"
	@echo "  make                  - Build basic auto-tuner and verification"
	@echo "  make autotune         - Generate and build extensive auto-tuner"
	@echo "  make run              - Run basic auto-tuning (12 configs)"
	@echo "  make run-autotune     - Run extensive auto-tuning (100+ configs)"
	@echo "  make verify           - Run correctness verification (vs cuBLAS)"
	@echo "  make clean            - Remove build artifacts"
	@echo "  make help             - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_PATH=$(CUDA_PATH)"
	@echo "  CUTLASS_PATH=$(CUTLASS_PATH)"
	@echo ""
	@echo "Note: Set CUTLASS_PATH if CUTLASS is installed elsewhere"
	@echo "Example: make CUTLASS_PATH=/path/to/cutlass"
