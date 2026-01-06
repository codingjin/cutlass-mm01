# CUTLASS Matrix Multiplication Auto-Tuning

Auto-tuning framework for CUTLASS matrix multiplication (C = A × B) targeting **NVIDIA GeForce RTX 3090** with TF32 Tensor Cores.

## Overview

- **Operation**: C = A × B (matrix multiplication)
- **Matrix Size**: 4096 × 4096 × 4096
- **Data Type**: float32 (using TF32 Tensor Cores)
- **Target GPU**: NVIDIA GeForce RTX 3090 (Ampere, SM86)
- **Theoretical Peak**: 71 TFLOPS (TF32)
- **Achieved Performance**: **45.4 TFLOPS (63.9% efficiency)**

## Prerequisites

```bash
# CUDA Toolkit 11.0+
# CUTLASS library (https://github.com/NVIDIA/cutlass)

# Install CUTLASS
git clone https://github.com/NVIDIA/cutlass.git /usr/local/cutlass
```

## Project Structure

```
.
├── cutlass_matmul_tuning.cu    # Basic auto-tuner (12 configs)
├── autotune.py                  # Python script to generate extensive search
├── verify_correctness.cu        # Correctness verification vs cuBLAS
├── analyze_results.py           # Results analysis tool
├── Makefile                     # Build system
├── CLAUDE.md                    # Claude Code instructions
└── README.md                    # This file
```

## Quick Start

### Option 1: Basic Auto-Tuning (Fast)

Tests 12 carefully selected configurations:

```bash
make
make run
```

### Option 2: Extensive Auto-Tuning (Recommended)

Generates and tests 16 optimized configurations (filtered to avoid resource limit failures):

```bash
python3 autotune.py       # Generate configurations
make autotune             # Compile
./cutlass_autotune_generated 2>&1 | tee results.csv
```

This creates:
- `results.csv` - Performance data for all configs
- Shows progress in real-time with best configuration summary

### Verify Correctness

Validate CUTLASS results against cuBLAS reference implementation:

```bash
make verify
```

This compares the output of CUTLASS with cuBLAS for the same 4096×4096×4096 matrix multiplication, reporting:
- Maximum and average absolute errors
- Maximum and average relative errors (computed only for significant values)
- Number of elements exceeding thresholds

**Expected results with TF32**: Max absolute error ~5e-4, average relative error ~0.005%

### Analyze Results

```bash
python3 analyze_results.py results.csv
```

Outputs:
- Top 10 configurations
- Optimal configuration
- Statistics by pipeline stages
- Statistics by threadblock size
- Copy-paste ready code template

## Best Configuration Found

For 4096×4096 matrix multiplication on RTX 3090:

```cpp
Threadblock: 64×128×32
Warp: 32×32×32
Pipeline Stages: 2
Performance: 45.4 TFLOPS (63.9% efficiency)
Execution Time: 3.03 ms
```

**Why this works well:**
- Non-square threadblock (64×128) better utilizes memory bandwidth
- Smaller stage count (2) reduces shared memory pressure
- Fits comfortably within 80KB shared memory limit
- Good SM occupancy with multiple warps per threadblock

## Configuration Parameters

### Threadblock Tile
Controls the size of work assigned to each threadblock:
- **M, N dimensions**: 64, 128, 256
- **K dimension**: **32 only** (TF32 constraint)

Larger tiles → Better memory reuse, but more shared memory usage

### Warp Tile
Controls work per warp (32 threads):
- Must evenly divide threadblock tile
- Must be divisible by instruction shape (16×8×8 for TF32)
- **Proven configs**: 32×32×32, 32×64×32, 64×32×32, 64×64×32

### Pipeline Stages
Software pipelining depth (2-5):
- More stages → Better latency hiding
- More stages → **More shared memory usage** (main constraint)
- **Optimal**: 2-3 stages for most configurations

## Performance Results

RTX 3090 measured performance for 4096³ float32 TF32 matmul:
- **Basic tuner (12 configs)**: 35.5 TFLOPS (50.0% efficiency)
- **Extensive tuner (16 configs)**: **45.4 TFLOPS (63.9% efficiency)**
- **Improvement**: +28% performance gain

**Typical range**: 24-45 TFLOPS depending on configuration
**Best achieved**: 45.4 TFLOPS (64% of theoretical 71 TFLOPS peak)

## Customization

### Different Matrix Sizes

Edit in source files:
```cpp
const int M = 4096;  // Your M dimension
const int N = 4096;  // Your N dimension
const int K = 4096;  // Your K dimension
```

### Different Data Types

For FP16:
```cpp
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
```

For INT8:
```cpp
using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
```

### Custom Search Space

Modify `autotune.py`:
```python
def generate_search_space() -> List[KernelConfig]:
    tb_m_values = [64, 128, 256]  # Add your values
    tb_n_values = [64, 128, 256]
    tb_k_values = [32, 64]
    # ...
```

## Build Options

```bash
# Build all tools (basic tuner + verification)
make

# Build extensive auto-tuner
make autotune

# Run basic tuning
make run

# Run verification
make verify

# Set custom CUTLASS path
make CUTLASS_PATH=/path/to/cutlass

# Clean build artifacts
make clean

# Help
make help
```

## Correctness Verification Details

The verification tool compares CUTLASS against cuBLAS with TF32-appropriate error thresholds:

### Why Relative Error Can Be Large

TF32 uses a **10-bit mantissa** (vs 23-bit for full FP32), providing ~3 decimal digits of precision. This means:
- **Typical relative error**: ~0.1% (2^-10)
- **Accumulated error**: Can reach several percent after 4096 accumulations

For values near zero, relative error becomes unreliable:
```
Example: absolute_error = 0.0005, reference_value = 0.003
Relative error = 0.0005 / 0.003 = 16.7%
```

The absolute error (0.0005) is tiny and acceptable, but dividing by a small number makes the relative error look large. This is why the verification tool:
- Only computes relative error for |value| > 0.001
- Accepts max relative error up to 25% (handles edge cases)
- Requires average relative error < 1% (ensures overall quality)

### Verification Pass Criteria

- **Max absolute error** < 0.01 (good absolute precision)
- **Max relative error** < 25% (tolerates outliers near zero)
- **Avg relative error** < 1% (overall quality check)
- **Bad elements** < 0.1% (few elements with abs > 0.001 AND rel > 1%)

### Typical Results

```
Total elements:         16777216
Significant values:     16774723 (|val| > 1e-3)
Max absolute error:     5.289e-04
Max relative error:     18.27% (1-2 outliers near zero)
Avg absolute error:     4.618e-05
Avg relative error:     0.005%
Errors > threshold:     0
Status:                 ✓ VERIFICATION PASSED
```

## Important Notes

### Architecture Tags: SM86 vs SM80

**This is correct and intentional:**
- **Compiler flag**: `-arch=sm_86` (RTX 3090 hardware compute capability)
- **CUTLASS kernel**: `ArchTag = Sm80` (TF32 Tensor Core template)

SM80 TF32 kernels run perfectly on SM86 GPUs. The templates are the same for both.

### Resource Limits

Configurations fail with "resource limits" when they exceed:
- **Shared memory**: 102 KB per SM (we use 80 KB safe limit)
- **Register pressure**: Too many registers per thread
- **Occupancy**: Can't launch enough warps

**Common failures:**
- Large threadblocks (256×256) with many stages (4+)
- K dimension = 64 (use K=32 for TF32)

The auto-tuner filters these automatically.

## Troubleshooting

### Compilation Errors

**Error: CUTLASS headers not found**
```bash
# Set CUTLASS_PATH
export CUTLASS_PATH=/path/to/cutlass
make CUTLASS_PATH=$CUTLASS_PATH
```

**Error: arch=sm_86 not supported**
```bash
# Check CUDA version (need 11.0+)
nvcc --version

# For other GPUs, change arch in Makefile
# sm_86 = Ampere (RTX 30 series - 3090, 3080, etc.)
# sm_80 = Ampere (A100 data center)
# sm_75 = Turing (RTX 20 series)
```

### Runtime Errors

**"FAILED (resource limits)"**
- Configuration exceeds shared memory or register limits
- This is normal and expected for large configurations
- The extensive auto-tuner filters these out automatically

**"FAILED (init error)"**
- Kernel initialization error (rare)
- Usually indicates invalid parameter combination

**Low performance**
- Check GPU is not thermal throttling: `nvidia-smi`
- Ensure no other processes using GPU
- Verify TF32 mode enabled (default in CUDA 11+)
- Try smaller threadblocks with fewer stages

## Understanding Output

```
TB:128x128x32 | W:64x64x32 | S:3 => 2.156 ms | 63847.23 GFLOPS
```

- **TB**: Threadblock tile (M×N×K)
- **W**: Warp tile (M×N×K)
- **S**: Pipeline stages
- **Time**: Average execution time
- **GFLOPS**: Billion floating-point operations per second

## References

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUTLASS Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)
- [Ampere Tensor Cores](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [TF32 Format](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)

## License

This code uses CUTLASS which is licensed under the BSD 3-Clause License.
# cutlass-mm01
