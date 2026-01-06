# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUTLASS-based matrix multiplication auto-tuning framework targeting **NVIDIA RTX 3090** with TF32 Tensor Cores. Performs exhaustive search over kernel configurations to find optimal threadblock/warp tile sizes and pipeline stages for 4096×4096 float32 GEMM (C = A × B).

**Target**: 4096×4096×4096 matmul, float32 input/output, TF32 Tensor Core execution
**Achieved**: 45.4 TFLOPS (63.9% of 71 TFLOPS theoretical peak)

## Build Commands

```bash
# Basic auto-tuner (12 hand-picked configs)
make                    # Build
make run               # Run and display results

# Extensive auto-tuner (16 filtered configs, recommended)
python3 autotune.py                              # Generate config combinations
make autotune                                    # Compile generated code
./cutlass_autotune_generated 2>&1 | tee results.csv   # Run with live output

# Correctness verification (compares CUTLASS vs cuBLAS)
make verify            # Build and run verification against cuBLAS reference

# Analysis
python3 analyze_results.py results.csv          # Parse CSV and show top configs

# Cleanup
make clean             # Remove binaries and generated files
```

## Architecture

### Three-Component System

1. **Basic tuner** (`cutlass_matmul_tuning.cu`): 12 hardcoded configurations, minimal compilation time
2. **Extensive tuner** (`autotune.py` → `cutlass_autotune_generated.cu`): Generates C++ with filtered search space
3. **Verification tool** (`verify_correctness.cu`): Compares CUTLASS results against cuBLAS reference to ensure correctness

### Key Constraints (RTX 3090 TF32)

**Architecture tags** (intentional mismatch):
- Compiler flag: `-arch=sm_86` (RTX 3090 hardware capability)
- CUTLASS kernel: `ArchTag = Sm80` (TF32 template works on both SM80/SM86)

**Tile dimension constraints**:
- **ThreadblockK must be 32** (not 64) - TF32 shared memory layout limitation
- Warp tiles must divide threadblock evenly
- Warp dimensions must be multiples of instruction shape (16×8×8 for TF32)

**Resource limits**:
- Shared memory: 102 KB/SM (code uses 80 KB safe limit)
- Formula: `smem ≈ (TB_M*TB_K + TB_K*TB_N) * 4 bytes * stages * 1.5`
- Large threadblocks (128×256+) with many stages (4+) hit resource limits

### Configuration Validation (`autotune.py`)

The `KernelConfig.is_valid()` method filters out configurations that would fail:
- Checks divisibility constraints (threadblock % warp, warp % instruction_shape)
- Estimates shared memory usage with 1.5× safety margin
- Rejects large threadblock areas (>128×128) with stages >3

**Proven warp configs** (guaranteed to compile):
- 32×32×32, 32×64×32, 64×32×32, 64×64×32

### Performance Characteristics

**Best configuration found**:
- Threadblock: 64×128×32
- Warp: 32×32×32
- Stages: 2
- Why: Non-square aspect ratio improves memory bandwidth, fewer stages reduce shared memory pressure

**Performance range**: 24-45 TFLOPS depending on config (basic tuner: 35.5, extensive: 45.4)

## Code Generation Flow

1. `autotune.py` generates combinations of (threadblock_size, warp_size, stages)
2. Filters via `is_valid()` to eliminate resource-violating configs
3. Writes `cutlass_autotune_generated.cu` with `BENCHMARK_CONFIG()` macro calls
4. Compilation instantiates all kernel templates (can take 1-2 minutes)
5. Runtime tests each config, outputs CSV: `tb_m,tb_n,tb_k,warp_m,warp_n,warp_k,stages,time_ms,gflops`

## Customization Points

**Matrix size**: Edit `M`, `N`, `K` constants in `.cu` files (currently 4096×4096×4096)

**Data type**: Change `ElementA/B/C` and `InstructionShape`:
- FP16: `cutlass::half_t`, instruction 16×8×16
- INT8: `int8_t`/`int32_t`, instruction 16×8×32

**Search space**: Modify `generate_search_space()` in `autotune.py`:
- `tb_sizes`: List of (M, N) threadblock dimensions
- `proven_warp_configs`: List of (warp_m, warp_n, warp_k) tuples
- `stage_values`: Pipeline stage counts to try

## Common Issues

**Compilation failures with new configs**: Warp dimensions not compatible with TF32 instruction shape (16×8×8). Stick to proven configs or ensure warp_m % 16 == 0, warp_n % 8 == 0, warp_k % 8 == 0.

**Runtime "FAILED (resource limits)"**: Config exceeds 80 KB shared memory or register limits. This is normal - extensive tuner filters these automatically via `is_valid()`.

**Low performance (<40 TFLOPS)**: Check `nvidia-smi` for thermal throttling or other GPU processes. Smaller threadblocks (64×128) with fewer stages (2-3) generally perform better than large tiles.

## Correctness Verification

The `verify_correctness.cu` program validates CUTLASS results against cuBLAS (NVIDIA's optimized GEMM library):

**What it checks**:
- Runs identical 4096×4096×4096 matmul with same inputs
- Compares ~16.7M output elements element-wise
- Reports absolute and relative errors

**Acceptance criteria** (TF32-appropriate thresholds):
- Max absolute error < 0.01
- Max relative error < 25% (allows outliers near zero)
- Average relative error < 1%
- < 0.1% of elements exceed individual thresholds (abs > 0.001 AND rel > 1%)

**Why relative error can be large**: TF32 uses 10-bit mantissa (vs 23-bit for FP32), giving ~0.1% typical precision. When dividing tiny absolute errors by small reference values near zero, relative error magnifies. Example: abs_error=0.0005, ref_value=0.003 → rel_error=16.7%. The absolute error is what matters for near-zero values.

**Typical results**: Max absolute error ~5e-4, average relative error ~0.005%, indicating excellent correctness.

## Dependencies

- CUDA 11.0+ (for TF32 support and sm_86 target)
- CUTLASS headers (default: `/home/jin/cutlass/include`), override via `make CUTLASS_PATH=/path/to/cutlass`
- cuBLAS library (included with CUDA) for verification
- Python 3 for code generation script
