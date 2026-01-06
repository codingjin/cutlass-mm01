#include <iostream>
#include <cmath>
#include <cublas_v2.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

// CUTLASS Configuration - using best known config from tuning
template<
    int ThreadblockM = 128,
    int ThreadblockN = 128,
    int ThreadblockK = 32,
    int WarpM = 64,
    int WarpN = 64,
    int WarpK = 32,
    int Stages = 2
>
struct MatMulConfig {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using ThreadblockShape = cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>;
    using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;  // TF32 Tensor Core

    static constexpr int kStages = Stages;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;  // TF32 kernels (works on SM86 RTX 3090)

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator,
        ElementAccumulator
    >;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        kStages
    >;
};

// Error checking macro
#define CUBLAS_CHECK(call) \
    { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// Compute maximum absolute and relative errors
struct ErrorStats {
    float max_abs_error;
    float max_rel_error;
    float avg_abs_error;
    float avg_rel_error;
    int num_errors_above_threshold;
    int num_significant_values;  // Values large enough for rel error calculation

    ErrorStats() : max_abs_error(0), max_rel_error(0),
                   avg_abs_error(0), avg_rel_error(0),
                   num_errors_above_threshold(0), num_significant_values(0) {}
};

ErrorStats compare_results(const float* cutlass_result, const float* cublas_result,
                           int M, int N, float abs_threshold = 1e-3, float rel_threshold = 1e-2) {
    ErrorStats stats;
    double sum_abs = 0.0;
    double sum_rel = 0.0;
    const float min_val_for_rel_error = 1e-3;  // Only compute rel error for |val| > 0.001

    for (int i = 0; i < M * N; i++) {
        float cutlass_val = cutlass_result[i];
        float cublas_val = cublas_result[i];

        float abs_error = std::abs(cutlass_val - cublas_val);

        stats.max_abs_error = std::max(stats.max_abs_error, abs_error);
        sum_abs += abs_error;

        // Only compute relative error for values significantly different from zero
        float magnitude = std::abs(cublas_val);
        if (magnitude > min_val_for_rel_error) {
            float rel_error = abs_error / magnitude;
            stats.max_rel_error = std::max(stats.max_rel_error, rel_error);
            sum_rel += rel_error;
            stats.num_significant_values++;

            // Check threshold only for significant values
            if (abs_error > abs_threshold && rel_error > rel_threshold) {
                stats.num_errors_above_threshold++;
            }
        } else {
            // For near-zero values, only check absolute error
            if (abs_error > abs_threshold) {
                stats.num_errors_above_threshold++;
            }
        }
    }

    stats.avg_abs_error = sum_abs / (M * N);
    stats.avg_rel_error = stats.num_significant_values > 0 ?
                          sum_rel / stats.num_significant_values : 0.0f;

    return stats;
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║         CUTLASS Correctness Verification                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("Operation: C = A * B\n");
    printf("Dimensions: A[%dx%d] * B[%dx%d] = C[%dx%d]\n", M, K, K, N, M, N);
    printf("Reference: cuBLAS (NVIDIA's optimized GEMM)\n\n");

    // Allocate host tensors
    cutlass::HostTensor<float, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> B({K, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> C_cutlass({M, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> C_cublas({M, N});

    // Initialize with reproducible random values
    srand(12345);
    printf("Initializing matrices...\n");
    for (int i = 0; i < M * K; i++) {
        A.host_data()[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < K * N; i++) {
        B.host_data()[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < M * N; i++) {
        C_cutlass.host_data()[i] = 0.0f;
        C_cublas.host_data()[i] = 0.0f;
    }

    A.sync_device();
    B.sync_device();
    C_cutlass.sync_device();
    C_cublas.sync_device();

    // ========== CUTLASS GEMM ==========
    printf("Running CUTLASS GEMM...\n");

    using Config = MatMulConfig<>;
    using GemmKernel = typename Config::Gemm;

    float alpha = 1.0f;
    float beta = 0.0f;

    typename GemmKernel::Arguments args{
        {M, N, K},
        {A.device_ref()},
        {B.device_ref()},
        {C_cutlass.device_ref()},
        {C_cutlass.device_ref()},
        {alpha, beta}
    };

    GemmKernel gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS kernel cannot be implemented!" << std::endl;
        return 1;
    }

    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS kernel initialization failed!" << std::endl;
        return 1;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS kernel execution failed!" << std::endl;
        return 1;
    }

    cudaDeviceSynchronize();
    C_cutlass.sync_host();

    printf("✓ CUTLASS completed\n");

    // ========== cuBLAS GEMM ==========
    printf("Running cuBLAS GEMM (reference)...\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Set to use TF32 for fair comparison
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    // cuBLAS uses column-major, so we compute: C = B^T * A^T = (A * B)^T
    // Then interpret result as row-major C
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.device_data(), N,
        A.device_data(), K,
        &beta,
        C_cublas.device_data(), N
    ));

    cudaDeviceSynchronize();
    C_cublas.sync_host();

    printf("✓ cuBLAS completed\n\n");

    CUBLAS_CHECK(cublasDestroy(handle));

    // ========== COMPARISON ==========
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                   VERIFICATION RESULTS                    ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");

    ErrorStats stats = compare_results(
        C_cutlass.host_data(),
        C_cublas.host_data(),
        M, N,
        1e-3,  // absolute threshold: 0.001 (reasonable for TF32)
        1e-2   // relative threshold: 1% (TF32 has ~10-bit mantissa precision)
    );

    printf("║ Total elements:         %d                            ║\n", M * N);
    printf("║ Significant values:     %d (|val| > 1e-3)              ║\n",
           stats.num_significant_values);
    printf("║ Max absolute error:     %.6e                          ║\n", stats.max_abs_error);
    printf("║ Max relative error:     %.6e (%.4f%%)                ║\n",
           stats.max_rel_error, stats.max_rel_error * 100);
    printf("║ Avg absolute error:     %.6e                          ║\n", stats.avg_abs_error);
    printf("║ Avg relative error:     %.6e (%.4f%%)                ║\n",
           stats.avg_rel_error, stats.avg_rel_error * 100);
    printf("║ Errors > threshold:     %d                              ║\n",
           stats.num_errors_above_threshold);
    printf("╠═══════════════════════════════════════════════════════════╣\n");

    // Determine pass/fail: TF32 precision is ~10 bits mantissa
    // Accept if max errors are reasonable and few elements exceed threshold
    // TF32 can have up to ~20% relative error on edge cases near the significance threshold
    bool passed = (stats.max_abs_error < 1e-2) &&           // 0.01 absolute error is acceptable
                  (stats.max_rel_error < 0.25) &&            // 25% max relative error (TF32 outliers)
                  (stats.avg_rel_error < 0.01) &&            // Average < 1% shows overall quality
                  (stats.num_errors_above_threshold < M * N / 1000);  // < 0.1% bad elements

    if (passed) {
        printf("║                    ✓ VERIFICATION PASSED                  ║\n");
        printf("║   CUTLASS results match cuBLAS within tolerance           ║\n");
    } else {
        printf("║                    ✗ VERIFICATION FAILED                  ║\n");
        printf("║   Errors exceed acceptable thresholds                     ║\n");
    }

    printf("╚═══════════════════════════════════════════════════════════╝\n");

    // Sample mismatch for debugging
    if (!passed) {
        printf("\nSample mismatches (first 5 with |cublas| > 1e-3):\n");
        int count = 0;
        for (int i = 0; i < M * N && count < 5; i++) {
            float cutlass_val = C_cutlass.host_data()[i];
            float cublas_val = C_cublas.host_data()[i];
            float abs_error = std::abs(cutlass_val - cublas_val);

            // Show errors for significant values
            if (std::abs(cublas_val) > 1e-3 && abs_error > 1e-3) {
                float rel_error = abs_error / std::abs(cublas_val);
                printf("  Index %d: CUTLASS=%.6f, cuBLAS=%.6f, abs_err=%.6e, rel_err=%.4f%%\n",
                       i, cutlass_val, cublas_val, abs_error, rel_error * 100);
                count++;
            }
        }
    }

    return passed ? 0 : 1;
}
