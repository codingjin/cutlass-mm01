#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>

// Templated Matrix Multiplication Configuration
template<
    int ThreadblockM,
    int ThreadblockN,
    int ThreadblockK,
    int WarpM,
    int WarpN,
    int WarpK,
    int Stages
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

// Benchmark function for C = A * B
template<typename GemmKernel>
float benchmark_matmul(int M, int N, int K, int iterations = 100, int warmup = 10) {
    using ElementA = typename GemmKernel::ElementA;
    using ElementB = typename GemmKernel::ElementB;
    using ElementC = typename GemmKernel::ElementC;

    cutlass::HostTensor<ElementA, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<ElementB, cutlass::layout::RowMajor> B({K, N});
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C({M, N});

    // Initialize with simple values
    for (int i = 0; i < M * K; i++) A.host_data()[i] = static_cast<ElementA>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) B.host_data()[i] = static_cast<ElementB>(rand()) / RAND_MAX;
    for (int i = 0; i < M * N; i++) C.host_data()[i] = static_cast<ElementC>(0);

    A.sync_device();
    B.sync_device();
    C.sync_device();

    // C = A * B (alpha=1, beta=0)
    float alpha = 1.0f;
    float beta = 0.0f;

    typename GemmKernel::Arguments args{
        {M, N, K},
        {A.device_ref()},
        {B.device_ref()},
        {C.device_ref()},
        {C.device_ref()},
        {alpha, beta}
    };

    GemmKernel gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        // Return error code: -1 = can't implement (resource limits)
        return -1.0f;
    }

    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        // Return error code: -2 = initialization failed
        return -2.0f;
    }

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        gemm_op();
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            return -1.0f;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / iterations;
}

#define BENCHMARK_CONFIG(TBM, TBN, TBK, WM, WN, WK, STAGES) \
    { \
        using Config = MatMulConfig<TBM, TBN, TBK, WM, WN, WK, STAGES>; \
        float time = benchmark_matmul<typename Config::Gemm>(M, N, K); \
        if (time > 0) { \
            double gflops = (2.0 * M * N * K * 1e-9) / (time * 1e-3); \
            printf("TB:%3dx%3dx%2d | W:%2dx%2dx%2d | S:%d => %6.3f ms | %7.0f GFLOPS\n", \
                   TBM, TBN, TBK, WM, WN, WK, STAGES, time, gflops); \
            if (gflops > best_gflops) { \
                best_gflops = gflops; \
                best_tb_m = TBM; best_tb_n = TBN; best_tb_k = TBK; \
                best_w_m = WM; best_w_n = WN; best_w_k = WK; \
                best_stages = STAGES; \
            } \
        } else if (time == -1.0f) { \
            printf("TB:%3dx%3dx%2d | W:%2dx%2dx%2d | S:%d => FAILED (resource limits)\n", \
                   TBM, TBN, TBK, WM, WN, WK, STAGES); \
        } else { \
            printf("TB:%3dx%3dx%2d | W:%2dx%2dx%2d | S:%d => FAILED (init error)\n", \
                   TBM, TBN, TBK, WM, WN, WK, STAGES); \
        } \
    }

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║       CUTLASS Matrix Multiplication Auto-Tuning          ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("Operation: C = A * B\n");
    printf("Dimensions: A[%dx%d] * B[%dx%d] = C[%dx%d]\n", M, K, K, N, M, N);
    printf("Data Type: float32 (TF32 Tensor Cores)\n");
    printf("GPU: NVIDIA GeForce RTX 3090 (Ampere, SM86)\n\n");

    double best_gflops = 0.0;
    int best_tb_m, best_tb_n, best_tb_k;
    int best_w_m, best_w_n, best_w_k;
    int best_stages;

    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║ Threadblock | Warp      | Stages | Time    | Performance ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");

    // Auto-tuning search space (K=32 for TF32 compatibility)
    BENCHMARK_CONFIG(64, 64, 32, 32, 32, 32, 3);
    BENCHMARK_CONFIG(64, 64, 32, 32, 32, 32, 4);
    BENCHMARK_CONFIG(128, 128, 32, 64, 64, 32, 3);
    BENCHMARK_CONFIG(128, 128, 32, 64, 64, 32, 4);
    BENCHMARK_CONFIG(128, 128, 32, 64, 64, 32, 5);
    BENCHMARK_CONFIG(128, 256, 32, 64, 64, 32, 3);
    BENCHMARK_CONFIG(128, 256, 32, 64, 64, 32, 4);
    BENCHMARK_CONFIG(256, 128, 32, 64, 64, 32, 3);
    BENCHMARK_CONFIG(256, 128, 32, 64, 64, 32, 4);
    BENCHMARK_CONFIG(64, 256, 32, 32, 64, 32, 3);
    BENCHMARK_CONFIG(64, 256, 32, 32, 64, 32, 4);
    BENCHMARK_CONFIG(256, 64, 32, 64, 32, 32, 3);

    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                     BEST CONFIGURATION                    ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ Threadblock: %3dx%3dx%2d                                   ║\n", best_tb_m, best_tb_n, best_tb_k);
    printf("║ Warp:        %2dx%2dx%2d                                    ║\n", best_w_m, best_w_n, best_w_k);
    printf("║ Stages:      %d                                            ║\n", best_stages);
    printf("║ Performance: %.0f GFLOPS                                ║\n", best_gflops);
    printf("║ Efficiency:  %.1f%% of peak (71 TFLOPS TF32)            ║\n", (best_gflops / 71000.0) * 100);
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    return 0;
}
