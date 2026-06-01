#include "./spmm_test_utils.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <fstream>
#include <iomanip>
#include "SpMM_API.cuh"

// ---------------------------------------------------------------------------
// 宏
// ---------------------------------------------------------------------------
#define WARM_UP_TIMES 5
#define BENCHMARK_TIMES 5

// ---------------------------------------------------------------------------
// 测试用例结构体
// ---------------------------------------------------------------------------
struct TestCase {
    int M;
    int K;
    int N;
    int sparsity;   // 百分比，例如 80 表示 80% 稀疏
    int Split_K;
};

// ---------------------------------------------------------------------------
// 封装测试函数（返回单个 kernel 的平均耗时 ms）
// ---------------------------------------------------------------------------
static float run_spinfer_kernel(
    cudaError_t (*api_fn)(cudaStream_t,
                          const half*, const half*, const int*, const int*,
                          const uint64_t*, const int*, const half*, half*,
                          int, int, int, half*, int),
    int M_GLOBAL,
    int K_GLOBAL,
    int N_GLOBAL,
    int Split_K,
    half* A_h,
    half* A,
    half* B,
    half* D_out)  // 输出矩阵，由外部分配
{
    // 与原来相同的稀疏矩阵准备过程
    half* Compressed_Val_cpu = nullptr;
    int* bitmap_TileOffsets_cpu = nullptr;
    int* bitmap_TileOffsets_median_cpu = nullptr;
    int* bitmap_TileOffsets_global_cpu = nullptr;
    uint64_t* bitmap_cpu = nullptr;
    int max_nnz_intile = 0;

    auto num_gtiles = InitSparseMatrixA_bitmap_v6(
        A_h, M_GLOBAL, K_GLOBAL, 8, 16, 64, 8, 64, 64,
        &Compressed_Val_cpu, &bitmap_TileOffsets_cpu,
        &bitmap_TileOffsets_median_cpu, &bitmap_TileOffsets_global_cpu,
        &bitmap_cpu, max_nnz_intile);

    int local_tile_num = 8 * 8;
    int median_tile_num = 4 * 1;
    int num_ltiles = num_gtiles * local_tile_num;
    int num_mtiles = num_gtiles * median_tile_num;
    int val_count = bitmap_TileOffsets_global_cpu[num_gtiles];

    // 对齐到 64
    if (max_nnz_intile % 64 != 0) {
        max_nnz_intile = ((max_nnz_intile / 64) + 1) * 64;
    }

    half* Compressed_Val_gpu = nullptr;
    int* bitmap_TileOffsets_gpu = nullptr;
    int* bitmap_TileOffsets_median_gpu = nullptr;
    int* bitmap_TileOffsets_global_gpu = nullptr;
    uint64_t* bitmap_gpu = nullptr;

    cudaMalloc(&bitmap_TileOffsets_gpu, sizeof(int) * (num_ltiles + 1));
    cudaMalloc(&bitmap_gpu, sizeof(uint64_t) * num_ltiles);
    cudaMalloc(&bitmap_TileOffsets_median_gpu, sizeof(int) * num_mtiles);
    cudaMalloc(&bitmap_TileOffsets_global_gpu, sizeof(int) * (num_gtiles + 1));

    if (val_count == 0) val_count = 1;
    cudaMalloc(&Compressed_Val_gpu, sizeof(half) * val_count);

    cudaMemcpy(bitmap_TileOffsets_gpu, bitmap_TileOffsets_cpu, sizeof(int) * (num_ltiles + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_TileOffsets_global_gpu, bitmap_TileOffsets_global_cpu, sizeof(int) * (num_gtiles + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_TileOffsets_median_gpu, bitmap_TileOffsets_median_cpu, sizeof(int) * num_mtiles, cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_gpu, bitmap_cpu, sizeof(uint64_t) * num_ltiles, cudaMemcpyHostToDevice);
    cudaMemcpy(Compressed_Val_gpu, Compressed_Val_cpu, sizeof(half) * val_count, cudaMemcpyHostToDevice);

    // 释放 CPU 临时数据
    free(bitmap_TileOffsets_cpu);
    free(bitmap_TileOffsets_median_cpu);
    free(bitmap_TileOffsets_global_cpu);
    free(bitmap_cpu);
    free(Compressed_Val_cpu);

    half* Reduction_Workspace = nullptr;
    cudaMalloc(&Reduction_Workspace, sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);

    int* max_nnz_intile_gpu = nullptr;
    cudaMalloc(&max_nnz_intile_gpu, sizeof(int));
    cudaMemcpy(max_nnz_intile_gpu, &max_nnz_intile, sizeof(int), cudaMemcpyHostToDevice);

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARM_UP_TIMES; ++i) {
        api_fn(0, A, Compressed_Val_gpu,
               bitmap_TileOffsets_global_gpu, bitmap_TileOffsets_median_gpu,
               bitmap_gpu, max_nnz_intile_gpu, B, D_out,
               M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace, Split_K);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_TIMES; ++i) {
        api_fn(0, A, Compressed_Val_gpu,
               bitmap_TileOffsets_global_gpu, bitmap_TileOffsets_median_gpu,
               bitmap_gpu, max_nnz_intile_gpu, B, D_out,
               M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace, Split_K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= BENCHMARK_TIMES;

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(bitmap_TileOffsets_gpu);
    cudaFree(bitmap_TileOffsets_global_gpu);
    cudaFree(bitmap_TileOffsets_median_gpu);
    cudaFree(bitmap_gpu);
    cudaFree(Compressed_Val_gpu);
    cudaFree(Reduction_Workspace);
    cudaFree(max_nnz_intile_gpu);

    return ms;
}

int main() {
    // 硬编码测试用例
    std::vector<TestCase> test_cases = {
        {8192, 29568, 8, 50, 7},
        {4096, 4096, 8, 70, 7},
        {32000, 5120, 8, 50, 3},
        {32000, 8192, 16, 60, 3},
        {28672, 8192, 16, 50, 4},
        {5120, 5120, 32, 60, 5},
        {18944, 3584, 32, 50, 7},
        {12288, 49152, 32, 70, 6}
    };

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, 0);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    // 存储每个用例的 baseline (v3) 和 optimized (v4_blackwell) 时间，用于计算总体加速比
    std::vector<float> times_v3;
    std::vector<float> times_v4;

    printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n",
           "M", "K", "N", "Sparsity", "SplitK", "v3_ms", "v4_ms", "Speedup");
    
    std::ofstream csv_file("benchmark_results.csv");
    if (!csv_file.is_open()) {
        printf("Failed to open CSV file for writing\n");
        return 1;
    }
    // 写入标题行（逗号分隔）
    csv_file << "M,K,N,Sparsity,SplitK,Baseline_ms,Optimized_ms,Speedup\n";

    for (const auto& tc : test_cases) {
        // 分配 CPU 稠密矩阵
        half* A_h = (half*)malloc(sizeof(half) * tc.M * tc.K);
        half* B_h = (half*)malloc(sizeof(half) * tc.K * tc.N);
        if (!A_h || !B_h) { printf("Host malloc failed\n"); return -1; }

        // 初始化随机稀疏矩阵（函数来自 Flashllm_utils）
        init_host_matrices(A_h, B_h, tc.M, tc.K, tc.N, tc.sparsity);

        // 分配 GPU 稠密矩阵
        half *A, *B;
        cudaMalloc(&A, sizeof(half) * tc.M * tc.K);
        cudaMalloc(&B, sizeof(half) * tc.N * tc.K); // B 是 col-major
        cudaMemcpy(A, A_h, sizeof(half) * tc.M * tc.K, cudaMemcpyHostToDevice);
        cudaMemcpy(B, B_h, sizeof(half) * tc.N * tc.K, cudaMemcpyHostToDevice);

        // 为每个 kernel 的输出分配内存
        half *D_v3, *D_v4;
        cudaMalloc(&D_v3, sizeof(half) * tc.M * tc.N);
        cudaMalloc(&D_v4, sizeof(half) * tc.M * tc.N);
        cudaMemset(D_v3, 0, sizeof(half) * tc.M * tc.N);
        cudaMemset(D_v4, 0, sizeof(half) * tc.M * tc.N);

        // 运行 SpInfer v3 (baseline)
        float ms_v3 = run_spinfer_kernel(
            SpMM_SplitK_API_bitmap_v3,
            tc.M, tc.K, tc.N, tc.Split_K, A_h, A, B, D_v3);

        // 运行 SpInfer v4 blackwell (optimized)
        float ms_v4 = run_spinfer_kernel(
            SpMM_SplitK_API_bitmap_v4,
            tc.M, tc.K, tc.N, tc.Split_K, A_h, A, B, D_v4);

        // 计算加速比
        float speedup = ms_v3 / ms_v4;

        // 记录时间用于总体计算
        times_v3.push_back(ms_v3);
        times_v4.push_back(ms_v4);

        printf("%-10d %-10d %-10d %-10d %-10d %-10.4f %-10.4f %-10.2fx\n",
               tc.M, tc.K, tc.N, tc.sparsity, tc.Split_K,
               ms_v3, ms_v4, speedup);
        
        csv_file << tc.M << "," << tc.K << "," << tc.N << ","
         << tc.sparsity << "," << tc.Split_K << ","
         << std::fixed << std::setprecision(4) << ms_v3 << ","
         << ms_v4 << ","
         << std::setprecision(2) << speedup << "\n";

        // 清理本次迭代的 GPU 内存
        cudaFree(A);
        cudaFree(B);
        cudaFree(D_v3);
        cudaFree(D_v4);
        free(A_h);
        free(B_h);

    }

    // 计算总体加权加速比（总时间之比）
    double total_v3 = std::accumulate(times_v3.begin(), times_v3.end(), 0.0);
    double total_v4 = std::accumulate(times_v4.begin(), times_v4.end(), 0.0);
    double overall_speedup = total_v3 / total_v4;

    printf("\nOverall weighted speedup (SpInfer v4 vs v3): %.2fx\n", overall_speedup);
    csv_file << "\nOverall weighted speedup,,,," << std::fixed << std::setprecision(2) << overall_speedup << "\n";
    csv_file.close();

    cublasDestroy(cublas_handle);
    return 0;
}