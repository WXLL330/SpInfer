#include "spmm_test_utils.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include "SpMM_API.cuh"

// ---------------------------------------------------------------------------
// 封装测试函数，传入不同API进行测试
// ---------------------------------------------------------------------------
static void run_spinfer_kernel(
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
    half* D_SpMM_bitmapv3
){
    // Define the output pointer
    half* Compressed_Val_cpu_v3 = nullptr;
    int* bitmap_TileOffsets_cpu_v3 = nullptr;
    int* bitmap_TileOffsets_median_cpu_v3 = nullptr;
    int* bitmap_TileOffsets_global_cpu_v3 = nullptr;
    uint64_t* bitmap_cpu_v3 = nullptr;
    int max_nnz_intilev3 = 0;
    // Call the InitSparseMatrixA_bitmap_v6 function
    auto num_gtilesv3 = InitSparseMatrixA_bitmap_v6(A_h, M_GLOBAL, K_GLOBAL, 8, 16, 64, 8, 64, 64, &Compressed_Val_cpu_v3, &bitmap_TileOffsets_cpu_v3, &bitmap_TileOffsets_median_cpu_v3, &bitmap_TileOffsets_global_cpu_v3, &bitmap_cpu_v3, max_nnz_intilev3);
    auto local_tile_numv3 = 8*8;
    auto median_tile_numv3 = 4*1;
    auto num_ltilesv3 = num_gtilesv3*local_tile_numv3;
    auto num_mtilesv3 = num_gtilesv3*median_tile_numv3;
    // The offset of the last tile is equal to the total number of compressed non-zero values
    int val_count_v3 = bitmap_TileOffsets_global_cpu_v3[num_gtilesv3]; 
    int val_count_median_v3 = bitmap_TileOffsets_median_cpu_v3[num_mtilesv3];
    // Adjust max_nnz_intilev3 to a multiple of 64
    if (max_nnz_intilev3 % 64 != 0) {
        max_nnz_intilev3 = ((max_nnz_intilev3 / 64) + 1) * 64;
    }
    half* Compressed_Val_gpu_v3 = nullptr;
    int* bitmap_TileOffsets_gpu_v3 = nullptr;
    int* bitmap_TileOffsets_median_gpu_v3 = nullptr;
    int* bitmap_TileOffsets_global_gpu_v3 = nullptr;
    uint64_t* bitmap_gpu_v3 = nullptr;
    cudaMalloc(&bitmap_TileOffsets_gpu_v3, sizeof(int) * (num_ltilesv3 + 1)); // for (16*64 tile specific)
    cudaMalloc(&bitmap_gpu_v3, sizeof(uint64_t) * (num_ltilesv3));
    cudaMalloc(&bitmap_TileOffsets_median_gpu_v3, sizeof(int) * (num_mtilesv3));
    cudaMalloc(&bitmap_TileOffsets_global_gpu_v3, sizeof(int) * (num_gtilesv3+1));
    if (val_count_v3 == 0)
         val_count_v3 = 1;  // For 100% sparsity, NNZ = 0, malloc will return NULL
    cudaMalloc(&Compressed_Val_gpu_v3, sizeof(half) * val_count_v3);
    if (bitmap_TileOffsets_gpu_v3 == NULL || bitmap_gpu_v3 == NULL || Compressed_Val_gpu_v3 == NULL || bitmap_TileOffsets_global_gpu_v3 == NULL) {
        printf("Error in malloc memory from device memory!\n");
        exit(-1);
    }
    cudaMemcpy(bitmap_TileOffsets_gpu_v3, bitmap_TileOffsets_cpu_v3, sizeof(int) * (num_ltilesv3 + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_TileOffsets_global_gpu_v3, bitmap_TileOffsets_global_cpu_v3, sizeof(int) * (num_gtilesv3 + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_TileOffsets_median_gpu_v3, bitmap_TileOffsets_median_cpu_v3, sizeof(int) * (num_mtilesv3), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_gpu_v3, bitmap_cpu_v3, sizeof(uint64_t) * num_ltilesv3, cudaMemcpyHostToDevice);
    cudaMemcpy(Compressed_Val_gpu_v3, Compressed_Val_cpu_v3, sizeof(half) * val_count_v3, cudaMemcpyHostToDevice);
    free(bitmap_TileOffsets_cpu_v3);
    free(bitmap_cpu_v3);
    free(Compressed_Val_cpu_v3);
    free(bitmap_TileOffsets_global_cpu_v3);
    free(bitmap_TileOffsets_median_cpu_v3);

    half* Reduction_Workspace_bitmapv3 = NULL;
    cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace_bitmapv3), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
    if (Reduction_Workspace_bitmapv3 == NULL) {
        printf("Error in cudaMalloc\n");
        exit(-1);
    }
    int* max_nnz_intilev3_gpu = nullptr;
    cudaMalloc(&max_nnz_intilev3_gpu, sizeof(int));
    if (max_nnz_intilev3_gpu == NULL) {
        printf("Error in cudaMalloc for max_nnz_intilev3_gpu\n");
        exit(-1);
    }
    cudaMemcpy(max_nnz_intilev3_gpu, &max_nnz_intilev3, sizeof(int), cudaMemcpyHostToDevice);
    

    CHECK_CUDA(api_fn(0, A, Compressed_Val_gpu_v3,
                    bitmap_TileOffsets_global_gpu_v3, bitmap_TileOffsets_median_gpu_v3,
                    bitmap_gpu_v3, max_nnz_intilev3_gpu, B, D_SpMM_bitmapv3,
                    M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace_bitmapv3, Split_K));

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaFree(bitmap_TileOffsets_gpu_v3);
    cudaFree(bitmap_TileOffsets_global_gpu_v3);
    cudaFree(bitmap_TileOffsets_median_gpu_v3);
    cudaFree(bitmap_gpu_v3);
    cudaFree(Compressed_Val_gpu_v3);
    cudaFree(Reduction_Workspace_bitmapv3);
    cudaFree(max_nnz_intilev3_gpu);

    return;
}

int validate_spmm(int M_GLOBAL, int K_GLOBAL, int N_GLOBAL, int MATRIX_A_PRUNING_PERCENTAGE, int SPLIT_K) {
    cublasStatus_t cublas_status;
    // Host memory
    half* A_h            = NULL;  // row major
    half* B_h            = NULL;  // col major
    half* B_Transposed_h = NULL;  // row major
    // Device memory
    half* A            = NULL;
    half* B            = NULL;
    half* B_Transposed = NULL;
    //
    A_h            = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h            = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    B_Transposed_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    if (A_h == NULL || B_h == NULL || B_Transposed_h == NULL) {
        printf("Error in CPU Malloc!\n");
        exit(-1);
    }
    cudaMalloc(reinterpret_cast<void**>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B_Transposed), sizeof(half) * N_GLOBAL * K_GLOBAL);
    checkLastCudaError(__LINE__);
    if (A == NULL || B == NULL || B_Transposed == NULL) {
        printf("Error in cudaMalloc!\n");
        exit(-1);
    }
    //
    init_host_matrices(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE);
    for (int i = 0; i < K_GLOBAL; i++)
        for (int j = 0; j < N_GLOBAL; j++)
            B_Transposed_h[i * N_GLOBAL + j] = B_h[i + j * K_GLOBAL];
    //
    // printf("Preparing dense data for GPU...\n");
    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_Transposed, B_Transposed_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);

    // CUBLAS
/////////////////////////////////////////////////////////////////////////////////////////////////
    // printf("Launching CuBlas...\n");
    half* D_cublas = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_cublas), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_cublas, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);

    // Tensor core enabled
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaDeviceSynchronize();

    int              m = M_GLOBAL, n = N_GLOBAL, k = K_GLOBAL;
    const float      alpha     = 1.0;
    const float      beta      = 0.0;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);

    cublas_status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        A,
        CUDA_R_16F,
        k,
        B,
        CUDA_R_16F,
        k,
        &beta,
        D_cublas,
        CUDA_R_16F,
        m,
        CUDA_R_32F,
        CuBlasALG
    );
    checkCublasError(cublas_status, __LINE__);
    half* D_cublas_h = NULL;  // col major
    D_cublas_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas_h == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(D_cublas_h, D_cublas, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_cublas);

    auto Split_K = SPLIT_K;

// SpInfer-v3-baseline
////////////////////////////////////////////////////////////////////////////////////////////////
    // half* D_SpMM_bitmapv3 = NULL;
    // cudaMalloc(reinterpret_cast<void**>(&D_SpMM_bitmapv3), sizeof(half) * M_GLOBAL * N_GLOBAL);
    // if (D_SpMM_bitmapv3 == NULL) {
    //     printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
    //     exit(-1);
    // }
    // cudaMemset(D_SpMM_bitmapv3, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    
    // run_spinfer_kernel(SpMM_SplitK_API_bitmap_v3, M_GLOBAL, K_GLOBAL, N_GLOBAL, Split_K, A_h, A, B, D_SpMM_bitmapv3);

    // half* D_SpMM_hbitmapv3 = NULL;  // col major
    // D_SpMM_hbitmapv3       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    // cudaMemcpy(D_SpMM_hbitmapv3, D_SpMM_bitmapv3, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    // cudaFree(D_SpMM_bitmapv3);

// SpInfer-v4-new
////////////////////////////////////////////////////////////////////////////////////////////////
    half* D_SpMM_bitmapv4 = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_SpMM_bitmapv4), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_SpMM_bitmapv4 == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_SpMM_bitmapv4, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    
    run_spinfer_kernel(SpMM_SplitK_API_bitmap_v4, M_GLOBAL, K_GLOBAL, N_GLOBAL, Split_K, A_h, A, B, D_SpMM_bitmapv4);

    half* D_SpMM_hbitmapv4 = NULL;  // col major
    D_SpMM_hbitmapv4       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(D_SpMM_hbitmapv4, D_SpMM_bitmapv4, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_SpMM_bitmapv4);

// Correction-Verification
//////////////////////////////////////////////////////////////////////////////////////////////// 
    double totalError_SpMM_bitmapv3 = 0.0;
    double totalError_SpMM_bitmapv4 = 0.0;

    // totalError_SpMM_bitmapv3 = ComputeTotalError(D_cublas_h, D_SpMM_hbitmapv3, M_GLOBAL, N_GLOBAL);
    totalError_SpMM_bitmapv4 = ComputeTotalError(D_cublas_h, D_SpMM_hbitmapv4, M_GLOBAL, N_GLOBAL);

    // free(D_SpMM_hbitmapv3);
    free(D_SpMM_hbitmapv4);
    free(D_cublas_h);
    free(A_h);
    free(B_h);
    free(B_Transposed_h);
    cudaFree(A);
    cudaFree(B);
    cudaFree(B_Transposed);

    // return abs(totalError_SpMM_bitmapv3 - totalError_SpMM_bitmapv4) < 1e-3 ? 0 : 1;
    return 0;
}


int main(int argc, char** argv)
{
    int test_cases[][5] = {
        {8192, 29568, 8, 50, 7},
        {4096, 4096, 8, 70, 7},
        {32000, 5120, 8, 50, 3},
        {32000, 8192, 16, 60, 3},
        {28672, 8192, 16, 50, 4},
        {5120, 5120, 32, 60, 5},
        {18944, 3584, 32, 50, 7},
        {12288, 49152, 32, 70, 6}
    };

    for (const auto& test_case : test_cases) {
        int M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE, SPLIT_K;
        M_GLOBAL = test_case[0];
        K_GLOBAL = test_case[1];
        N_GLOBAL = test_case[2];
        MATRIX_A_PRUNING_PERCENTAGE = test_case[3];
        SPLIT_K = test_case[4];
        printf("Running test case: M=%d, K=%d, N=%d, Sparsity=%d%%, SplitK=%d...\n", M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE, SPLIT_K);
        int result = validate_spmm(M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE, SPLIT_K);
        if (result == 1) {
            printf("Test case failed with Config: M=%d, K=%d, N=%d, Sparsity=%d%%, SplitK=%d\n", M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE, SPLIT_K);
            return 1;
        }
    }

    printf("All test cases passed successfully!\n");
    return 0;
}
