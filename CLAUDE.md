# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpInfer is a EuroSys'25 artifact that accelerates sparse matrix multiplication (SpMM) for LLM inference on NVIDIA GPUs (Ampere+). It uses a bitmap-based sparse storage format and hand-tuned PTX-level Tensor Core kernels, building on ideas from Flash-LLM but replacing metadata-driven sparse loads with bitmap popcount decoding.

**Target hardware:** sm_80+ (Ampere A6000, Ada RTX 4090). Default compile target is sm_89.

## Build & Run Commands

### Environment setup (one-time)
```bash
conda env create -f spinfer.yml && conda activate spinfer
source Init_SpInfer.sh
cd $SpInfer_HOME/third_party/FasterTransformer && git apply ../ft_spinfer.patch
cd $SpInfer_HOME/third_party/sputnik && git apply ../sputnik.patch
```

### Build the core SpMM library
```bash
cd $SpInfer_HOME/build && make -j
```
Produces `libSpMM_API.so` and `SpMM_API.cuh` — the shared library and header for integration.

### Build & run kernel benchmarks (Figure 10 reproduction)
```bash
# Build dependencies first
cd $SpInfer_HOME/third_party/ && source build_sputnik.sh && source preparse_cusparselt.sh
# Then build and run benchmarks
cd $SpInfer_HOME/kernel_benchmark && source test_env && make -j && source benchmark.sh
```
Results: raw CSV files + `Figure10.png` in the `kernel_benchmark/` directory.

### Build end-to-end FasterTransformer with SpInfer
```bash
cd $SpInfer_HOME/third_party/FasterTransformer/ && mkdir -p build && cd build
cmake -DSM=89 -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON -DSpInfer=ON ..
make -j
```
Replace `-DSpInfer=ON` with `-DFLASH_LLM=ON` or `-DFLASH_LLM=OFF` to build Flash-LLM or standard cuBLAS variants.

### Run end-to-end inference
```bash
cd $SpInfer_HOME/third_party/
bash run_1gpu_loop.sh   # 1-GPU
bash run_2gpu_loop.sh   # 2-GPU tensor parallelism
bash run_4gpu_loop.sh   # 4-GPU tensor parallelism
```

### Override GPU architecture
```bash
make SMS=80 -j   # For Ampere A6000 instead of Ada
```

## Architecture

### Core sparse format: bitmap-based compression

The matrix A (M×K weight matrix) is compressed into a 3-level tiling hierarchy:

1. **Global tiles** (64×64): top-level partitioning with offset-based indexing
2. **Median tiles** (16×64): intermediate grouping within global tiles
3. **Local tiles** (8×8): atomic tiling unit — each 8×8 tile stores a `uint64_t` bitmap (1 bit per element), with non-zero values packed contiguously

The compression function `InitSparseMatrixA_bitmap()` in `csrc/SpMM_API.cu` traverses the dense matrix, builds all three levels, and writes binary files. The kernel uses popcount (`__popcll`) on bitmaps to decode compressed values on-the-fly during shared memory loads.

### Kernel pipeline (`csrc/SpMM_Kernel.cuh`)

`SpMM_Kernel_bitmap_v3` is the main kernel. Execution flow per threadblock:

1. **Load bitmap** from global → shared memory (cp.async)
2. **Load compressed A values** from global → shared memory
3. **Load B tile** from global → shared memory (double-buffered)
4. **Decode A fragments from shared**: for each 8×8 sub-tile, `maskloadingv1()` uses popcount on the bitmap to extract non-zero values, zero-fills the rest, and packs into `half2` register fragments
5. **Tensor Core MMA loop**: iterates over K-dimension tiles, double-buffering B in shared memory while computing `mma.sync.m16n8k16` fused with sparse A loading
6. **Split-K reduction**: when `Split_K > 1`, partial results are accumulated in a workspace and reduced via `SplitK_Reduction` kernel

### Tiling configuration (compile-time polymorphism)

`TilingConfig.h` defines three bitmap kernel variants via template structs:

| Variant | `TILE_M` | `TILE_BITMAP_M` | `TILE_BITMAP_K` | Use case |
|---------|----------|-----------------|-----------------|----------|
| `TilingConfigBitmapV1` | 16 | 1 | 16 | Small tiles, dense-like |
| `TilingConfigBitmapV2` | 64 | 1 | 64 | Medium tiles |
| `TilingConfigBitmapV3` | 16 | 1 | 64 | Default; used in published results |

The API in `SpMM_API.cu` dispatches to different `TilingConfigBitmapV3` template instantiations based on N_Global (8, 16, 32, 64, 128, or multiples of 128).

### Source file map

| File | Role |
|------|------|
| `csrc/SpMM_API.cu` | Host-side API: sparse conversion (`Our_GenSparseMatrixBinFile`), kernel launch dispatcher (`SpMM_SplitK_API_bitmap_v3`) |
| `csrc/SpMM_Kernel.cuh` | GPU kernel: `SpMM_Kernel_bitmap_v3`, bitmap-decoded A fragment loading |
| `csrc/TilingConfig.h` | Compile-time tile dimensions; three bitmap config structs + base `TilingConfig` |
| `csrc/MatMulUtilities.cuh` | Async copy helpers, pipelined MMA core, store-registers-to-shmem |
| `csrc/MMA_PTX.cuh` | PTX wrappers: `ldmatrix`, `mma.sync.m16n8k16`, `mma.sp.sync.m16n8k32` |
| `csrc/AsyncCopy_PTX.cuh` | PTX wrappers: `cp.async` (cg/ca variants), `cp.async.commit_group`, `cp.async.wait_group` |
| `csrc/Reduction_Kernel.cuh` | Split-K partial sum reduction kernel |

### Integration points

- **FasterTransformer**: Patched via `third_party/ft_spinfer.patch` to call `libSpMM_API.so` for sparse linear layers in OPT models
- **Model conversion**: `end2end_inference/ft_tools/huggingface_opt_convert_Phase2.py` converts pruned OPT weights to SpInfer's bitmap-based binary format
- **DeepSpeed**: `end2end_inference/ds_scripts/` contains standalone DeepSpeed inference scripts for comparison

### Kernel benchmark structure

`kernel_benchmark/` builds four separate test binaries, each linking a different SpMM backend:
- `spmm_test` — SpInfer (links `libSpMM_API.so`)
- `spmm_test_sparta` — SparTA/cuSPARSELt
- `spmm_test_sputnik` — Google Sputnik
- `spmm_test_cusparse` — cuSPARSE

All share `spmm_test_utils.h` for data generation and validation. `sparTA.h` provides the SparTA wrapper API.

### Key constants

- `MMA_M=16, MMA_N=16, MMA_K=16` — Tensor Core tile dimensions
- `TILE_K=64` — K-dimension tile size (4 MMA_K steps)
- `COPY_UNIT_FP16_ROWS=8, COPY_UNIT_FP16_COLS=64` — async copy granularity
- `REG_PER_C_TENSOR_16_16=8` — FP32 accumulation registers per 16×16 output tile
- `WARP_SIZE=32`
- `PADDING_SHARED_MEM_FOR_C=4` — column padding to avoid bank conflicts

## Third-party dependencies

- **FasterTransformer** (NVIDIA, pinned commit): patched to integrate SpInfer sparse matmul
- **Sputnik** (Google Research): baseline SpMM library, patched for build compatibility
- **glog** (Google): logging, used by Sputnik
- **cuSPARSELt**: NVIDIA's sparse matmul library, used as a baseline (not a submodule — must be installed separately)
