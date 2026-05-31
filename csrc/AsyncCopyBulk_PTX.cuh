/***************************************************************************
 * Copyright 2025 The SpInfer Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
// cp.async.bulk and TMA PTX wrappers for Hopper+ (sm_90 / sm_100 / sm_120).
// These instructions require .target sm_90 or higher and are NOT compatible
// with the sm_80 cp.async.cg/ca instructions in AsyncCopy_PTX.cuh.
//
// Reference: NVIDIA PTX ISA for sm_100 / sm_90.
#ifndef ASYNCCOPYBULK_PTX_CUH
#define ASYNCCOPYBULK_PTX_CUH

#include "TilingConfig.h"

#if __CUDA_ARCH__ >= 900

// ---------------------------------------------------------------------------
// cp.async.bulk — non-tensor bulk async copy (global → shared)
// Replaces cp.async for A (sparse) and bitmap loads on Hopper+.
// ---------------------------------------------------------------------------

template <int SizeInBytes>
__device__ __forceinline__ void cp_async_bulk(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert(SizeInBytes >= 16 && SizeInBytes % 16 == 0,
                  "cp.async.bulk size must be a multiple of 16 bytes");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.bulk.shared::cluster.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

template <int SizeInBytes>
__device__ __forceinline__ void cp_async_bulk(uint64_t* smem_ptr, const uint64_t* global_ptr, bool pred_guard = true)
{
    static_assert(SizeInBytes >= 16 && SizeInBytes % 16 == 0,
                  "cp.async.bulk size must be a multiple of 16 bytes");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.bulk.shared::cluster.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

__device__ __forceinline__ void cp_async_bulk_commit_group()
{
    asm volatile("cp.async.bulk.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_bulk_wait_group()
{
    asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N));
}

// ---------------------------------------------------------------------------
// cp.async.bulk.tensor — TMA 2D tile load with mbarrier completion
// The tensor map descriptor (CUtensorMap, 128 bytes) is passed by address.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    half* smem_ptr, const void* tensor_map_ptr, int tile_k, int tile_n)
{
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    uint64_t tmap_addr    = __cvta_generic_to_global(tensor_map_ptr);

    int coords[2] = {tile_k, tile_n};
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}];\n"
        :
        : "r"(smem_int_ptr), "l"(tmap_addr), "r"(coords[0]), "r"(coords[1])
        : "memory");
}

#endif  // __CUDA_ARCH__ >= 900
#endif  // ASYNCCOPYBULK_PTX_CUH
