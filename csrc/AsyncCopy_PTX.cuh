/***************************************************************************
 * Copyright 2025 The SpInfer Authors. All rights reserved.
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
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
// Extended from CUTLASS and https://github.com/AlibabaResearch/flash-llm/blob/main/csrc/AsyncCopy_PTX.cuh
template<int SizeInBytes>
__device__ __forceinline__ void cp_async(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}
template<int SizeInBytes>
__device__ __forceinline__ void cp_async(uint32_t* smem_ptr, const uint32_t* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}
template<int SizeInBytes>
__device__ __forceinline__ void cp_async_8(uint32_t* smem_ptr, const uint32_t* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}
template<int SizeInBytes>
__device__ __forceinline__ void cp_async(uint64_t* smem_ptr, const uint64_t* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
__device__ __forceinline__ void cp_async_group_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template<int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// -----------------------------------------------------------------------------
// cp.async.bulk wrappers (sm_120 / Blackwell)
// -----------------------------------------------------------------------------

/// Bulk async copy from global to shared memory (non-tensor, variable size).
/// Replaces cp.async for A (sparse) and bitmap loads on Blackwell.
template<int SizeInBytes>
__device__ __forceinline__ void cp_async_bulk(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert(SizeInBytes >= 16 && SizeInBytes % 16 == 0, "cp.async.bulk size must be a multiple of 16 bytes");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.bulk.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

/// Bulk async copy from global to shared memory (uint64_t variant for bitmap).
template<int SizeInBytes>
__device__ __forceinline__ void cp_async_bulk(uint64_t* smem_ptr, const uint64_t* global_ptr, bool pred_guard = true)
{
    static_assert(SizeInBytes >= 16 && SizeInBytes % 16 == 0, "cp.async.bulk size must be a multiple of 16 bytes");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.bulk.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

/// Commit a group of cp.async.bulk operations. Does not block.
__device__ __forceinline__ void cp_async_bulk_commit_group()
{
    asm volatile("cp.async.bulk.commit_group;\n" ::);
}

/// Blocks until all but <N> previous cp.async.bulk.commit_group ops complete.
template<int N>
__device__ __forceinline__ void cp_async_bulk_wait_group()
{
    asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N));
}

/// TMA 2D tensor load: copies a dense 2D tile from global to shared memory
/// using the hardware tensor map descriptor (128 bytes, passed by address).
/// The tensor map encodes dimensions, strides, swizzle, and element type.
__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    half* smem_ptr, const void* tensor_map_ptr, int tile_k, int tile_n)
{
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    uint64_t tmap_addr    = __cvta_generic_to_global(tensor_map_ptr);

    // tile_k, tile_n are the starting coordinates within the tensor map's global view
    int coords[2] = {tile_k, tile_n};
    asm volatile(
        "cp.async.bulk.tensor.2d.shared.global.tile [%0], [%1, {%2, %3}];\n"
        :
        : "r"(smem_int_ptr), "l"(tmap_addr), "r"(coords[0]), "r"(coords[1])
        : "memory");
}
