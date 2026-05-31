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
// mbarrier PTX wrappers for Blackwell (sm_120) warp specialization.
// Each mbarrier object occupies 64 bytes in shared memory and must be
// 64-byte aligned. The mbarrier object is treated as an opaque uint64_t[8]
// array; all operations use the shared-memory address of the object.

#include "TilingConfig.h"

/// Initialize an mbarrier object in shared memory.
/// count: expected arrival count (typically PRODUCER_WARPS for forward,
///        CONSUMER_WARPS for reverse barrier).
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_addr, int count)
{
    unsigned smem_int_ptr = __cvta_generic_to_shared(mbar_addr);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                 :
                 : "r"(smem_int_ptr), "r"(count));
}

/// Signal arrival at the mbarrier. Non-blocking.
/// Used by producers to signal data readiness (forward barrier) or
/// by consumers to signal A/bitmap consumption complete (reverse barrier).
__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_addr)
{
    unsigned smem_int_ptr = __cvta_generic_to_shared(mbar_addr);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n"
                 :
                 : "r"(smem_int_ptr));
}

/// Non-blocking poll: returns true if the mbarrier has reached the given
/// phase, i.e., all expected arrivals for that phase have occurred.
/// Consumers use this to wait for data readiness (forward barrier).
/// Producer uses this to wait for A consumption (reverse barrier).
__device__ __forceinline__ int mbarrier_try_wait(uint64_t* mbar_addr, int phase)
{
    int complete;
    unsigned smem_int_ptr = __cvta_generic_to_shared(mbar_addr);
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "mbarrier.try_wait.shared.b64 p, [%1], %2;\n\t"
                 "selp.b32 %0, 1, 0, p;\n\t"
                 "}\n"
                 : "=r"(complete)
                 : "r"(smem_int_ptr), "r"(phase));
    return complete;
}

/// Invalidate the mbarrier, resetting it for reuse. Not needed in the
/// steady-state K-loop (phase-based protocol handles reuse automatically).
__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar_addr)
{
    unsigned smem_int_ptr = __cvta_generic_to_shared(mbar_addr);
    asm volatile("mbarrier.inval.shared.b64 [%0];\n"
                 :
                 : "r"(smem_int_ptr));
}
