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
// mbarrier PTX wrappers for Hopper+ (sm_90 / sm_100 / sm_120) warp specialization.
// Each mbarrier object occupies 64 bytes in shared memory and must be
// 64-byte aligned.
//
// mbarrier uses a parity-based phase protocol:
//   - init: set expected arrival count, initial parity = 0
//   - arrive: each producer/consumer signals completion (decrements count)
//   - try_wait.parity: polls for parity flip (all arrivals complete)
//   - After parity flip, the phase toggles and the barrier auto-resets
//
// Reference: NVIDIA PTX ISA for sm_100 (mbarrier section).
#ifndef MBARRIER_PTX_CUH
#define MBARRIER_PTX_CUH

#include "TilingConfig.h"

#if __CUDA_ARCH__ >= 900

/// Initialize an mbarrier in shared memory.
/// count: expected arrival count per phase cycle.
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_addr, int count)
{
    unsigned smem_int_ptr = __cvta_generic_to_shared(mbar_addr);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                 :
                 : "r"(smem_int_ptr), "r"(count));
}

/// Signal arrival. Non-blocking. Decrements the expected arrival count.
/// When count reaches 0, the mbarrier parity flips automatically.
__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_addr)
{
    unsigned smem_int_ptr = __cvta_generic_to_shared(mbar_addr);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n"
                 :
                 : "r"(smem_int_ptr));
}

/// Signal arrival with byte-expectation for TMA (producer pre-arrives with
/// expected transfer byte count; TMA hardware completes the transaction).
/// Not used in the current Phase 1 manual-arrive protocol; available for
/// future TMA completion optimization.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar_addr, int tx_bytes)
{
    unsigned smem_int_ptr = __cvta_generic_to_shared(mbar_addr);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                 :
                 : "r"(smem_int_ptr), "r"(tx_bytes));
}

/// Non-blocking parity poll: returns true if the mbarrier has completed
/// all expected arrivals and flipped parity to match `phase_parity`.
/// phase_parity toggles (0→1 or 1→0) each time the barrier completes a
/// full arrival cycle. Callers track this externally.
__device__ __forceinline__ int mbarrier_try_wait_parity(uint64_t* mbar_addr, int phase_parity)
{
    int complete;
    unsigned smem_int_ptr = __cvta_generic_to_shared(mbar_addr);
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\t"
                 "selp.b32 %0, 1, 0, p;\n\t"
                 "}\n"
                 : "=r"(complete)
                 : "r"(smem_int_ptr), "r"(phase_parity));
    return complete;
}

/// Invalidate the mbarrier (reset). Rarely needed in steady-state K-loop
/// since the parity protocol handles auto-reset.
__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar_addr)
{
    unsigned smem_int_ptr = __cvta_generic_to_shared(mbar_addr);
    asm volatile("mbarrier.inval.shared.b64 [%0];\n"
                 :
                 : "r"(smem_int_ptr));
}

#endif  // __CUDA_ARCH__ >= 900
#endif  // MBARRIER_PTX_CUH
