# SpMM Kernel v4 — Phase 1 Implementation Plan Draft

## Goal

Replace the Ampere-era `cp.async` / `__syncthreads()` pipeline in `SpMM_Kernel_bitmap_v3` with a Blackwell-native TMA + warp specialization + mbarrier pipeline in `SpMM_Kernel_bitmap_v4`. Phase 1 prioritizes correctness and clean design over peak performance.

---

## 1. Baseline Analysis (v3 Pipeline)

### 1.1 Current Data Flow

```
Global: SparseA (compressed) │ DenseB │ Bitmap
         cp.async (all 128 threads) │ cp.async (4-warps) │ cp.async (warp 0)
              ▼                      ▼                      ▼
Shared:  smem_A[NNZ]          smem_B[buf0|buf1]    smem_Bitmap[64]
              │                      │                      │
         SpMM_LoadFragAWithBitmap    ldmatrix          (read by decompress)
         (per-warp popcount unpack)    │
              ▼                      ▼
Regs:     a[4][4]              b[N_TENSORS*2][4]
              └────────── mma.m16n8k16 ──────────┘
                              ▼
Regs:                   c[N_TENSORS][8] (FP32)
                              ▼
                    Shared → Global (Reduction_Workspace)
```

### 1.2 v3 Synchronization

- Global→shared copies issued with `cp.async.commit_group` (2 groups: A+bitmap, then B)
- `cp.async.wait_group<N>()` to block until copies complete
- `__syncthreads()` barriers between load phases (all 4 warps synchronize)
- All warps perform identical work: load A, load B, decompress, compute, store

### 1.3 Key v3 Limitations for Blackwell

- `__syncthreads()` is coarse-grained: all warps must rendezvous even when only producer→consumer signaling is needed
- `cp.async` requires manual address computation per thread (SFU pressure)
- No hardware-managed data movement (TMA on Blackwell would offload address generation)
- All-warps-identical model wastes compute capability during load phases

### 1.4 What Stays the Same

- **Bitmap-directed A decompression** (`SpMM_LoadFragAwithBitmapFromShem` / `maskloadingv1`): This is the core innovation and remains unchanged in logic
- **MMA operations**: `mma.m16n8k16` for dense columns, `mma.sp.m16n8k32` for sparse — same PTX instructions, different scheduling
- **Tiling hierarchy**: Global 64×64 tiles, median 16×64, local 8×8 — same data format
- **Split-K reduction**: Same `SplitK_Reduction` kernel (unchanged)
- **API signature**: Same `SpMM_SplitK_API_bitmap_v4` args as v3 (no new arguments)

---

## 2. Target Architecture (v4)

### 2.1 Warp Specialization Model

```
Warp 0 (Producer ×1):                 Warps 1-3 (Consumer ×3):
┌──────────────────────┐              ┌─────────────────────────┐
│ Issue TMA for B tile │              │ Wait mbarrier (A ready) │
│ cp.async.bulk for A  │   mbarrier   │ Decompress A (bitmap)   │
│ cp.async.bulk bitmap │─────────────▶│ Wait mbarrier (B ready) │
│ Signal mbarrier      │              │ ldmatrix B fragments    │
│                      │              │ mma loop (4 k-steps)    │
│ (advance to next)    │              │ Accumulate C registers  │
│ Issue TMA for B      │              │                         │
│ ...                  │              │ Store C to shared       │
└──────────────────────┘              └─────────────────────────┘
```

**Rationale for 1P/3C split**: The producer's job (issuing TMA descriptors, cp.async.bulk calls) is lightweight — one warp is sufficient to keep three compute warps fed. Three consumer warps match v3's compute throughput.

### 2.2 Synchronization with mbarrier

Replace all `__syncthreads()` with mbarrier primitives:

| v3 Pattern | v4 Replacement |
|---|---|
| `__syncthreads()` after A load | Consumers: `mbarrier.try_wait` on A-ready phase |
| `__syncthreads()` after B load | Consumers: `mbarrier.try_wait` on B-ready phase |
| `__syncthreads()` before C store | `mbarrier.arrive` + `mbarrier.try_wait` for store sync |

The mbarrier is initialized in shared memory. The producer does `mbarrier.arrive` after issuing each batch of loads. Consumers do `mbarrier.try_wait` before accessing the data.

### 2.3 TMA for Dense B Tiles

B is the ideal TMA target: dense, regular 2D tiles, repeated across K iterations.

- Create a `cudaTensorMap` (or raw PTX tensor descriptor) in host code, passed via kernel argument
- Producer warp issues `cp.async.bulk.tensor` with the descriptor — hardware computes all addresses
- TMA copies B tile from global→shared with double buffering

### 2.4 cp.async.bulk for Sparse A + Bitmap

A (sparse, variable NNZ) and bitmap (fixed 64 uint64_t per tile) use `cp.async.bulk` (non-tensor variant), keeping the same load granularity as v3 but with the bulk API.

Alternative considered: TMA for bitmap (fixed size, known layout). Could be done in a future phase.

---

## 3. New Files and Changes

### 3.1 New/Modified Files

| File | Action | Purpose |
|---|---|---|
| `csrc/TilingConfig.h` | Add `TilingConfigBitmapV4` | V4 tiling parameters |
| `csrc/SpMM_Kernel.cuh` | Add `SpMM_Kernel_bitmap_v4` | Main v4 kernel |
| `csrc/MatMulUtilities.cuh` | Add TMA/mbarrier helpers | Producer/consumer utilities |
| `csrc/AsyncCopy_PTX.cuh` | Add `cp.async.bulk` wrappers | Bulk async copy PTX |
| `build/SpMM_API.cuh` | Add v4 declaration | Public header |
| `csrc/SpMM_API.cu` | Add `SpMM_SplitK_API_bitmap_v4` | API entry point |

### 3.2 TilingConfigBitmapV4

```cpp
template <int BLOCK_ROW_WARPS_, int BLOCK_COL_WARPS_, int WARP_COL_TENSORS_, int N8_>
struct TilingConfigBitmapV4 : TilingConfigBitmapV3<BLOCK_ROW_WARPS_, BLOCK_COL_WARPS_, WARP_COL_TENSORS_, N8_> {
    // Inherit all v3 dimensions
    // Add v4-specific:
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int CONSUMER_WARPS = BLOCK_WARPS - PRODUCER_WARPS;  // 3
    static constexpr int TMA_LOAD_SIZE = TILE_K * TILE_N * sizeof(half); // B tile bytes
};
```

### 3.3 Shared Memory Layout (v4)

```
Low address:
  mbarrier object (64 bytes, 64B aligned)
  smem_A[0 .. max_nnz_intile-1]                      = Compressed sparse A (half)
  smem_B[buf 0 | buf 1]                               = B double buffer (half)
    - Buffer 0: offset max_nnz_intile
    - Buffer 1: offset max_nnz_intile + TILE_K * TILE_N
  smem_Bitmap[64]                                     = Bitmap (uint64_t)
    - Per-warp window: smem_Bitmap + warp_i * 16

High address (compute phase, reuses A/B space):
  smem_CFrag[TILE_N2][TILE_M + PADDING]              = C output (float)
```

Key change: mbarrier object at start of shared memory (needs to be pre-initialized).

### 3.4 Host-Side TMA Descriptor Setup

In `SpMM_SplitK_Kernel_Ex_bitmap_v4` (or inline in the API function):
1. Allocate a `cudaTensorMap` for B on the host
2. Set rank=2, dimensions=[TILE_K, K_Global] (row-major view for tile iteration)
3. Pass the descriptor (128 bytes) as a kernel argument

The descriptor encodes the global→shared copy pattern so the producer warp doesn't compute addresses — it just issues `cp.async.bulk.tensor` with the descriptor and an offset.

---

## 4. Kernel Pseudocode

### 4.1 SpMM_Kernel_bitmap_v4

```cpp
__global__ void SpMM_Kernel_bitmap_v4(
    half* A, half* Compressed_A, int* TileOffsets, int* TileOffsets_Median,
    uint64_t* bitmap, int* max_nnz_intile,
    half* B, half* C, int M, int N, int K,
    half* Reduction_Workspace, int Split_K,
    cudaTensorMap* B_tensor_map   // NEW: TMA descriptor for B
) {
    extern __shared__ char smem[];

    // --- Shared memory partitioning ---
    mbarrier_t* mbar = reinterpret_cast<mbarrier_t*>(smem);
    half* smem_A = (half*)(smem + MBARRIER_SZ);
    half* smem_B_buf0 = smem_A + max_nnz_intile;
    half* smem_B_buf1 = smem_B_buf0 + TILE_K * TILE_N;
    uint64_t* smem_Bitmap = (uint64_t*)(smem_B_buf1 + TILE_K * TILE_N);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // --- Warp role dispatch ---
    if (warp_id < PRODUCER_WARPS) {
        producer<Config>(smem, mbar, ...);
    } else {
        consumer<Config>(smem, mbar, warp_id - PRODUCER_WARPS, ...);
    }
}
```

### 4.2 Producer Warp (Warp 0)

```
producer():
    // Initialize mbarrier (warp 0, lane 0 only)
    if (lane_id == 0):
        mbarrier.init(mbar, 1);  // 1 producer, expect 3 consumers

    for k_tile in 0..NumIter:
        int buf_idx = k_tile % 2;

        // Load A (sparse) — all 32 lanes participate
        nnz = TileOffsets_ThisBlock[k_tile+1] - TileOffsets_ThisBlock[k_tile];
        cp.async.bulk(smem_A, Compressed_A + offset, nnz * sizeof(half));
        cp.async.bulk.commit_group();

        // Load Bitmap — all 32 lanes
        cp.async.bulk(smem_Bitmap, bitmap + k_tile * 64, 64 * sizeof(uint64_t));
        cp.async.bulk.commit_group();

        // Load B via TMA — single TMA instruction for entire 2D tile
        cp.async.bulk.tensor(smem_B_buf[buf_idx], B_tensor_map, k_tile * TILE_K);
        // TMA commit is implicit with bulk.tensor

        // Signal consumers: data for this k_tile is ready
        mbarrier.arrive(mbar);

    // Emit final barrier signal (arrive for remaining consumer warps if needed)
```

### 4.3 Consumer Warp (Warps 1, 2, 3)

```
consumer(consumer_id):
    float c[WARP_COL_TENSORS][8];  // FP32 accumulators
    zero_init(c);

    for k_tile in 0..NumIter:
        // Wait for data to be ready
        mbarrier.try_wait(mbar, phase);

        int buf_idx = k_tile % 2;
        int read_buf = smem_B_buf[buf_idx];

        // Decompress A from shared memory (identical to v3)
        SpMM_LoadFragAwithBitmapFromShem(a, smem_A + warp_offset, smem_Bitmap + consumer_id * 16);

        // B fragments via ldmatrix (identical to v3)
        for each k_step in 0..3:
            B_FragLoadFromSharedToRegisters(b, read_buf, ...);

        // MMA loop (identical to v3 pipeline)
        for k_step in 0..3:
            for col_tensor in 0..WARP_COL_TENSORS:
                mma.m16n8k16(c[col_tensor], a[k_step], b[col_tensor]);

    // Store C to shared memory, then global
    StoreToSharedMemoryFromRegister(smem_CFrag, c);
    // Use mbarrier or __syncthreads for store synchronization
```

### 4.4 Double Buffering Summary

| k_tile | Producer writes B to | Consumer reads B from | Producer loads A to |
|--------|---------------------|----------------------|-------------------|
| 0      | smem_B_buf0         | smem_B_buf0          | smem_A            |
| 1      | smem_B_buf1         | smem_B_buf1          | smem_A (overwrite) |
| 2      | smem_B_buf0         | smem_B_buf0          | smem_A (overwrite) |

B is double-buffered. A and Bitmap are single-buffered (overwritten each iteration) — same as v3. This works because the consumer consumes A+Bitmap immediately after waiting on the barrier, before the producer can overwrite them for the next iteration.

---

## 5. PTX Instructions to Add

### 5.1 TMA Load (cp.async.bulk.tensor)

```
cp.async.bulk.tensor.dim.2.shared.global.read [dst], [tensor_map, offset], [mbar]
```

This single instruction replaces the entire `CopyTileFromGlobalToShared_X_64` function for B loading. The tensor_map encodes stride, boundary, and format info. The `[mbar]` operand auto-increments the mbarrier.

### 5.2 Non-Tensor Bulk Copy (cp.async.bulk)

```
cp.async.bulk.shared.global [dst], [src], size_bytes;
cp.async.bulk.commit_group;
cp.async.bulk.wait_group N;
```

Replaces `cp.async.cg.shared.global` + `cp.async.commit_group` + `cp.async.wait_group`.

### 5.3 mbarrier Primitives

```
mbarrier.init.shared.b64 [mbar], count;       // Initialize
mbarrier.arrive.shared.b64 _, [mbar];          // Producer signals
mbarrier.try_wait.shared.b64 p, [mbar], phase; // Consumer polls (non-blocking)
mbarrier.inval.shared.b64 [mbar];              // Reset for next use
```

### 5.4 Tensor Descriptor Creation (host-side PTX)

```cpp
// Construct a 2D tensor descriptor for B
// Fields: tensor_dim (TILE_K, N_Global), global_stride (N, 1), 
//         smem_stride (TILE_N, 1), element_size (2 bytes for half)
asm volatile(
    "tensor.map.to_shared.tile.128.b16384.b1024.b64.v2.f16 "
    "[%0], [%1], ..."
    :: "l"(&tensor_map), "l"(B_ptr), ...
);
```

---

## 6. Implementation Order

### Step 1: Scaffolding (no functional changes)
- Add `TilingConfigBitmapV4` to `TilingConfig.h`
- Add `SpMM_Kernel_bitmap_v4` declaration to `build/SpMM_API.cuh`
- Add stub `SpMM_SplitK_API_bitmap_v4` to `csrc/SpMM_API.cu` that just calls v3
- Verify the test/benchmark files compile and link

### Step 2: PTX Wrappers
- Add `cp_async_bulk`, `cp_async_bulk_tensor`, `cp_async_bulk_commit`, `cp_async_bulk_wait` to `AsyncCopy_PTX.cuh`
- Add `mbarrier_init`, `mbarrier_arrive`, `mbarrier_try_wait` to a new `MBarrier_PTX.cuh` or to existing file
- Test each PTX wrapper compiles for sm_120

### Step 3: Host-Side TMA Descriptor
- Implement tensor descriptor creation for B in `SpMM_SplitK_Kernel_Ex_bitmap_v4`
- Verify descriptor layout matches PTX spec for 2D fp16 tensor

### Step 4: Producer Warp (data movement only)
- Implement producer logic: TMA for B, cp.async.bulk for A+bitmap, mbarrier signaling
- Test in isolation: verify data arrives correctly in shared memory

### Step 5: Consumer Warp (compute only)
- Port the v3 compute pipeline (decompress + MMA + store) to consumer-only context
- Replace `__syncthreads()` waits with `mbarrier.try_wait`
- Verify compute produces correct results for a single K iteration

### Step 6: Full Kernel Integration
- Wire producer + consumer into `SpMM_Kernel_bitmap_v4`
- Add per-N_Global dispatch in `SpMM_SplitK_API_bitmap_v4`
- Handle edge cases: last K tile, Split-K > 1, N=8 special case

### Step 7: Correctness Validation
- Run `bash tests/correction_tests/validate.sh`
- Run `compute-sanitizer` for barrier errors
- Fix all correctness issues before optimizing

### Step 8: Performance Baseline
- Profile with `ncu` — compare v4 vs v3
- Identify bottlenecks (producer starvation? TMA latency? mbarrier overhead?)
- Document for Phase 2 optimization

---

## 7. Open Design Questions

1. **TMA descriptor: host API vs raw PTX?** CUDA 12.8 provides both `cudaTensorMap` API and PTX `tensor.map` instructions. PTX gives more control and avoids host-side CUDA API dependency, but is harder to get right. *Recommendation: start with raw PTX for maximum control.*

2. **A-load: TMA or cp.async.bulk?** A is sparse with variable NNZ per tile — TMA's fixed-size 2D load doesn't map well to sparse data. However, the bitmap (fixed 64 uint64_t) could benefit from TMA. *Recommendation: Phase 1 uses cp.async.bulk for A. Phase 2 evaluates TMA for bitmap.*

3. **mbarrier signaling granularity.** Should we use one mbarrier for all data (A+bitmap+B) or separate barriers per resource? One barrier is simpler. *Recommendation: single mbarrier for Phase 1.*

4. **Pipeline depth.** v3 uses depth-2 double buffering (compute tile N while loading tile N+1). With warp specialization, producer can get 1-2 tiles ahead of consumers. *Recommendation: start with depth 1 (producer loads tile N+1 while consumers compute tile N), same as v3.*

5. **Consumer warp count.** With 1 producer + 3 consumers, do we have enough producer bandwidth? The producer issues TMA (no address computation, just issue) and cp.async.bulk — this is very lightweight. *Recommendation: 1 producer is sufficient for Phase 1. Profile in Phase 2.*

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| TMA descriptor incorrect → silent wrong data | High | Validate with known dense GEMM reference first |
| mbarrier deadlock → kernel hang | High | Use compute-sanitizer; add timeout detection |
| Producer slower than 3 consumers → compute starvation | Medium | Profile; can increase producer warps or pipeline depth |
| N=8 special case (N8=1) breaks with new load pattern | Medium | Keep v3 N=8 handling, validate early |
| TMA on sparse A causes wasted bandwidth | Low | Use cp.async.bulk for A (not TMA) |
