# SpMM Kernel v4 — TMA + Warp Specialization (Phase 1)

## Goal Description

Replace the Ampere-era `cp.async` / `__syncthreads()` pipeline in `SpMM_Kernel_bitmap_v3` with a Blackwell-native pipeline in `SpMM_Kernel_bitmap_v4` that uses Tensor Memory Accelerator (TMA) for dense B tile data movement and a producer-consumer warp specialization model with dual `mbarrier` synchronization. Phase 1 prioritizes correctness and a clean baseline design over peak performance. The bitmap-directed sparse A decompression and `mma.sp` compute logic are preserved from v3.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: `SpMM_SplitK_API_bitmap_v4` compiles and links successfully for sm_120 (Blackwell/Pro6000)
  - Positive Tests (expected to PASS):
    - `make SMS=120` in `build/` produces `libSpMM_API.so` containing the v4 kernel symbol
    - Test binary `spmm_test` in `tests/correction_tests/` links against `SpMM_SplitK_API_bitmap_v4` without undefined symbol errors
  - Negative Tests (expected to FAIL):
    - `make SMS=89` (Ada-only) reports an error or produces a stub if TMA instructions are unavailable at that SM target
- AC-2: Validation tests produce bit-exact or tolerance-equivalent results vs v3 baseline
  - Positive Tests (expected to PASS):
    - `bash tests/correction_tests/validate.sh` exits 0, all matrix shapes (N=8,16,32,64,128) match reference within 1e-3 relative tolerance
    - Split-K cases (Split_K=1,2,4) produce identical results to the same N with Split_K=1
    - Sparse patterns at 50% density (2:4 structured sparsity) and 90% density both pass
  - Negative Tests (expected to FAIL):
    - `compute-sanitizer --tool barrier` reports zero "Illegal barrier arrive" or "Missing wait" errors for any validated shape
    - All-zero input matrices do not produce NaN or inf in output
- AC-2.1: No deadlocks or hangs during stress testing
  - Positive: 100 consecutive kernel launches with varying M/K/Split_K complete without GPU timeout
  - Negative: `compute-sanitizer --tool racecheck` reports no data races
- AC-3: Kernel handles all N_Global dispatch paths (N=8,16,32,64,128) supported by v3
  - Positive Tests (expected to PASS):
    - `SpMM_SplitK_API_bitmap_v4` dispatches to the correct template specialization for each N value
    - Each N value produces correct output dimensions matching v3
  - Negative Tests (expected to FAIL):
    - An unsupported N (e.g., N=24) is rejected at the API level with an error, not a silent wrong result
- AC-4: TMA tensor descriptor for B is correctly constructed and validated
  - Positive Tests (expected to PASS):
    - A host-side self-check verifies the descriptor fields: dimensions, strides, element size match B layout (column-major, K_Global stride)
    - The same descriptor works correctly across multiple K tiles within a single kernel launch
  - Negative Tests (expected to FAIL):
    - Descriptor with intentionally wrong stride produces mismatched results that validation catches
- AC-5: mbarrier synchronization replaces `__syncthreads()` in the producer-consumer K-loop; a single `__syncthreads()` is permitted for the final C-store phase only (outside the main pipeline)
  - Positive Tests (expected to PASS):
    - The kernel K-loop body contains zero `__syncthreads()` calls (confirmed by source grep)
    - `compute-sanitizer --tool barrier` confirms all mbarrier arrive/wait pairs in the K-loop are matched
    - The C-store `__syncthreads()` does not participate in any mbarrier arrive/wait protocol
  - Negative Tests (expected to FAIL):
    - Removing a single `mbarrier.arrive` causes consumers to hang in `mbarrier.try_wait` (detected by sanitizer or watchdog timeout)
    - Moving the C-store `__syncthreads()` into the K-loop body causes a barrier divergence error

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes a fully functional `SpMM_Kernel_bitmap_v4` with TMA-based B tile loading, `cp.async.bulk` for sparse A and bitmap loading, warp specialization with 1 producer warp and 4 consumer warps, dual `mbarrier` synchronization (forward: data-ready; reverse: A-consumed) replacing `__syncthreads()` in the K-loop, and a host-side TMA tensor descriptor constructed via raw PTX for the dense B matrix. The kernel supports all N_Global dispatch paths (8/16/32/64/128), Split-K > 1 with the existing reduction kernel, and the N=8 special case. PTX wrappers for `cp.async.bulk.tensor`, `cp.async.bulk`, and `mbarrier` primitives are added to the existing PTX utility headers. The `TilingConfigBitmapV4` struct inherits from v3 and adds warp-specialization constants. Shared memory budget is guarded by compile-time static_assert and runtime fallback. Validation tests pass with tolerance-equivalent results against v3. No performance regression vs v3 on Blackwell for the target workload shapes.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes a correctness-first `SpMM_Kernel_bitmap_v4` targeting sm_120 only. At minimum, the kernel supports a single N_Global path (N=64, the most common case) with 1 producer warp + 4 consumer warps using TMA for B loading and `mbarrier` for synchronization. The sparse A and bitmap loading may reuse the existing v3 `cp.async` path (not `cp.async.bulk`) as a temporary correctness baseline. Split-K may be limited to Split_K=1 initially. The TMA tensor descriptor may be a minimal hand-crafted structure that encodes only the necessary fields for the 2D fp16 copy pattern. The kernel must pass `compute-sanitizer` with zero barrier errors and produce correct results for N=64, M%64==0, K%64==0 matrices.

### Allowed Choices
- Can use: Raw PTX inline assembly for TMA tensor descriptor creation (`tensor.map.to_shared`), `cp.async.bulk.tensor`, `cp.async.bulk`, and `mbarrier` primitives
- Can use: `cudaTensorMap` host API (CUDA 12.8) as an alternative to raw PTX if the PTX path proves difficult
- Can use: Dual mbarrier (forward data-ready + reverse a-consumed) is the required design for correctness; a third mbarrier for B-specific signaling is optional
- Can use: Either 1P/4C or 2P/3C warp split; 1P/4C is the starting point, switch if profiling shows producer starvation
- Can use: Direct inheritance from `TilingConfigBitmapV3` for v4 config or a standalone struct
- Cannot use: Host-side `cudaMemcpy` or synchronous copies inside the kernel loop
- Cannot use: `__syncthreads()` in the producer-consumer pipeline (acceptable only for the final C-store synchronization if mbarrier for that phase proves problematic)
- Cannot use: TMA for sparse A loading (the variable-NNZ nature makes this impractical for Phase 1)
- Cannot use: Blackwell-specific `tcgen05` instructions (out of scope for Phase 1; mma.sp from v3 is sufficient)

> **Note on Deterministic Designs**: If the draft specifies a highly deterministic design with no choices (e.g., "must use JSON format", "must use algorithm X"), then the path boundaries should reflect this narrow constraint. In such cases, upper and lower bounds may converge to the same point, and "Allowed Choices" should explicitly state that the choice is fixed per the draft specification.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The implementation transforms the existing all-warps-identical v3 kernel into a two-role pipeline:

**Producer warp (warp 0):** Runs a K-tile loop issuing asynchronous bulk copies for all data movement. For each iteration: (1) issue `cp.async.bulk` for sparse A into shared memory, (2) issue `cp.async.bulk` for the 64-entry bitmap, (3) issue `cp.async.bulk.tensor` with the B tensor descriptor targeting the double-buffered B region, (4) commit all copy groups, (5) issue `mbarrier.arrive` to signal data readiness to consumers. The producer does no computation — it purely issues copy instructions and advances pointers.

**Consumer warps (warps 1-4):** Run a K-tile loop performing computation. For each iteration: (1) `mbarrier.try_wait` for data arrival, (2) decompress A from shared memory using the bitmap (exact v3 `SpMM_LoadFragAwithBitmapFromShem` logic, unchanged), (3) load B fragments from the correct double-buffer via `ldmatrix`, (4) execute 4 k-step iterations of `mma.m16n8k16` with B fragment prefetching (exact v3 `PipelinedCoreComputationsBitmap` logic), (5) accumulate into FP32 C registers. After the K-loop, store accumulated C to shared memory and write to global memory.

**mbarrier protocol (dual-barrier, forward + reverse):** Two `mbarrier` objects in shared memory (64 bytes each, 64B-aligned):

- `mbar_data_ready`: Producer→Consumer forward barrier. Expected arrival count = 1 (producer). Producer issues `mbarrier.arrive` after all loads (A + bitmap + B TMA) are issued for a tile. Each consumer calls `mbarrier.try_wait` before reading any data from shared memory for that tile. This replaces the v3 `cp.async.wait_group + __syncthreads()` for data arrival.

- `mbar_a_consumed`: Consumer→Producer reverse barrier. Expected arrival count = CONSUMER_WARPS = 4. Each consumer issues `mbarrier.arrive` immediately after `SpMM_LoadFragAwithBitmapFromShem` finishes decompressing A from shared memory into registers (the A/bitmap buffers are no longer needed after this point). The producer calls `mbarrier.try_wait` before overwriting smem_A and smem_Bitmap for the next K-tile. This ensures the producer never overwrites A/bitmap while consumers are reading from them. Note: B is double-buffered and does not need reverse protection — consumers read from a different buffer than the producer writes to.

**Per-tile synchronization sequence:**
```
Producer                          Consumers (×4)
  │                                  │
  ├─ try_wait(mbar_a_consumed) ──┐   │  (skip iteration 0; phase starts ready)
  │  (ensures prev A consumed)   │   │
  ├─ cp.async.bulk A ────────────┤   │
  ├─ cp.async.bulk bitmap ───────┤   │
  ├─ cp.async.bulk.tensor B ─────┤   │
  ├─ arrive(mbar_data_ready) ────┼──▶├─ try_wait(mbar_data_ready)
  │                              │   ├─ SpMM_LoadFragAwithBitmapFromShem
  │                              │   ├─ arrive(mbar_a_consumed) ──────┐
  │                              │   ├─ ldmatrix B from double-buffer │
  │                              │   ├─ mma loop (4 k-steps)          │
  │  (continues to next tile)    │   │                                 │
  │◀─────────────────────────────┼───┘                                 │
  │                              │                                     │
```

The C-store phase uses a single `__syncthreads()` after all consumers have written their registers to shared memory — this is outside the K-loop and does not interact with the mbarrier protocol.

**Phase initialization:** Lane 0 of the producer warp calls `mbarrier.init` on both barriers with their respective expected arrival counts before the K-loop. The producer skips `try_wait(mbar_a_consumed)` on the first iteration (no previous consumers to wait for). After the final iteration, consumers issue one extra `arrive(mbar_a_consumed)` so the producer can exit cleanly.

**TMA descriptor for B:** The B matrix is column-major with dimensions [K_Global, N_Global] and leading dimension K_Global. A 2D TMA tensor descriptor encodes: box size = [TILE_K=64, TILE_N2] (where TILE_N2 depends on N_Global), global strides = [1, K_Global] elements, shared memory strides = [TILE_N2, 1] (row-major layout in smem), element size = 2 bytes (half). The descriptor is constructed via a single PTX `tensor.map.to_shared` instruction in host code and passed as a kernel argument. Within the kernel, the producer warp issues `cp.async.bulk.tensor` with an offset to select the correct K-tile.

**TMA Descriptor Contract (exact specification):**
- Tensor rank: 2
- Global dimensions: `[K_Global, N_Global]` elements
- Global strides: `[1, K_Global]` elements (contiguous in K, stride between N columns)
- Box (tile) dimensions: `[TILE_K=64, TILE_N2]` elements, where TILE_N2 ∈ {8, 16, 32, 64}
- Shared memory strides: `[TILE_N2, 1]` elements (row-major: contiguous in K within smem)
- Element type: `.f16` (2 bytes per element)
- Swizzle mode: `tile.128b` (128-byte swizzle for optimal bank access)
- Alignment requirement: B global pointer must be 128-byte aligned; smem target must be 128-byte aligned
- Host pre-condition: `K_Global % 64 == 0` and `N_Global % TILE_N2 == 0` (enforced by v3's existing dispatch)
- Runtime layout check: before constructing descriptor, verify `B` pointer alignment and dimensions against descriptor constraints; if any constraint fails, fall back to v3 kernel path

**Producer-Consumer Buffer Ownership Contract:**
| Buffer | Producer writes | Consumer reads | Protection |
|--------|---------------|---------------|------------|
| smem_A | start of each K iter (cp.async.bulk) | decompress phase (SpMM_LoadFragAwithBitmapFromShem) | `mbar_a_consumed` reverse barrier: producer waits for consumers before overwriting |
| smem_Bitmap | start of each K iter (cp.async.bulk) | decompress phase (SpMM_LoadFragAwithBitmapFromShem reads bitmap) | Same reverse barrier as smem_A |
| smem_B_buf0 | even K iterations (TMA) | even K iterations (ldmatrix) | Double buffering: no overlap between read and write for same buffer index |
| smem_B_buf1 | odd K iterations (TMA) | odd K iterations (ldmatrix) | Same double-buffering guarantee |

Producer must NOT overwrite smem_A or smem_Bitmap until `mbarrier.try_wait(mbar_a_consumed)` passes. Consumers must NOT access smem_A, smem_Bitmap, or smem_B before `mbarrier.try_wait(mbar_data_ready)` passes. After decompress, smem_A and smem_Bitmap are no longer needed by consumers (all data is in registers), so consumers signal `mbarrier.arrive(mbar_a_consumed)` immediately after decompress completes.

**Shared Memory Budget (compile-time and runtime guard):**
```
SHMEM_BYTES = 2 × 64                          // two mbarrier objects (128 B)
            + max_nnz_intile × sizeof(half)   // compressed A values
            + 2 × TILE_K × TILE_N × sizeof(half)  // B double-buffer
            + 64 × sizeof(uint64_t)           // bitmap (512 B)
            + (TILE_M + 4) × TILE_N2 × sizeof(float)  // C output (reuses space)

// Compile-time assert: A+B+bitmap must fit before C overlay
static_assert(128 + 2304*2 + 2*64*TILE_N*2 + 512 <= SHMEM_SZ, "...");

// Runtime guard: max_nnz_intile must not exceed the budgeted A space
if (max_nnz_intile > 2304) fallback to v3 or adjust SHMEM_SZ;
```

For N=64 (TILE_N=64, TILE_N2=64): SHMEM ≈ 128 + 4608 + 16384 + 512 = 21632 bytes minimum. The C output region (68×64×4 = 17408 bytes) fits within this when overlaid. For N=128 (two tiles of 64): same per-block footprint.

### Relevant References
- `csrc/SpMM_Kernel.cuh` — Baseline v3 kernel with full load-compute-store pipeline to port into consumers
- `csrc/MatMulUtilities.cuh` — `PipelinedCoreComputationsBitmap` (MMA loop), `SpMM_LoadFragAwithBitmapFromShem` (decompress), `CopyTileFromGlobalToShared_X_64` (B copy to replace with TMA), `StoreToSharedMemoryFromRegisterBitmapV3` (C store)
- `csrc/AsyncCopy_PTX.cuh` — Existing `cp.async` PTX wrappers; add `cp.async.bulk` and `cp.async.bulk.tensor` variants
- `csrc/MMA_PTX.cuh` — `B_FragLoadFromSharedToRegisters` (ldmatrix, used unchanged) and `MMA_FP16_M16N8K16` (mma PTX, used unchanged)
- `csrc/TilingConfig.h` — `TilingConfigBitmapV3` struct; extend or replicate for V4
- `csrc/SpMM_API.cu` — `SpMM_SplitK_Kernel_Ex_bitmap_v3` (grid launch parameters to replicate) and `SpMM_SplitK_API_bitmap_v3` (N-dispatch switch to replicate)
- `csrc/Reduction_Kernel.cuh` — `SplitK_Reduction` kernel (used unchanged for Split_K > 1)
- `build/SpMM_API.cuh` — Public API declarations; add v4 entry
- CUDA 12.8 PTX ISA reference — `cp.async.bulk.tensor`, `mbarrier`, `tensor.map` instruction encodings for sm_120

## Dependencies and Sequence

### Milestones

1. **M1: Scaffolding and PTX Infrastructure** — Establish the v4 skeleton so tests compile and link; add all new PTX wrapper instructions
   - Add `TilingConfigBitmapV4` struct to `csrc/TilingConfig.h`
   - Add `SpMM_SplitK_API_bitmap_v4` declaration to `build/SpMM_API.cuh`
   - Add stub `SpMM_SplitK_API_bitmap_v4` in `csrc/SpMM_API.cu` (initially forwarding to v3)
   - Add `cp.async.bulk`, `cp.async.bulk.tensor`, `cp.async.bulk.commit_group`, `cp.async.bulk.wait_group` PTX wrappers to `csrc/AsyncCopy_PTX.cuh`
   - Add `mbarrier.init`, `mbarrier.arrive`, `mbarrier.try_wait`, `mbarrier.inval` PTX wrappers to `csrc/MBarrier_PTX.cuh` (new file)
   - Verify compilation with `make SMS=120` for all new wrappers

2. **M2: TMA B-Loading Path** — Implement and validate TMA-based B tile loading in isolation
   - Implement host-side tensor descriptor creation for B via PTX `tensor.map.to_shared` in the kernel launcher
   - Implement producer-warp logic: TMA B load + cp.async.bulk A/bitmap loads + mbarrier signaling
   - Implement minimal consumer that only verifies B data correctness in shared memory (no MMA yet)
   - Validate B tile data matches v3 for a single K iteration

3. **M3: Producer-Consumer Integration** — Wire the full pipeline end-to-end
   - Port v3 compute logic (decompress + MMA + store) into consumer warps
   - Replace consumer `__syncthreads()` with `mbarrier.try_wait` polling
   - Add K-tile loop with double-buffered B, single-buffered A/bitmap
   - Implement C-store phase with final synchronization
   - Handle last-K-tile edge case (no prefetch copy needed)

4. **M4: Full Dispatch and Edge Cases** — Complete the N_Global dispatch matrix and Split-K support
   - Add per-N_Global template specializations (8/16/32/64/128) in `SpMM_SplitK_API_bitmap_v4`
   - Verify Split-K > 1 path works (reuse existing `SplitK_Reduction` kernel unchanged)
   - Verify N=8 special case (N8=1 flag) works with the specialized load pattern
   - Handle M_Global % TILE_M != 0 boundary (add bounds-checked output store if needed)

5. **M5: Validation and Sanitization** — Prove correctness
   - Run `bash tests/correction_tests/validate.sh` — all shapes pass
   - Run `compute-sanitizer --tool barrier` — zero errors
   - Run `compute-sanitizer --tool racecheck` — zero errors
   - Run stress test: 100 consecutive launches with varying shapes — no hangs

6. **M6: Performance Baseline** — Establish v4 performance characteristics
   - Profile v4 vs v3 with `ncu` on representative shapes (N=64,128; K=4096,8192; Split_K=1,2)
   - Document throughput (TFLOPS), occupancy, and shared memory utilization
   - Identify bottlenecks for Phase 2 optimization
   - Record all profiling data for comparison

Relative dependencies: M1 must complete before M2 (need PTX wrappers). M2 must complete before M3 (need validated TMA path). M3 must complete before M4 (need end-to-end pipeline before handling edge cases). M4 must complete before M5 (need all shapes working before validation). M5 must complete before M6 (need correctness confirmed before performance measurement).

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| t1 | Add `TilingConfigBitmapV4` to `csrc/TilingConfig.h` with producer/consumer constants | AC-1 | coding | - |
| t2 | Add v4 API declaration (`SpMM_SplitK_API_bitmap_v4`) to `build/SpMM_API.cuh` | AC-1 | coding | t1 |
| t3 | Add `cp.async.bulk` and `cp.async.bulk.tensor` PTX wrappers to `csrc/AsyncCopy_PTX.cuh` | AC-1 | coding | - |
| t4 | Create `csrc/MBarrier_PTX.cuh` with `mbarrier.init/arrive/try_wait/inval` PTX wrappers | AC-1, AC-5 | coding | - |
| t5 | Implement host-side TMA tensor descriptor construction via PTX for B matrix | AC-4 | coding | t3 |
| t6 | Implement stub `SpMM_SplitK_API_bitmap_v4` in `csrc/SpMM_API.cu` forwarding to v3; verify compilation | AC-1 | coding | t1, t2 |
| t7 | Implement `SpMM_Kernel_bitmap_v4` with producer warp (TMA B + cp.async.bulk A/bitmap + mbarrier) for N=64 only | AC-2, AC-5 | coding | t3, t4, t5 |
| t8 | Port v3 compute pipeline (decompress + MMA + C-store) into consumer warps with mbarrier.try_wait | AC-2, AC-5 | coding | t7 |
| t9 | Analyze correctness of mbarrier arrive/wait pairing and buffer lifetime across all K iterations | AC-5 | analyze | t7, t8 |
| t10 | Add per-N_Global dispatch (8/16/32/64/128) in `SpMM_SplitK_API_bitmap_v4` | AC-3 | coding | t8 |
| t11 | Verify Split-K > 1 path works with warp specialization (reuse Reduction_Kernel unchanged) | AC-2 | coding | t10 |
| t12 | Handle N=8 special case (N8=1 flag) with consumer ldmatrix specialization | AC-3 | coding | t10 |
| t13 | Run full validation suite and fix all correctness issues | AC-2, AC-2.1 | coding | t11, t12 |
| t14 | Run compute-sanitizer (barrier + racecheck) and fix all reported errors | AC-5, AC-2.1 | coding | t13 |
| t15 | Profile v4 vs v3 baseline with ncu; document performance characteristics | - | analyze | t14 |

## Claude-Codex Deliberation

### Agreements
- Both agree that TMA is appropriate and beneficial for dense B tile loading, and that the B matrix layout (column-major with K_Global stride) maps cleanly to a 2D TMA tensor descriptor without data reorganization
- Both agree that sparse A loading should NOT use TMA in Phase 1 — the variable NNZ per tile makes fixed-box TMA a poor fit; `cp.async.bulk` (or even the existing `cp.async` path) is the right choice for A
- Both agree that `mbarrier` correctness is the highest-risk aspect of the implementation and requires thorough sanitizer validation
- Both agree that preserving the v3 bitmap-directed decompression and mma.sp compute logic unchanged is the correct approach — these are proven, orthogonal to the data-movement changes
- Both agree that a "minimal correctness-first" pipeline proving one tile path end-to-end is essential before adding all N-dispatch paths and Split-K support
- Both agree that Phase 1 is Blackwell-only (sm_120) with no requirement for backward compatibility to Ampere/Ada

### Resolved Disagreements
- **1P/4C vs 2P/3C warp split**: Codex suggested starting with 2P as safer for load balance. Claude maintains 1P/4C because the producer work (issue TMA + cp.async.bulk + arrive) is lightweight and a single warp suffices. Resolution: start with 1P/4C per user decision; if profiling shows producer starvation, the design allows switching to 2P/3C via a template parameter.
- **Introduce mbarrier incrementally vs all-at-once**: Codex suggested keeping `__syncthreads()` initially and introducing mbarrier only in the inner pipeline. Claude maintains that mbarrier must replace all `__syncthreads()` for the producer-consumer model to function — the producer and consumer warps must NOT rendezvous at a block-wide barrier. Resolution: mbarrier replaces all synchronization in the K-loop. A final `__syncthreads()` for C-store coordination is an acceptable fallback per Path Boundaries.
- **TMA descriptor via host API vs raw PTX**: Codex raised concerns about raw PTX descriptor complexity. Claude agrees raw PTX carries risk but prefers it for control and avoiding host-side API coupling. Resolution: raw PTX is the primary approach; `cudaTensorMap` host API is an allowed fallback if PTX proves problematic. The tensor descriptor is validated in M2 via a host-side self-check.

### Round 2: Second Codex Review

**Codex identified 5 REQUIRED_CHANGES:**

1. **Single-mbarrier design → dual-barrier**: Codex correctly identified that a single mbarrier cannot express both "data ready" and "buffer consumed" signals, creating a race condition where the producer could overwrite single-buffered A/bitmap while consumers are still decompressing. Resolution: Adopted dual mbarrier design — `mbar_data_ready` (producer→consumers, expected=1) and `mbar_a_consumed` (consumers→producer, expected=4). This eliminates the data race without requiring double-buffering for A/bitmap. The per-tile synchronization sequence, buffer ownership contract, and phase initialization rules are now specified in the plan.

2. **AC-5 / __syncthreads() contradiction**: The plan claimed mbarrier replaces ALL __syncthreads() while simultaneously allowing a C-store fallback. Resolution: AC-5 updated to explicitly permit a single __syncthreads() for the C-store phase only, which is outside the K-loop and does not participate in mbarrier protocol. The AC now tests that the K-loop body has zero __syncthreads() calls, not the entire kernel.

3. **Missing B descriptor contract**: The draft lacked exact descriptor dimensions, strides, alignment requirements, and failure-mode behavior. Resolution: Added TMA Descriptor Contract subsection specifying rank=2, global dims=[K_Global, N_Global], strides=[1, K_Global], box=[64, TILE_N2], element type=f16, swizzle=tile.128b, alignment=128B, and a runtime fallback to v3 if constraints are violated.

4. **Missing progress contract**: The plan did not specify when each buffer can be read vs written. Resolution: Added Producer-Consumer Buffer Ownership Contract table specifying write/read phases per buffer and which barrier protects each.

5. **Missing shared memory budget**: No compile-time guard against max_nnz_intile exceeding the budgeted A space. Resolution: Added Shared Memory Budget subsection with compile-time static_assert for the fixed portion and a runtime guard comparing max_nnz_intile against the hardcoded 2304 limit, with fallback to v3.

All 5 REQUIRED_CHANGES have been applied. No remaining `DISAGREE` items.

### Convergence Status
- Final Status: `converged`
- Reason: After Round 1 (Codex v1 → Claude revision) and Round 2 (Codex v2 → Claude revision), all REQUIRED_CHANGES are resolved. The dual-barrier design satisfies correctness requirements. The B descriptor contract, buffer ownership contract, and shared memory budget guard close the specification gaps. No unresolved disagreements remain. The remaining Pending User Decisions (DEC-1 through DEC-4) are policy/preference questions that do not block plan soundness.

## Pending User Decisions

- DEC-1: Should v4 be a strict drop-in replacement for v3 on all GPU architectures, or Blackwell-only?
  - Claude Position: Blackwell-only (sm_120).
  - Codex Position: N/A — open question from Codex analysis.
  - Tradeoff Summary: Blackwell-only simplifies the kernel. API dispatches to v3 for non-Blackwell.
  - Decision Status: `RESOLVED: Blackwell-only`
- DEC-2: Must Phase 1 achieve a specific performance speedup over v3, or is correctness parity sufficient?
  - Claude Position: Correctness parity is sufficient. Performance is observational only for Phase 1.
  - Codex Position: N/A — open question from Codex analysis.
  - Tradeoff Summary: User confirmed observational only — measure, document, fix in Phase 2.
  - Decision Status: `RESOLVED: Observational only`
- DEC-3: Is the B matrix layout changeable to suit TMA tensor-map constraints, or must it remain as-is?
  - Claude Position: B layout already maps naturally to TMA descriptor. No change needed.
  - Codex Position: N/A — open question from Codex analysis.
  - Tradeoff Summary: Moot for Phase 1 — existing column-major layout is TMA-compatible.
  - Decision Status: `RESOLVED: No change needed`
- DEC-4: Is 1 producer warp the fixed design, or should the implementation support tunable producer/consumer ratio?
  - Claude Position: 1P/4C fixed design. Tunability deferred to Phase 2 if profiling warrants.
  - Codex Position: Suggested tunable ratio for load balancing.
  - Tradeoff Summary: User chose 1P/4C with 5 total warps (160 threads per block). The plan is updated to reflect BLOCK_WARPS=5 with 1 producer + 4 consumers. Warp 0 is producer, warps 1-4 are consumers.
  - Decision Status: `RESOLVED: 1P/4C fixed`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

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

**Rationale for 1P/4C split**: The producer's job (issuing TMA descriptors, cp.async.bulk calls) is lightweight — one warp is sufficient to keep three compute warps fed. Three consumer warps match v3's compute throughput.

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
    static constexpr int CONSUMER_WARPS = BLOCK_WARPS - PRODUCER_WARPS;  // 4
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

--- Original Design Draft End ---
