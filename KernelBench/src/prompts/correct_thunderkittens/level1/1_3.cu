// ThunderKittens warpgroup-level fp16 *batched* matmul for H100 (half inputs/outputs)
/*
BATCHED FIX — WHAT CHANGED AND WHY

This kernel is a standard batched GEMM:
    C[b, m, n] = sum_k A[b, m, k] * B[b, k, n]
computed with ThunderKittens warpgroup MMA and a double-buffered TMA pipeline.
The math/algorithm (producer/consumer warpgroups, TMA prefetch, MMA, final store) is unchanged.

The failures you saw (“Depth dimension mismatch. Expected: 1, Got: <batch>”) were caused by
how the batch dimension was declared and indexed in the ThunderKittens global views (gl<>).

TK’s gl<> logical shape is (B, H, M, N) around a tile type. In the original, the second
logical dimension (H, often used as “heads/depth”) was hard-coded to 1. The runtime however
placed the batch size there (e.g., 128), so TK rightfully rejected it.

Fixes (minimal, surgical):

1) Make the batch dimension runtime-resolved on the SECOND logical axis of gl<>,
   and index it explicitly at load/store time:
   - BEFORE: using tile_gl = kittens::gl<kittens::half, 1, 1, -1, -1, sub_tile>;
   - AFTER : using tile_gl = kittens::gl<kittens::half, 1, -1, -1, -1, sub_tile>;
             // B=1, H=-1 (runtime), M=-1, N=-1
   Rationale: your evaluator provided batch in the “depth/heads” slot (H). Declaring H=-1
   makes that dimension match the runtime batch. This removes the “depth mismatch” error.

2) Launch one z-slice per batch and thread it through all memory ops:
   - grid() -> dim3(gx, gy, batch);   // grid.z = batch
   - In-kernel: const int b = blockIdx.z;
   - All TMA loads and the final store include b in the coordinate tuple:
       kittens::tma::load_async(As[...], g.A, {0, b, row, kt}, bar);
       kittens::tma::load_async(Bs[...], g.B, {0, b, kt, col}, bar);
       kittens::warpgroup::store(g.C, C_accum, {0, b, row, col});

3) (Kept correct tile indexing semantics.) TK’s last two coordinates are TILE INDICES, not
   element offsets. We continue to use:
       row = blockIdx.y;  col = blockIdx.x;   // no *BLOCK_SIZE needed
   This matches your working non-batched kernel and avoids off-by-(tile size) OOB errors.

4) Everything else stays the same:
   - Same BLOCK_SIZE, NUM_THREADS, warpgroup producer/consumer split.
   - Same double-buffered shared tiles (As/Bs), semaphore, and TMA expect/load pattern.
   - Same register tile shapes and float accumulator.
   - Same store path (optionally switch to masked stores if M,N aren’t multiples of BLOCK_SIZE).

Summary:
- It’s still standard GEMM (C = A × B) per batch. 
- The key change is *where* the batch lives in gl<> (H made runtime, H indexed with b), plus
  a 3D grid with grid.z = batch and the added b in every load/store coordinate.
- With those, TK’s view aligns with the actual tensor layout, eliminating the depth mismatch
  and keeping the rest of your optimized warpgroup pipeline intact.
*/

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define BLOCK_SIZE 64
#define NUM_WORKERS 8
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct micro_globals {
    using sub_tile = kittens::st_hf<BLOCK_SIZE, BLOCK_SIZE>;
    // Batch is passed in the second logical dim (depth/heads) at runtime:
    //            dtype,  B,  H,  M,  N, sub_tile
    using tile_gl  = kittens::gl<kittens::half, 1, -1, -1, -1, sub_tile>;

    tile_gl A, B, C;
    int batch;   // NEW: runtime batch size
    int M, K, N;

    dim3 grid() const {
        return dim3(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE,   // tiles across N
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE,   // tiles down   M
            batch                                // one grid slice per batch
        );
    }
    dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
    size_t dynamic_shared_memory() const { return 100000; } // keep your setting
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    extern __shared__ kittens::alignment_dummy __shm[];
    // keep your allocator style to match the working kernel
    kittens::shared_allocator al((int*)&__shm[0]);

    // Double-buffered shared tiles
    kittens::st_hf<BLOCK_SIZE, BLOCK_SIZE> (&As)[2] = al.allocate<kittens::st_hf<BLOCK_SIZE, BLOCK_SIZE>, 2>();
    kittens::st_hf<BLOCK_SIZE, BLOCK_SIZE> (&Bs)[2] = al.allocate<kittens::st_hf<BLOCK_SIZE, BLOCK_SIZE>, 2>();

    int tic = 0, toc = 1;

    // One 16x64 register tile per consumer warpgroup (same as your working code)
    kittens::rt_fl<16, BLOCK_SIZE> C_accum;

    // Tile indices (NOT element offsets)
    const int row = blockIdx.y;
    const int col = blockIdx.x;
    const int b   = blockIdx.z;  // NEW: batch index lives in the 2nd logical dim of gl<>

    const int warpid      = kittens::warpid();
    const int warpgroupid = warpid / 4;

    __shared__ kittens::semaphore bar;
    if (threadIdx.x == 0) {
        kittens::init_semaphore(bar, 0, 1);
        kittens::tma::expect_bytes(
            bar,
            kittens::size_bytes<decltype(As[0])> +
            kittens::size_bytes<decltype(Bs[0])>
        );
        // Initial prefetch for this batch slice
        kittens::tma::load_async(As[tic], g.A, {0, b, row, 0},  bar);
        kittens::tma::load_async(Bs[tic], g.B, {0, b, 0,   col}, bar);
    }
    __syncthreads();

    kittens::warp::zero(C_accum);

    const int k_tiles = (g.K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < k_tiles; ++tile, tic ^= 1, toc ^= 1) {
        // Arrive memory
        kittens::wait(bar, tic);
        __syncthreads();

        if (warpgroupid == 0) {
            // Producer: schedule next prefetch
            kittens::warpgroup::decrease_registers<32>();
            if (threadIdx.x == 0 && tile + 1 < k_tiles) {
                kittens::tma::expect_bytes(
                    bar,
                    kittens::size_bytes<decltype(As[0])> +
                    kittens::size_bytes<decltype(Bs[0])>
                );
                kittens::tma::load_async(As[toc], g.A, {0, b, row, tile + 1}, bar);
                kittens::tma::load_async(Bs[toc], g.B, {0, b, tile + 1, col}, bar);
            }
        } else {
            // Consumers: compute
            kittens::warpgroup::increase_registers<256>();
            kittens::warpgroup::mma_AB(C_accum, As[tic], Bs[tic]);
            kittens::warpgroup::mma_async_wait();
        }
        __syncthreads();
    }

    // Store result for this batch slice
    if (warpgroupid == 1) {
        // If you later need tail handling (when M/N not multiples of 64), switch to a masked store
        kittens::warpgroup::store(g.C, C_accum, {0, b, row, col});
    }
}

void dispatch_micro(micro_globals g) {
    size_t smem = g.dynamic_shared_memory();
    cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    micro_tk<<<g.grid(), g.block(), smem>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::batch,                // NEW
        &micro_globals::M, &micro_globals::K, &micro_globals::N
    );
    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::batch,                // NEW
        &micro_globals::M, &micro_globals::K, &micro_globals::N
    );
}
