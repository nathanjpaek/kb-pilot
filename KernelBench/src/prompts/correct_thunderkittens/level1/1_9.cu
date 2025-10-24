/*
============================================================
🎯 EVALUATION RESULT for Level 1 Problem 9
============================================================
Problem: 9_Tall_skinny_matrix_multiplication_
Result: compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.86 runtime_stats={'mean': 3.86, 'std': 0.00657, 'min': 3.85, 'max': 3.88, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.62, 'std': 0.00852, 'min': 2.59, 'max': 2.63, 'num_trials': 100}, 'speedup_ratio': 0.679}}
============================================================
*/

// tk_kernels.cu  --- ThunderKittens 16×16 matmul (bf16 A/B/C, fp32 accum on warp path)
//
// WHY THE PREVIOUS VERSION FAILED:
//   • The warp-level MMA for bf16 inputs requires a FLOAT accumulator:
//       mma_AB_base(rt_base<float,row>& d,
//                   const rt_base<bf16,row>& A,
//                   const rt_base<bf16,col>& B,
//                   const rt_base<float,row>& C);
//     Your code used a bf16 accumulator (rt_bf), so no overload matched.
//   • Also, give each gl<> the correct per-tensor tile type so GL↔RT load shapes align.
//   • Coordinates passed to gl are TILE INDICES {B,D,R_tile,C_tile}, not element offsets.
//
// HOW TO AVOID THIS ERROR:
//   • For bf16 A/B with warp::mma_AB, declare the accumulator as rt_fl<…> and pass it
//     both as D and C (destination and prior accumulator).
//   • Use a_st=st_bf<16,16> along K for A, b_st for B, c_st for C.
//   • Use GL→SHARED→RT path; shapes must match exactly.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <pybind11/pybind11.h>

constexpr int TILE_M  = 16;
constexpr int TILE_N  = 16;
constexpr int TILE_K  = 16;

constexpr int NUM_WARPS   = 1;                           // one warp per block
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

// Per-tensor shared tile types (use half at the interface, like 1_7)
using a_st = kittens::st<kittens::half, TILE_M, TILE_K>; // A: 16 (rows) × 16 (K-tile)
using b_st = kittens::st<kittens::half, TILE_K, TILE_N>; // B: 16 (K-tile) × 16 (cols)
using c_st = kittens::st<kittens::half, TILE_M, TILE_N>; // C: 16 × 16

// Global views (half at the interface to match Python and 1_7)
using a_gl = kittens::gl<kittens::half, 1, -1, -1, -1, a_st>;
using b_gl = kittens::gl<kittens::half, 1, -1, -1, -1, b_st>;
using c_gl = kittens::gl<kittens::half, 1, -1, -1, -1, c_st>;

struct micro_globals {
    a_gl A;                                              // (M×K) tiled as 16×16 along K
    b_gl B;                                              // (K×N) tiled as 16×16 along K
    c_gl C;                                              // (M×N) tiled as 16×16
    int  M, K, N;

    // Map grid.x over N-tiles, grid.y over M-tiles
    __host__ dim3 grid() const {
        return dim3( (unsigned)((N + TILE_N - 1) / TILE_N),
                     (unsigned)((M + TILE_M - 1) / TILE_M),
                     1 );
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }

    // One A tile + one B tile staged in shared
    __host__ size_t dynamic_shared_memory() const {
        return sizeof(a_st) + sizeof(b_st) + sizeof(c_st);
    }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    // ---- Shared allocator (dynamic smem) ----
    extern __shared__ kittens::alignment_dummy __shm[];
    // Use modest alignment; we are not using TMA here. Avoid 1024B alignment to prevent
    // dynamic smem overallocation gaps that can cause OOB when only reserving sizeof(tiles).
    kittens::shared_allocator<16> al((int*)&__shm[0]);
    a_st &A_s = al.allocate<a_st>();                     // shared A tile (16×16K)
    b_st &B_s = al.allocate<b_st>();                     // shared B tile (16K×16)
    c_st &C_s = al.allocate<c_st>();                     // shared C tile (16×16)

    // ---- Tile indices (NOT element offsets) ----
    const int tile_c = blockIdx.x;                       // which 16-col tile of N
    const int tile_r = blockIdx.y;                       // which 16-row tile of M

    // ---- Accumulator in FLOAT (required by warp bf16 MMA) ----
    kittens::rt_fl<TILE_M, TILE_N, kittens::ducks::rt_layout::row> acc;
    kittens::warp::zero(acc);                            // acc ← 0.f

    // Number of K tiles
    const int k_tiles = (g.K + TILE_K - 1) / TILE_K;

    // ---- K loop: GL→SHARED, SHARED→RT, MMA ----
    for (int kt = 0; kt < k_tiles; ++kt) {
        // Global → Shared; coords are {B=0, D=0, R_tile, C_tile}
        // A uses (row tile, k tile), B uses (k tile, col tile).
        kittens::warp::load(A_s, g.A, {0, 0, tile_r, kt});
        kittens::warp::load(B_s, g.B, {0, 0, kt,     tile_c});
        __syncthreads();

        // Shared → Registers: shapes must match the st types above.
        kittens::rt_bf<TILE_M, TILE_K, kittens::ducks::rt_layout::row> A_rt;
        kittens::rt_bf<TILE_K, TILE_N, kittens::ducks::rt_layout::col> B_rt;
        kittens::warp::load(A_rt, A_s);
        kittens::warp::load(B_rt, B_s);

        // Warp MMA: bf16 × bf16 → float accumulate
        // Both dest and prior-accumulator operands must be FLOAT tiles.
        kittens::warp::mma_AB(acc, A_rt, B_rt, acc);
        __syncthreads();
    }

    // ---- Store via shared (handles conversion) ----
    kittens::warp::store(C_s, acc);
    __syncthreads();
    kittens::warp::store(g.C, C_s, {0, 0, tile_r, tile_c});
}

// ------------------------------ Dispatcher ------------------------------
void dispatch_micro(micro_globals g) {
    const size_t smem = g.dynamic_shared_memory();
    cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    micro_tk<<<g.grid(), g.block(), smem>>>(g);
    cudaDeviceSynchronize();
}

// ------------------------------ pybind11 ------------------------------
PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::M, &micro_globals::K, &micro_globals::N);

    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::M, &micro_globals::K, &micro_globals::N);
}
