// tk_kernels.cu — Matmul with large K, fixed
#include <cuda.h>
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int TILE_M  = 16;
constexpr int TILE_N  = 16;
constexpr int TILE_K  = 16;

constexpr int WARPS_PER_BLOCK = 4;                               // 2×2 warps
constexpr int NUM_THREADS     = WARPS_PER_BLOCK * kittens::WARP_THREADS;

using sub_tile  = kittens::st<kittens::half, TILE_M, TILE_N>;    // 16×16
using gl_layout = kittens::gl<kittens::half, -1, -1, -1, -1, sub_tile>;

struct micro_globals {
    gl_layout A;   // (M×K), tiles {r = m_tile, c = k_tile}
    gl_layout B;   // (K×N), tiles {r = k_tile, c = n_tile}
    gl_layout C;   // (M×N), tiles {r = m_tile, c = n_tile}
    int M, K, N;

    __host__ dim3 grid() const {
        // 4 warps = 2×2 tiles per block → block covers 32×32 outputs
        int blocks_m = (M + (TILE_M * 2) - 1) / (TILE_M * 2);
        int blocks_n = (N + (TILE_N * 2) - 1) / (TILE_N * 2);
        return dim3(blocks_n, blocks_m, 1);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
    __host__ size_t dynamic_shared_memory() const {
        // 4 A tiles (16×16) + 4 B tiles (16×16)
        return WARPS_PER_BLOCK * (sizeof(sub_tile) + sizeof(sub_tile));
    }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g)
{
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);

    // Per-warp shared tiles (match register tile widths to avoid width assertion)
    sub_tile (&A_s)[WARPS_PER_BLOCK] = al.allocate<sub_tile, WARPS_PER_BLOCK>();
    sub_tile (&B_s)[WARPS_PER_BLOCK] = al.allocate<sub_tile, WARPS_PER_BLOCK>();

    // Register accumulator per warp
    // (You can use rt_fl<16,16> if you prefer float accum.)
    kittens::rt<kittens::half, TILE_M, TILE_N, kittens::ducks::rt_layout::row> c_rt;
    kittens::warp::sub(c_rt, c_rt, c_rt);   // zero

    const int warp_id = threadIdx.x / kittens::WARP_THREADS;

    // 2×2 warp layout within the block
    const int warp_tile_row = warp_id / 2;  // {0,1}
    const int warp_tile_col = warp_id % 2;  // {0,1}

    // Tile indices (NOT element offsets)
    const int block_m = blockIdx.y;                             // which 32-row macro tile
    const int block_n = blockIdx.x;                             // which 32-col macro tile
    const int m_tile  = block_m * 2 + warp_tile_row;            // 16-row tile index
    const int n_tile  = block_n * 2 + warp_tile_col;            // 16-col tile index

    const int k_tiles = (g.K + TILE_K - 1) / TILE_K;

    for (int kt = 0; kt < k_tiles; ++kt) {
        // Global → Shared: per-warp, matching 16×16 shapes
        kittens::warp::load(A_s[warp_id], g.A, {0, 0, m_tile, kt});
        kittens::warp::load(B_s[warp_id], g.B, {0, 0, kt, n_tile});
        __syncthreads();

        // Shared → Registers
        kittens::rt<kittens::half, TILE_M, TILE_K, kittens::ducks::rt_layout::row> a_rt;
        kittens::rt<kittens::half, TILE_K, TILE_N, kittens::ducks::rt_layout::col> b_rt;
        kittens::warp::load(a_rt, A_s[warp_id]);
        kittens::warp::load(b_rt, B_s[warp_id]);

        // Matrix multiply-accumulate (NOT elementwise)
        kittens::warp::mma_AB(c_rt, a_rt, b_rt, c_rt);

        __syncthreads();
    }

    // Register → Global (no need for a shared bounce if shapes match)
    kittens::warp::store(g.C, c_rt, {0, 0, m_tile, n_tile});
}

__host__ void dispatch_micro(micro_globals g)
{
    const size_t shm = g.dynamic_shared_memory();
    cudaFuncSetAttribute(micro_tk,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm);
    micro_tk<<<g.grid(), g.block(), shm>>>(g);
    cudaDeviceSynchronize();
}

// ---------------- pybind11 bindings -----------------
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m,
        "micro_tk",
        &micro_globals::A,
        &micro_globals::B,
        &micro_globals::C,
        &micro_globals::M,
        &micro_globals::K,
        &micro_globals::N);

    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m,
        "dispatch_micro",
        &micro_globals::A,
        &micro_globals::B,
        &micro_globals::C,
        &micro_globals::M,
        &micro_globals::K,
        &micro_globals::N);
}
