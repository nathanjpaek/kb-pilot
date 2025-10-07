// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <pybind11/pybind11.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16
#define NUM_WARPS 1
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

using tile_shape_h = kittens::st<kittens::half, TILE_M, TILE_N>;
using gl_half      = kittens::gl<kittens::half, -1, -1, -1, -1, tile_shape_h>;

struct micro_globals {
    gl_half A;
    gl_half B;
    gl_half C;
    int M;
    int K;
    int N;

    __host__ dim3 grid() const {
        return dim3((unsigned)((N + TILE_N - 1) / TILE_N),
                    (unsigned)((M + TILE_M - 1) / TILE_M),
                    1);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS); }
    __host__ static size_t dynamic_shared_memory() { return size_t(100000); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);

    // Shared tiles ----------------------------------------------------------
    auto& A_s = al.allocate<kittens::st<kittens::half, TILE_M, TILE_K>>();
    auto& B_s = al.allocate<kittens::st<kittens::half, TILE_K, TILE_N>>();
    auto& C_s = al.allocate<kittens::st<kittens::half, TILE_M, TILE_N>>();

    // Register tiles --------------------------------------------------------
    kittens::rt_fl<TILE_M, TILE_N, kittens::ducks::rt_layout::row> acc;
    kittens::warp::zero(acc);

    const int tile_row = blockIdx.y; // which 16x16 output tile (rows)
    const int tile_col = blockIdx.x; // which 16x16 output tile (cols)
    const int k_tiles  = (g.K + TILE_K - 1) / TILE_K;

    for (int kt = 0; kt < k_tiles; ++kt) {

        // load global -> shared
        kittens::warp::load(A_s, g.A, {0, 0, tile_row, kt});
        kittens::warp::load(B_s, g.B, {0, 0, kt, tile_col});
        __syncthreads();

        // shared -> register
        kittens::rt_bf<TILE_M, TILE_K, kittens::ducks::rt_layout::row>  A_rt;
        kittens::rt_bf<TILE_K, TILE_N, kittens::ducks::rt_layout::col>  B_rt;
        kittens::warp::load(A_rt, A_s);
        kittens::warp::load(B_rt, B_s);

        // tensor-core matmul
        kittens::warp::mma_AB(acc, A_rt, B_rt, acc);
        __syncthreads();
    }

    // store accumulator back
    kittens::warp::store(C_s, acc);
    __syncthreads();
    kittens::warp::store(g.C, C_s, {0, 0, tile_row, tile_col});
}

// -----------------------------------------------------------------------
// Dispatcher
// -----------------------------------------------------------------------
void dispatch_micro(micro_globals g) {
    size_t smem = micro_globals::dynamic_shared_memory();
    cudaFuncSetAttribute(micro_tk,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    micro_tk<<<g.grid(), g.block(), smem>>>(g);
    cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------
// pybind11 bindings
// -----------------------------------------------------------------------
PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk",
        &micro_globals::A,
        &micro_globals::B,
        &micro_globals::C,
        &micro_globals::M,
        &micro_globals::K,
        &micro_globals::N);

    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::A,
        &micro_globals::B,
        &micro_globals::C,
        &micro_globals::M,
        &micro_globals::K,
        &micro_globals::N);
}