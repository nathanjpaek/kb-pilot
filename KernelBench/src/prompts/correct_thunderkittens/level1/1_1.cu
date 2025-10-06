// // Minimal matmul kernel with pybind11 bindings (compatible with Modal build)
// // tk_kernels.cu (ThunderKittens)
// ThunderKittens warp-level fp16 matmul for H100 (half inputs/outputs)
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

#define NUM_WARPS 1
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

using adtype = kittens::half;   // A: fp16
using bdtype = kittens::half;   // B: fp16
using cdtype = kittens::half;   // C: fp16 (accumulate in half for Hopper warp mma)

using a_rt_layout = kittens::ducks::rt_layout::row;
using b_rt_layout = kittens::ducks::rt_layout::col;
using c_rt_layout = kittens::ducks::rt_layout::row;

using a_gl_t = kittens::gl<adtype, -1, -1, -1, -1, kittens::st<adtype, TILE_M, TILE_K>>;
using b_gl_t = kittens::gl<bdtype, -1, -1, -1, -1, kittens::st<bdtype, TILE_K, TILE_N>>;
using c_gl_t = kittens::gl<cdtype, -1, -1, -1, -1, kittens::st<cdtype, TILE_M, TILE_N>>;

struct micro_globals {
    a_gl_t A;
    b_gl_t B;
    c_gl_t C;
    int     M;
    int     K;
    int     N;

    dim3 grid()  const { return dim3((N + TILE_N - 1) / TILE_N,
                                     (M + TILE_M - 1) / TILE_M, 1); }
    dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
    size_t dynamic_shared_memory() const {
        return sizeof(kittens::st<adtype, TILE_M, TILE_K>)
             + sizeof(kittens::st<bdtype, TILE_K, TILE_N>)
             + sizeof(kittens::st<cdtype, TILE_M, TILE_N>);
    }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);

    kittens::st<adtype, TILE_M, TILE_K> (&a_s) = al.allocate<kittens::st<adtype, TILE_M, TILE_K>>();
    kittens::st<bdtype, TILE_K, TILE_N> (&b_s) = al.allocate<kittens::st<bdtype, TILE_K, TILE_N>>();
    kittens::st<cdtype, TILE_M, TILE_N> (&c_s) = al.allocate<kittens::st<cdtype, TILE_M, TILE_N>>();

    kittens::rt<cdtype, TILE_M, TILE_N, c_rt_layout> c_r;
    kittens::warp::zero(c_r); // zero accumulator

    const int row0 = blockIdx.y * TILE_M;
    const int col0 = blockIdx.x * TILE_N;

    // for (int k0 = 0; k0 < g.K; k0 += TILE_K) {
    //     if (row0 < g.M && k0 < g.K) {
    //         kittens::warp::load(a_s, g.A, {0, 0, row0, k0});
    //     }
    //     if (k0 < g.K && col0 < g.N) {
    //         kittens::warp::load(b_s, g.B, {0, 0, k0, col0});
    //     }
    //     __syncthreads();

    //     kittens::rt<adtype, TILE_M, TILE_K, a_rt_layout> a_r;
    //     kittens::rt<bdtype, TILE_K, TILE_N, b_rt_layout> b_r;
    //     kittens::warp::load(a_r, a_s);
    //     kittens::warp::load(b_r, b_s);

    //     // Half-precision tensor core MMA on Hopper (accumulates in half for this warp path)
    //     kittens::warp::mma_AB(c_r, a_r, b_r, c_r);

    //     __syncthreads();
    // }

    if ((row0 + TILE_M) <= g.M && (col0 + TILE_N) <= g.N) {
        kittens::warp::store(g.C, c_r, {0, 0, row0, col0});
    }
}

void dispatch_micro(micro_globals g) {
    size_t smem = g.dynamic_shared_memory();
    cudaFuncSetAttribute(micro_tk,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    micro_tk<<<g.grid(), g.block(), smem>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::M, &micro_globals::K, &micro_globals::N
    );
    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::M, &micro_globals::K, &micro_globals::N
    );
}
