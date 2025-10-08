// tk_kernels.cu — Sigmoid with ThunderKittens (fixed)
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <pybind11/pybind11.h>

#define TILE_M 16
#define TILE_N 16
#define NUM_WARPS 1
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

struct micro_globals {
    using ST = kittens::st<kittens::half, TILE_M, TILE_N>;
    kittens::gl<kittens::half, -1, -1, -1, -1, ST> X;
    kittens::gl<kittens::half, -1, -1, -1, -1, ST> Y;

    int64_t M, N;

    __host__ dim3 grid() const {
        return dim3((N + TILE_N - 1) / TILE_N,
                    (M + TILE_M - 1) / TILE_M, 1);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
    __host__ size_t dynamic_shared_memory() const { return sizeof(ST); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    using ST   = kittens::st<kittens::half, TILE_M, TILE_N>;
    using RT_h = kittens::rt<kittens::half, TILE_M, TILE_N, kittens::ducks::rt_layout::row>;
    using RT_f = kittens::rt_fl<TILE_M, TILE_N>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);
    ST &x_s = al.allocate<ST>();

    const int r_tile = blockIdx.y;
    const int c_tile = blockIdx.x;

    // Global -> Shared
    kittens::warp::load(x_s, g.X, {0, 0, r_tile, c_tile});
    __syncthreads();

    // Shared -> Registers (compute in float)
    RT_f x_f, neg_f, exp_f, denom_f, y_f, one_f;
    kittens::warp::load(x_f, x_s);

    // y = 1 / (1 + exp(-x))
    kittens::warp::mul(neg_f, x_f, -1.0f);       // -x
    kittens::warp::exp(exp_f, neg_f);            // exp(-x)
    kittens::warp::add(denom_f, exp_f, 1.0f);    // 1 + exp(-x)

    // Build a tile of ones and divide tile-by-tile
    kittens::warp::zero(one_f);
    kittens::warp::add(one_f, one_f, 1.0f);      // one_f = 1
    kittens::warp::div(y_f, one_f, denom_f);     // y = 1 / denom

    // Convert back to half and store
    RT_h y_h;
    kittens::warp::copy(y_h, y_f);
    kittens::warp::store(g.Y, y_h, {0, 0, r_tile, c_tile});
}

__host__ void dispatch_micro(micro_globals g) {
    const size_t shm = g.dynamic_shared_memory();
    cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    micro_tk<<<g.grid(), g.block(), shm>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk",
        &micro_globals::X, &micro_globals::Y,
        &micro_globals::M, &micro_globals::N);

    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::X, &micro_globals::Y,
        &micro_globals::M, &micro_globals::N);
}
