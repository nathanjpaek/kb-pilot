// tk_kernels.cu — Tanh with ThunderKittens built from exp/mul/add/sub/div
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <pybind11/pybind11.h>

#define TILE_M 16
#define TILE_N 16
#define NUM_WARPS 1
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

struct micro_globals {
    using tile_t   = kittens::st<kittens::half, TILE_M, TILE_N>;
    using layout_t = kittens::gl<kittens::half, 1, 1, -1, -1, tile_t>; // B=1, D=1, R/C runtime

    layout_t A;   // input
    layout_t B;   // output
    int      M;
    int      N;

    __host__ dim3 grid()  const {
        return dim3((N + TILE_N - 1) / TILE_N,
                    (M + TILE_M - 1) / TILE_M, 1);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
    __host__ size_t dynamic_shared_memory() const { return sizeof(tile_t); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void tanh_tk(const micro_globals g) {
    using RT = kittens::rt<kittens::half, TILE_M, TILE_N, kittens::ducks::rt_layout::row>;

    // Shared allocator
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);
    micro_globals::tile_t &x_s = al.allocate<micro_globals::tile_t>();

    const int r_tile = blockIdx.y;  // tile indices (NOT element offsets)
    const int c_tile = blockIdx.x;

    // Global -> Shared
    kittens::warp::load(x_s, g.A, {0, 0, r_tile, c_tile});
    __syncthreads();

    // Shared -> Registers
    RT x, two_x, e2x, num, den, y;
    kittens::warp::load(x, x_s);

    // implements the tanh operation
    // two_x = 2 * x
    kittens::warp::mul(two_x, x, __float2half(2.0f));
    // e2x = exp(two_x)
    kittens::warp::exp(e2x, two_x);
    // num = e2x - 1
    kittens::warp::sub(num, e2x, __float2half(1.0f));
    // den = e2x + 1
    kittens::warp::add(den, e2x, __float2half(1.0f));
    // y = num / den
    kittens::warp::div(y, num, den);

    // Register -> Global (no shared bounce needed)
    kittens::warp::store(g.B, y, {0, 0, r_tile, c_tile});
}

void dispatch_micro(micro_globals g) {
    const size_t shm = g.dynamic_shared_memory();
    cudaFuncSetAttribute(tanh_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    tanh_tk<<<g.grid(), g.block(), shm>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<tanh_tk, micro_globals>(
        m, "tanh_tk",
        &micro_globals::A, &micro_globals::B, &micro_globals::M, &micro_globals::N);
    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::A, &micro_globals::B, &micro_globals::M, &micro_globals::N);
}
