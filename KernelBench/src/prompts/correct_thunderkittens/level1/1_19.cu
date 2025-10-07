// tk_kernels.cu – ReLU with TK tiles
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define TILE_M 16
#define TILE_N 16
#define NUM_THREADS (1 * kittens::WARP_THREADS)

struct micro_globals {
    kittens::gl<kittens::half, -1, -1, -1, -1,
                kittens::st<kittens::half, TILE_M, TILE_N>> X;
    kittens::gl<kittens::half, -1, -1, -1, -1,
                kittens::st<kittens::half, TILE_M, TILE_N>> Y;
    int M, N;

    __host__ dim3 grid()  const { return dim3((N + TILE_N - 1) / TILE_N,
                                              (M + TILE_M - 1) / TILE_M, 1); }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
    __host__ size_t dynamic_shared_memory() const {
        return sizeof(kittens::st<kittens::half, TILE_M, TILE_N>);
    }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    using ST = kittens::st<kittens::half, TILE_M, TILE_N>;
    using RT = kittens::rt<kittens::half, TILE_M, TILE_N, kittens::ducks::rt_layout::row>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);

    ST &x_s = al.allocate<ST>();

    const int r_tile = blockIdx.y;   // tile indices (NOT element offsets)
    const int c_tile = blockIdx.x;

    // Global -> Shared (pin B=D=0)
    kittens::warp::load(x_s, g.X, {0, 0, r_tile, c_tile});
    __syncthreads();

    // Shared -> Reg
    RT x_rt, y_rt;
    kittens::warp::load(x_rt, x_s);

    // ReLU: y = max(x, 0) using scalar overload (no zero tile needed)
    kittens::warp::max(y_rt, x_rt, (kittens::half)0);

    // Reg -> Global (no shared bounce needed)
    kittens::warp::store(g.Y, y_rt, {0, 0, r_tile, c_tile});
}

__host__ void dispatch_micro(micro_globals g) {
    const size_t shm = g.dynamic_shared_memory();
    cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
    micro_tk<<<g.grid(), g.block(), shm>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk", &micro_globals::X, &micro_globals::Y, &micro_globals::M, &micro_globals::N);
    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro", &micro_globals::X, &micro_globals::Y, &micro_globals::M, &micro_globals::N);
}
