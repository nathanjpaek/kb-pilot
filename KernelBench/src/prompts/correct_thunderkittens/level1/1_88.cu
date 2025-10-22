// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

#define TILE_M 16
#define TILE_N 16
#define NUM_WORKERS 4
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct micro_globals {
    kittens::gl<kittens::half, 1, 1, -1, -1, kittens::st<kittens::half, TILE_M, TILE_N>> X;
    kittens::gl<kittens::half, 1, 1, -1, -1, kittens::st<kittens::half, TILE_M, TILE_N>> Y;
    int M;
    int N;

    __host__ dim3 grid() const {
        int tilesN = (N + TILE_N - 1) / TILE_N;
        int tilesM = (M + TILE_M - 1) / TILE_M;
        return dim3(tilesN, tilesM, 1);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;

    const int start_row = tile_row * TILE_M;
    const int start_col = tile_col * TILE_N;

    const int tid = threadIdx.x;

    const kittens::half* __restrict__ in_ptr  = g.X.raw_ptr;
    kittens::half* __restrict__ out_ptr = g.Y.raw_ptr;

    for (int idx = tid; idx < TILE_M * TILE_N; idx += blockDim.x) {
        int local_row = idx / TILE_N;
        int local_col = idx % TILE_N;

        int global_row = start_row + local_row;
        int global_col = start_col + local_col;

        if (global_row < g.M && global_col < g.N) {
            kittens::half h = in_ptr[global_row * g.N + global_col];
            float x = __half2float(h);
            float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
            float gelu  = 0.5f * x * (1.0f + tanhf(inner));
            out_ptr[global_row * g.N + global_col] = __float2half(gelu);
        }
    }
}

__host__ void dispatch_micro(micro_globals g) {
    micro_tk<<<g.grid(), g.block(), 0>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(m, "micro_tk",
        &micro_globals::X, &micro_globals::Y, &micro_globals::M, &micro_globals::N);
    kittens::py::bind_function<dispatch_micro, micro_globals>(m, "dispatch_micro",
        &micro_globals::X, &micro_globals::Y, &micro_globals::M, &micro_globals::N);
}