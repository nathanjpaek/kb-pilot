// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

#define NUM_WORKERS 1
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)
#define TILE_ROWS 16
#define TILE_VEC 32

struct micro_globals {
    kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, TILE_ROWS, TILE_VEC>> X;
    kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, TILE_ROWS, TILE_VEC>> Mask;
    kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, TILE_ROWS, TILE_VEC>> Y;
    int M;
    int N;

    __host__ dim3 grid()  const { return dim3(M, 1, 1); }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);

    const int row = blockIdx.x;
    if (row >= g.M) return;

    const int lane_id   = threadIdx.x & 31;
    const unsigned mask = 0xffffffff;

    float carry = 0.0f;

    for (int base = 0; base < g.N; base += 32) {
        const int col = base + lane_id;

        float x_val = 0.0f;
        float m_val = 0.0f;

        if (col < g.N) {
            const size_t idx = static_cast<size_t>(row) * g.N + col;
            x_val = __half2float(g.X.raw_ptr[idx]);
            m_val = __half2float(g.Mask.raw_ptr[idx]);
        }

        float acc = x_val * m_val;

        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float t = __shfl_up_sync(mask, acc, offset);
            if (lane_id >= offset) acc += t;
        }

        acc += carry;

        if (col < g.N) {
            const size_t idx = static_cast<size_t>(row) * g.N + col;
            g.Y.raw_ptr[idx] = __float2half(acc);
        }

        // Update carry with the last valid element in this 32-wide chunk
        float last_val = __shfl_sync(mask, acc, 31);
        carry = last_val;
    }

    kittens::warp::sync();
}

void dispatch_micro(micro_globals g) {
    constexpr int shm_bytes = 0;
    cudaFuncSetAttribute(micro_tk,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_bytes);
    micro_tk<<<g.grid(), g.block(), shm_bytes>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m,
        "micro_tk",
        &micro_globals::X,
        &micro_globals::Mask,
        &micro_globals::Y,
        &micro_globals::M,
        &micro_globals::N);

    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m,
        "dispatch_micro",
        &micro_globals::X,
        &micro_globals::Mask,
        &micro_globals::Y,
        &micro_globals::M,
        &micro_globals::N);
}