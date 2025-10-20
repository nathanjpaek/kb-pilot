// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

#define NUM_WORKERS  (1)
#define NUM_THREADS  (NUM_WORKERS * kittens::WARP_THREADS)

struct micro_globals {
    kittens::gl<kittens::half, 1, -1, -1, -1, kittens::st<kittens::half, 16, 16>> X;
    kittens::gl<kittens::half, 1, -1, -1, -1, kittens::st<kittens::half, 16, 16>> Y;
    int M;
    int N;

    __host__ dim3 grid()  const { return dim3(M); }
    __host__ dim3 block() const { return dim3(NUM_THREADS); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);

    const int row = blockIdx.x;
    if (row >= g.M) return;

    const int N = g.N;

    const kittens::half* __restrict__ x_row =
        g.X.raw_ptr + static_cast<size_t>(row) * N;
    kittens::half* __restrict__ y_row =
        g.Y.raw_ptr + static_cast<size_t>(row) * N;

    float local_sum = 0.f;
    for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
        float v = __half2float(x_row[idx]);
        local_sum += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    __shared__ float row_sum;
    if (threadIdx.x == 0) row_sum = local_sum;
    __syncthreads();

    float inv_norm = (row_sum > 0.f) ? rsqrtf(row_sum) : 0.f;

    for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
        float v = __half2float(x_row[idx]);
        y_row[idx] = __float2half(v * inv_norm);
    }

    kittens::warp::sync();
}

void dispatch_micro(micro_globals g) {
    constexpr size_t shm_bytes = 0;
    cudaFuncSetAttribute(micro_tk,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_bytes);
    micro_tk<<<g.grid(), g.block(), shm_bytes>>>(g);
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