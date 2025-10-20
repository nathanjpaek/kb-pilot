// tk_kernels.cpp
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

// -----------------------------------------------------------------------------
// Kernel configuration
// -----------------------------------------------------------------------------
#define NUM_WORKERS  (1)                              // one warp
#define NUM_THREADS  (NUM_WORKERS * kittens::WARP_THREADS)

// -----------------------------------------------------------------------------
// Per-kernel global layouts & runtime parameters
// -----------------------------------------------------------------------------
struct micro_globals {
    // 2-D tensors with runtime row/col sizes
    kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, 16, 16>> X;
    kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, 16, 16>> Y;
    int M;                  // rows
    int N;                  // cols

    __host__ dim3 grid()  const { return dim3(M); }
    __host__ dim3 block() const { return dim3(NUM_THREADS); }
};

// -----------------------------------------------------------------------------
// CUDA kernel : row-wise L1 normalisation
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g)
{
    const int row  = blockIdx.x;
    if (row >= g.M) return;

    const int lane = threadIdx.x;            // 0-31

    const kittens::half* xptr = g.X.raw_ptr + row * g.N;
    kittens::half*       yptr = g.Y.raw_ptr + row * g.N;

    // Pass 1 : compute L1 norm of the row
    float partial_sum = 0.f;
    for (int col = lane; col < g.N; col += blockDim.x) {
        float v = __half2float(reinterpret_cast<const __half*>(xptr)[col]);
        partial_sum += fabsf(v);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);

    float l1_norm = __shfl_sync(0xffffffff, partial_sum, 0);
    if (l1_norm == 0.f) l1_norm = 1.f;       // avoid div-by-zero

    // Lightweight warp-scope sync (TK API use)
    kittens::warp::sync();

    // Pass 2 : normalise row
    for (int col = lane; col < g.N; col += blockDim.x) {
        float v = __half2float(reinterpret_cast<const __half*>(xptr)[col]);
        v /= l1_norm;
        reinterpret_cast<__half*>(yptr)[col] = __float2half(v);
    }
}

// -----------------------------------------------------------------------------
// Host dispatcher
// -----------------------------------------------------------------------------
__host__ void dispatch_micro(micro_globals g)
{
    constexpr int shm_bytes = 0;
    cudaFuncSetAttribute(micro_tk,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_bytes);
    micro_tk<<<g.grid(), g.block(), shm_bytes>>>(g);
    cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------
// PyBind11 module
// -----------------------------------------------------------------------------
PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk",
        &micro_globals::X,
        &micro_globals::Y,
        &micro_globals::M,
        &micro_globals::N);

    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::X,
        &micro_globals::Y,
        &micro_globals::M,
        &micro_globals::N);
}