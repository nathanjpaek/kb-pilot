// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

static constexpr int TILE_M = 16;
static constexpr int TILE_N = 16;

static constexpr int NUM_WARPS   = 4;
static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

struct micro_globals {
  kittens::gl<
      kittens::half,
      -1, -1, -1, -1,
      kittens::st<kittens::half, TILE_M, TILE_N>> X;

  kittens::gl<
      kittens::half,
      -1, -1, -1, -1,
      kittens::st<kittens::half, TILE_M, TILE_N>> Y;

  float norm;   // Frobenius norm (FP32)
  int   M;      // rows
  int   N;      // cols

  __host__ __device__ dim3 grid() const {
    const int gx = (M + TILE_M - 1) / TILE_M;
    const int gy = (N + TILE_N - 1) / TILE_N;
    return dim3(gx, gy, 1);
  }
  __host__ __device__ dim3 block() const {
    return dim3(NUM_THREADS, 1, 1);
  }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
  const int total = g.M * g.N;
  const float inv_norm = 1.0f / g.norm;

  const kittens::half* __restrict__ x_ptr =
      reinterpret_cast<const kittens::half*>(g.X.raw_ptr);
  kittens::half* __restrict__ y_ptr =
      reinterpret_cast<kittens::half*>(g.Y.raw_ptr);

  for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < total;
       idx += blockDim.x * gridDim.x) {
    const __half xh = static_cast<__half>(x_ptr[idx]);
    const float xf = __half2float(xh);
    const float yf = xf * inv_norm;
    y_ptr[idx] = static_cast<kittens::half>(__float2half(yf));
  }

  kittens::warp::sync();
}

__host__ void dispatch_micro(micro_globals g) {
  const int total  = g.M * g.N;
  const int blocks = (total + NUM_THREADS - 1) / NUM_THREADS;

  cudaFuncSetAttribute(
      micro_tk,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      0);

  micro_tk<<<blocks, NUM_THREADS, 0>>>(g);
  cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
  kittens::py::bind_kernel<micro_tk, micro_globals>(
      m, "micro_tk",
      &micro_globals::X,
      &micro_globals::Y,
      &micro_globals::norm,
      &micro_globals::M,
      &micro_globals::N);

  kittens::py::bind_function<dispatch_micro, micro_globals>(
      m, "dispatch_micro",
      &micro_globals::X,
      &micro_globals::Y,
      &micro_globals::norm,
      &micro_globals::M,
      &micro_globals::N);
}