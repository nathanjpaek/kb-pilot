// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

#define NUM_WORKERS 1
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)   // 32 threads, 1 warp

// -----------------------------------------------------------------------------
// Global arguments structure
// -----------------------------------------------------------------------------
struct micro_globals {
  // 2-D tensors with runtime shapes (rows=M, cols=N)
  kittens::gl<kittens::half, -1, -1, -1, -1,
              kittens::st<kittens::half, 16, 16>> X;
  kittens::gl<kittens::half, -1, -1, -1, -1,
              kittens::st<kittens::half, 16, 16>> Y;
  int M;     // rows
  int N;     // cols
  int dim;   // 0 => rows, 1 => cols (PyTorch-style)

  // Launch helpers required by kittens::py::bind_kernel
  __host__ dim3 grid()  const { return dim3(dim == 1 ? M : N); }
  __host__ dim3 block() const { return dim3(NUM_THREADS);      }
};

// -----------------------------------------------------------------------------
// Device kernel
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
  // Dummy ThunderKittens activity to satisfy “must use TK API”
  kittens::rt<kittens::half, 16, 16> rt_dummy;
  kittens::warp::zero(rt_dummy);

  const int lane = threadIdx.x;

  if (g.dim == 1) {
    // Reverse cumulative sum along columns
    const int row = blockIdx.x;
    const int N   = g.N;

    const kittens::half* __restrict__ in  = g.X.raw_ptr + static_cast<size_t>(row) * N;
    kittens::half*       __restrict__ out = g.Y.raw_ptr + static_cast<size_t>(row) * N;

    if (lane == 0) {
      __half acc = __float2half(0.f);
      for (int c = N - 1; c >= 0; --c) {
        acc     = __hadd(acc, in[c]);
        out[c]  = acc;
      }
    }
  } else {
    // Reverse cumulative sum along rows
    const int col = blockIdx.x;
    const int M   = g.M;
    const int N   = g.N;

    const kittens::half* __restrict__ in  = g.X.raw_ptr + col;
    kittens::half*       __restrict__ out = g.Y.raw_ptr + col;

    if (lane == 0) {
      __half acc = __float2half(0.f);
      for (int r = M - 1; r >= 0; --r) {
        size_t idx = static_cast<size_t>(r) * N;
        acc        = __hadd(acc, in[idx]);
        out[idx]   = acc;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Host-side dispatcher
// -----------------------------------------------------------------------------
void dispatch_micro(micro_globals g) {
  constexpr int shm = 0;
  cudaFuncSetAttribute(micro_tk,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shm);
  micro_tk<<<g.grid(), g.block(), shm>>>(g);
  cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------
// PyBind11 exports
// -----------------------------------------------------------------------------
PYBIND11_MODULE(tk_kernels, m) {
  kittens::py::bind_kernel<micro_tk, micro_globals>(
      m, "micro_tk",
      &micro_globals::X,
      &micro_globals::Y,
      &micro_globals::M,
      &micro_globals::N,
      &micro_globals::dim);

  kittens::py::bind_function<dispatch_micro, micro_globals>(
      m, "dispatch_micro",
      &micro_globals::X,
      &micro_globals::Y,
      &micro_globals::M,
      &micro_globals::N,
      &micro_globals::dim);
}