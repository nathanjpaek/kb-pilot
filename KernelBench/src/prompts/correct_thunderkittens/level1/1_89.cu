// Speedup ratio: 0.339x

// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

#define NUM_WORKERS 4
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct micro_globals {
  // 2-D tensors (row-major), staged through 16×16 half tiles
  kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st_hf<16, 16>> X;
  kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st_hf<16, 16>> Y;
  int M;   // rows
  int N;   // cols

  // launch helpers used by kittens::py::bind_kernel
  __host__ dim3 grid()  const { return dim3((M + NUM_WORKERS - 1) / NUM_WORKERS); }
  __host__ dim3 block() const { return dim3(NUM_THREADS); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
  const int tid          = threadIdx.x;
  const int warp_id_blk  = tid / kittens::WARP_THREADS;
  const int lane_id      = tid & (kittens::WARP_THREADS - 1);
  const int row_idx      = blockIdx.x * NUM_WORKERS + warp_id_blk;
  if (row_idx >= g.M) return;

  const kittens::half* __restrict__ x_row =
      g.X.raw_ptr + static_cast<size_t>(row_idx) * g.N;
  kittens::half* __restrict__ y_row =
      g.Y.raw_ptr + static_cast<size_t>(row_idx) * g.N;

  float running = 0.f;
  const unsigned mask = 0xffffffffu;

  for (int col_base = 0; col_base < g.N; col_base += kittens::WARP_THREADS) {
    const int col_idx = col_base + lane_id;

    float v = 0.f;
    if (col_idx < g.N) { v = __half2float(x_row[col_idx]); }

    // Inclusive scan within the warp.
    #pragma unroll
    for (int offs = 1; offs < kittens::WARP_THREADS; offs <<= 1) {
      float n = __shfl_up_sync(mask, v, offs);
      if (lane_id >= offs) v += n;
    }

    v += running;

    if (col_idx < g.N) { y_row[col_idx] = __float2half(v); }

    const float chunk_total = __shfl_sync(mask, v, kittens::WARP_THREADS - 1);
    running = chunk_total;
  }

  kittens::warp::sync();  // light-weight scope sync
}

void dispatch_micro(micro_globals g) {
  cudaFuncSetAttribute(micro_tk,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       0);
  micro_tk<<<g.grid(), g.block(), 0>>>(g);
  cudaDeviceSynchronize();
}

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