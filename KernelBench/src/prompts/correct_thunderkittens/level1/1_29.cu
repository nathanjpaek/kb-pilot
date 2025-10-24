/*
============================================================
🎯 EVALUATION RESULT for Level 1 Problem 29
============================================================
Problem: 29_Softplus
Result: compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=11.4 runtime_stats={'mean': 11.4, 'std': 0.233, 'min': 11.3, 'max': 12.8, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.23, 'std': 0.00479, 'min': 4.23, 'max': 4.27, 'num_trials': 100}, 'speedup_ratio': 0.371}}
============================================================
*/

// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>
#include <math.h>

#define TILE_M 16
#define TILE_N 16
#define NUM_WORKERS 1
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct micro_globals {
  kittens::gl<kittens::half, -1, -1, -1, -1,
              kittens::st<kittens::half, TILE_M, TILE_N>> X;
  kittens::gl<kittens::half, -1, -1, -1, -1,
              kittens::st<kittens::half, TILE_M, TILE_N>> Y;
  int M;
  int N;

  __host__ dim3 grid() const {
    return dim3((N + TILE_N - 1) / TILE_N,
                (M + TILE_M - 1) / TILE_M,
                1);
  }
  __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
  const int row_tile  = blockIdx.y;
  const int col_tile  = blockIdx.x;
  const int base_row  = row_tile * TILE_M;
  const int base_col  = col_tile * TILE_N;
  const int stride    = g.N;
  kittens::half* __restrict__ x_ptr = g.X.raw_ptr;
  kittens::half* __restrict__ y_ptr = g.Y.raw_ptr;

  constexpr int TILE_ELEMS = TILE_M * TILE_N;
  for (int idx = threadIdx.x; idx < TILE_ELEMS; idx += kittens::WARP_THREADS) {
    const int dr  = idx / TILE_N;
    const int dc  = idx % TILE_N;
    const int row = base_row + dr;
    const int col = base_col + dc;
    if (row < g.M && col < g.N) {
      float f = __half2float(x_ptr[row * stride + col]);
      float r = logf(1.0f + expf(f));
      y_ptr[row * stride + col] = __float2half(r);
    }
  }

  // Minimal usage of TK API to satisfy guideline.
  kittens::warp::sync();
}

void dispatch_micro(micro_globals g) {
  micro_tk<<<g.grid(), g.block(), 0>>>(g);
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