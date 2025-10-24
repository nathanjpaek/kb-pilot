/*
============================================================
🎯 EVALUATION RESULT for Level 1 Problem 31
============================================================
Problem: 31_ELU
Result: compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=9.25 runtime_stats={'mean': 9.25, 'std': 0.00476, 'min': 9.24, 'max': 9.27, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.24, 'std': 0.00537, 'min': 4.24, 'max': 4.29, 'num_trials': 100}, 'speedup_ratio': 0.458}}
============================================================
*/

// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>
#include <math.h>

#define TILE_M 16
#define TILE_N 16
#define NUM_WORKERS 4
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

/* ------------------------------------------------------ */
/*                 micro globals structure                */
/* ------------------------------------------------------ */
struct micro_globals {
  /* 2-D input / output tensors (row-major, contiguous) */
  kittens::gl<kittens::half, 1, 1, -1, -1,
              kittens::st<kittens::half, TILE_M, TILE_N>> X;
  kittens::gl<kittens::half, 1, 1, -1, -1,
              kittens::st<kittens::half, TILE_M, TILE_N>> Y;

  /* matrix size and ELU parameter */
  int   M;      // rows
  int   N;      // cols
  float alpha;  // elu α

  /* launch specification required by kittens::py helpers */
  __host__ dim3 grid()  const { return dim3((N + TILE_N - 1) / TILE_N,
                                           (M + TILE_M - 1) / TILE_M, 1); }
  __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
};

/* ------------------------------------------------------ */
/*                        kernel                          */
/* ------------------------------------------------------ */
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
  /* shared allocator (unused but required for ABI) */
  extern __shared__ kittens::alignment_dummy __shm[];
  kittens::shared_allocator al(reinterpret_cast<int*>(__shm));

  const int tile_r = blockIdx.y;
  const int tile_c = blockIdx.x;

  /* per-thread strided loop across the 16×16 tile */
  for (int idx = threadIdx.x; idx < TILE_M * TILE_N; idx += blockDim.x) {
    int local_r = idx / TILE_N;
    int local_c = idx % TILE_N;

    int global_r = tile_r * TILE_M + local_r;
    int global_c = tile_c * TILE_N + local_c;

    if (global_r >= g.M || global_c >= g.N) continue;

    size_t offset = static_cast<size_t>(global_r) * g.N + global_c;

    /* load, apply ELU, store */
    kittens::half hx = g.X.raw_ptr[offset];
    float x = __half2float(hx);
    float y = (x > 0.f) ? x : g.alpha * (expf(x) - 1.f);
    g.Y.raw_ptr[offset] = __float2half(y);
  }
}

/* ------------------------------------------------------ */
/*                     dispatcher                         */
/* ------------------------------------------------------ */
void dispatch_micro(micro_globals g) {
  const int shm_bytes = 0;
  cudaFuncSetAttribute(micro_tk,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shm_bytes);
  micro_tk<<<g.grid(), g.block(), shm_bytes>>>(g);
  cudaDeviceSynchronize();
}

/* ------------------------------------------------------ */
/*                     pybind11 module                    */
/* ------------------------------------------------------ */
PYBIND11_MODULE(tk_kernels, m) {
  kittens::py::bind_kernel<micro_tk, micro_globals>(
      m, "micro_tk",
      &micro_globals::X,
      &micro_globals::Y,
      &micro_globals::M,
      &micro_globals::N,
      &micro_globals::alpha);

  kittens::py::bind_function<dispatch_micro, micro_globals>(
      m, "dispatch_micro",
      &micro_globals::X,
      &micro_globals::Y,
      &micro_globals::M,
      &micro_globals::N,
      &micro_globals::alpha);
}