// Speedup ratio: 0.544x

// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

#define NUM_WORKERS 4
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

// ---------------------------------------------------------------------
// micro_globals : tensor descriptors + runtime sizes
// ---------------------------------------------------------------------
struct micro_globals {
  kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, 16, 16>> X;
  kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, 16, 16>> Mask;
  kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, 16, 16>> Y;
  int M;   // rows
  int N;   // cols

  __host__ dim3 grid()  const { return dim3(M); }
  __host__ dim3 block() const { return dim3(NUM_THREADS); }
};

// ---------------------------------------------------------------------
// Kernel : masked inclusive prefix-sum per row
// ---------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
  const int row = blockIdx.x;
  if (row >= g.M) return;

  const int tid  = threadIdx.x;
  const int Ncol = g.N;
  const int elems_per_thread = (Ncol + NUM_THREADS - 1) / NUM_THREADS;
  const int start = tid * elems_per_thread;
  const int end   = min(start + elems_per_thread, Ncol);

  // Mandatory ThunderKittens touch
  kittens::rv_bf<16> dummy;
  kittens::warp::zero(dummy);

  // ------------------------------------------------------------------
  // shared memory for per-thread partial reductions
  // ------------------------------------------------------------------
  extern __shared__ kittens::alignment_dummy __shm[];
  float* sh_prefix = reinterpret_cast<float*>(__shm);   // NUM_THREADS floats

  // Row pointers
  const kittens::half* __restrict__ x_row = g.X.raw_ptr    + static_cast<size_t>(row) * Ncol;
  const kittens::half* __restrict__ m_row = g.Mask.raw_ptr + static_cast<size_t>(row) * Ncol;
  kittens::half*       __restrict__ y_row = g.Y.raw_ptr    + static_cast<size_t>(row) * Ncol;

  // Pass-1 : per-thread masked sum
  float local_sum = 0.f;
  for (int idx = start; idx < end; ++idx) {
    __half xh = *reinterpret_cast<const __half*>(&x_row[idx]);
    __half mh = *reinterpret_cast<const __half*>(&m_row[idx]);
    local_sum += __half2float(xh) * __half2float(mh);
  }
  sh_prefix[tid] = local_sum;
  __syncthreads();

  // Serial exclusive scan over thread results (tid==0)
  if (tid == 0) {
    float acc = 0.f;
    for (int i = 0; i < NUM_THREADS; ++i) {
      float tmp = sh_prefix[i];
      sh_prefix[i] = acc;
      acc += tmp;
    }
  }
  __syncthreads();

  // Pass-2 : write inclusive masked prefix sums
  float running = sh_prefix[tid];
  for (int idx = start; idx < end; ++idx) {
    __half xh = *reinterpret_cast<const __half*>(&x_row[idx]);
    __half mh = *reinterpret_cast<const __half*>(&m_row[idx]);
    running += __half2float(xh) * __half2float(mh);
    y_row[idx] = __float2half(running);
  }
}

// ---------------------------------------------------------------------
// Host dispatcher
// ---------------------------------------------------------------------
void dispatch_micro(micro_globals g) {
  const size_t shm_bytes = NUM_THREADS * sizeof(float) + 256;  // padding for alignment
  cudaFuncSetAttribute(micro_tk,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(shm_bytes));
  micro_tk<<<g.grid(), g.block(), shm_bytes>>>(g);
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------
// PyBind11 bindings
// ---------------------------------------------------------------------
PYBIND11_MODULE(tk_kernels, m) {
  kittens::py::bind_kernel<micro_tk, micro_globals>(
      m, "micro_tk",
      &micro_globals::X,
      &micro_globals::Mask,
      &micro_globals::Y,
      &micro_globals::M,
      &micro_globals::N);

  kittens::py::bind_function<dispatch_micro, micro_globals>(
      m, "dispatch_micro",
      &micro_globals::X,
      &micro_globals::Mask,
      &micro_globals::Y,
      &micro_globals::M,
      &micro_globals::N);
}