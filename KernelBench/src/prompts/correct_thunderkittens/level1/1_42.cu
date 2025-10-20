// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

#define NUM_WORKERS 4
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

/* -------------------------------------------------------------------------- */
/*                               Global Struct                                */
/* -------------------------------------------------------------------------- */
struct micro_globals {
  /* Input  (B, C, H, W) */
  kittens::gl<kittens::half, -1, -1, -1, -1,
              kittens::st<kittens::half, 16, 16>>
      X;
  /* Output (B, C, H_out, W_out) */
  kittens::gl<kittens::half, -1, -1, -1, -1,
              kittens::st<kittens::half, 16, 16>>
      Y;

  int B, C, H, W;
  int K, stride, pad, dilation;

  /* --------------------------- launch parameters -------------------------- */
  __host__ dim3 grid() const {
    int H_out = (H + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
    int W_out = (W + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
    long long total = static_cast<long long>(B) * C * H_out * W_out;
    int blocks = static_cast<int>((total + NUM_THREADS - 1) / NUM_THREADS);
    if (blocks < 1) blocks = 1;
    return dim3(blocks);
  }

  __host__ dim3 block() const { return dim3(NUM_THREADS); }
};

/* -------------------------------------------------------------------------- */
/*                                   Kernel                                   */
/* -------------------------------------------------------------------------- */
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
  extern __shared__ kittens::alignment_dummy __shm[];
  kittens::shared_allocator al((int *)&__shm[0]);
  (void)al;  // silence unused var warning

  const int H_out =
      (g.H + 2 * g.pad - g.dilation * (g.K - 1) - 1) / g.stride + 1;
  const int W_out =
      (g.W + 2 * g.pad - g.dilation * (g.K - 1) - 1) / g.stride + 1;

  const __half *Xptr = reinterpret_cast<const __half *>(g.X.raw_ptr);
  __half *Yptr = reinterpret_cast<__half *>(g.Y.raw_ptr);
  const long long total =
      static_cast<long long>(g.B) * g.C * H_out * W_out;

  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<long long>(gridDim.x) * blockDim.x) {
    long long t = idx;
    int w_out = t % W_out;
    t /= W_out;
    int h_out = t % H_out;
    t /= H_out;
    int c = t % g.C;
    int b = t / g.C;

    int h_start = h_out * g.stride - g.pad;
    int w_start = w_out * g.stride - g.pad;

    float max_val = -3.402823466e38f;  // -FLT_MAX

    for (int kh = 0; kh < g.K; ++kh) {
      int h_in = h_start + kh * g.dilation;
      if (h_in < 0 || h_in >= g.H) continue;
      for (int kw = 0; kw < g.K; ++kw) {
        int w_in = w_start + kw * g.dilation;
        if (w_in < 0 || w_in >= g.W) continue;
        long long in_idx =
            (((static_cast<long long>(b) * g.C + c) * g.H + h_in) * g.W) +
            w_in;
        float val = __half2float(Xptr[in_idx]);
        if (val > max_val) max_val = val;
      }
    }

    long long out_idx =
        (((static_cast<long long>(b) * g.C + c) * H_out + h_out) * W_out) +
        w_out;
    Yptr[out_idx] = __float2half(max_val);
  }

  kittens::warp::sync();
}

/* -------------------------------------------------------------------------- */
/*                                 Dispatcher                                 */
/* -------------------------------------------------------------------------- */
void dispatch_micro(micro_globals g) {
  size_t smem = 0;
  micro_tk<<<g.grid(), g.block(), smem>>>(g);
  cudaDeviceSynchronize();
}

/* -------------------------------------------------------------------------- */
/*                                 PyBind11                                   */
/* -------------------------------------------------------------------------- */
PYBIND11_MODULE(tk_kernels, m) {
  kittens::py::bind_kernel<micro_tk, micro_globals>(
      m, "micro_tk", &micro_globals::X, &micro_globals::Y, &micro_globals::B,
      &micro_globals::C, &micro_globals::H, &micro_globals::W, &micro_globals::K,
      &micro_globals::stride, &micro_globals::pad, &micro_globals::dilation);

  kittens::py::bind_function<dispatch_micro, micro_globals>(
      m, "dispatch_micro", &micro_globals::X, &micro_globals::Y,
      &micro_globals::B, &micro_globals::C, &micro_globals::H, &micro_globals::W,
      &micro_globals::K, &micro_globals::stride, &micro_globals::pad,
      &micro_globals::dilation);
}