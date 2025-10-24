/*
============================================================
🎯 EVALUATION RESULT for Level 1 Problem 96
============================================================
Problem: 96_HuberLoss
Result: compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=11.4 runtime_stats={'mean': 11.4, 'std': 0.00533, 'min': 11.4, 'max': 11.4, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 5.53, 'std': 0.0119, 'min': 5.52, 'max': 5.65, 'num_trials': 100}, 'speedup_ratio': 0.485}}
============================================================
*/

// SPEEDUP: 0.486x (Attempt 1/1 - SUCCESS)
// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

#define TILE_M 16
#define TILE_N 16
#define NUM_WORKERS 1
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct micro_globals {
    // 2-D tensors (runtime rows & cols) with half precision tiles
    kittens::gl<kittens::half, 1, 1, -1, -1,
                kittens::st<kittens::half, TILE_M, TILE_N>> predictions;
    kittens::gl<kittens::half, 1, 1, -1, -1,
                kittens::st<kittens::half, TILE_M, TILE_N>> targets;
    kittens::gl<kittens::half, 1, 1, -1, -1,
                kittens::st<kittens::half, TILE_M, TILE_N>> output;

    int M;
    int N;

    __host__ dim3 grid() const {
        int rows = (M + TILE_M - 1) / TILE_M;
        int cols = (N + TILE_N - 1) / TILE_N;
        return dim3(rows * cols);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    const __half* pred_ptr = reinterpret_cast<const __half*>(g.predictions.raw_ptr);
    const __half* targ_ptr = reinterpret_cast<const __half*>(g.targets.raw_ptr);
    __half*       out_ptr  = reinterpret_cast<__half*>(g.output.raw_ptr);

    int total  = g.M * g.N;
    int stride = gridDim.x * blockDim.x;
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < total) {
        float pred = __half2float(pred_ptr[idx]);
        float targ = __half2float(targ_ptr[idx]);

        float diff = pred - targ;
        float ad   = fabsf(diff);
        float loss = (ad < 1.0f) ? 0.5f * diff * diff : ad - 0.5f;

        out_ptr[idx] = __float2half(loss);
        idx += stride;
    }
    kittens::warp::sync();
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
        &micro_globals::predictions,
        &micro_globals::targets,
        &micro_globals::output,
        &micro_globals::M,
        &micro_globals::N);

    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::predictions,
        &micro_globals::targets,
        &micro_globals::output,
        &micro_globals::M,
        &micro_globals::N);
}