/*
============================================================
🎯 EVALUATION RESULT for Level 1 Problem 13
============================================================
Problem: 13_Matmul_for_symmetric_matrices
Result: compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=141.0 runtime_stats={'mean': 141.0, 'std': 0.154, 'min': 141.0, 'max': 142.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.67, 'std': 0.00505, 'min': 2.67, 'max': 2.7, 'num_trials': 100}, 'speedup_ratio': 0.0189}}
============================================================
*/

// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

constexpr int TM = 16;   // tile rows
constexpr int TN = 16;   // tile cols

#define NUM_WORKERS 1
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct micro_globals {
    /* Global tensors */
    kittens::gl<kittens::half, 1, 1, -1, -1> A; // [M, K]
    kittens::gl<kittens::half, 1, 1, -1, -1> B; // [K, N]
    kittens::gl<kittens::half, 1, 1, -1, -1> C; // [M, N]

    int M;   // rows of A
    int K;   // shared dimension
    int N;   // cols of B / C

    __host__ dim3 grid() const {
        return dim3((N + TN - 1) / TN, (M + TM - 1) / TM, 1);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    const int tile_m = blockIdx.y;   // row-tile index
    const int tile_n = blockIdx.x;   // col-tile index

    const int base_row = tile_m * TM;
    const int base_col = tile_n * TN;

    constexpr int TILE_ELEMS = TM * TN;
    constexpr int ELEMS_PER_THREAD = (TILE_ELEMS + NUM_THREADS - 1) / NUM_THREADS;

    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        int elem_idx = threadIdx.x + e * NUM_THREADS;
        if (elem_idx >= TILE_ELEMS) break;

        int row_offset = elem_idx % TM;
        int col_offset = elem_idx / TM;

        int global_row = base_row + row_offset;
        int global_col = base_col + col_offset;

        if (global_row < g.M && global_col < g.N) {
            float accum = 0.0f;
            for (int k = 0; k < g.K; ++k) {
                size_t a_idx = global_row * g.A.stride<2>() + k * g.A.stride<3>();
                size_t b_idx = k * g.B.stride<2>() + global_col * g.B.stride<3>();

                kittens::half a_val = g.A.raw_ptr[a_idx];
                kittens::half b_val = g.B.raw_ptr[b_idx];

                accum += __half2float(a_val) * __half2float(b_val);
            }
            size_t c_idx = global_row * g.C.stride<2>() + global_col * g.C.stride<3>();
            g.C.raw_ptr[c_idx] = __float2half_rn(accum);
        }
    }
}

__host__ void dispatch_micro(micro_globals g) {
    micro_tk<<<g.grid(), g.block()>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(m, "micro_tk",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::M, &micro_globals::K, &micro_globals::N);

    kittens::py::bind_function<dispatch_micro, micro_globals>(m, "dispatch_micro",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::M, &micro_globals::K, &micro_globals::N);
}