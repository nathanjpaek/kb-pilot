// // Minimal matmul kernel with pybind11 bindings (compatible with Modal build)
// // tk_kernels.cu (ThunderKittens)
// Use kittens only for binding helpers; implement a simple fp16 matmul
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

#define TILE_SIZE 32

using a_gl_t = kittens::gl<kittens::half, -1, -1, -1, -1>;
using b_gl_t = kittens::gl<kittens::half, -1, -1, -1, -1>;
using c_gl_t = kittens::gl<kittens::half, -1, -1, -1, -1>;

struct micro_globals {
    a_gl_t A;
    b_gl_t B;
    c_gl_t C;
    int     M; // unused for square matmul; present to match caller
    int     K; // unused (assume square); present to match caller
    int     N; // side length
};

__global__ void matrix_multiply_kernel(const __half* __restrict__ A,
                                       const __half* __restrict__ B,
                                       __half* __restrict__ C,
                                       int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        float a = __half2float(A[row * K + kk]);
        float b = __half2float(B[kk * N + col]);
        sum += a * b;
    }
    C[row * N + col] = __float2half(sum);
}

// Host launcher compatible with kittens::py::bind_function
void dispatch_micro(micro_globals g) {
    const __half* A = reinterpret_cast<const __half*>(g.A.raw_ptr);
    const __half* B = reinterpret_cast<const __half*>(g.B.raw_ptr);
    __half*       C = reinterpret_cast<__half*>(g.C.raw_ptr);

    const int M = g.M;
    const int K = g.K;
    const int N = g.N;
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);

    matrix_multiply_kernel<<<blocks, threads>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    // Bind only the host function; Python passes torch tensors and ints
    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::M, &micro_globals::K, &micro_globals::N
    );
}
