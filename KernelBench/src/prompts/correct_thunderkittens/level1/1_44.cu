// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cuda_fp16.h>

#define NUM_WORKERS  (1)
#define NUM_THREADS  (NUM_WORKERS * kittens::WARP_THREADS)

// Runtime parameters and tensor handles.
struct micro_globals {
    // Allow all runtime dimensions so any (B, C, I) shapes are accepted.
    kittens::gl<kittens::half, -1, -1, -1, -1> X;   // (B, C, I)  mapped to (B, D, R, C)
    kittens::gl<kittens::half, -1, -1, -1, -1> Y;   // (B, C, O)

    int K;   // kernel size
    int S;   // stride
    int P;   // padding
    int B;   // batch size
    int C;   // channel count
    int I;   // input length

    __host__ dim3 grid()  const { return dim3(B * C, 1, 1); }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);
    (void)al;  // no shared-memory allocations needed

    const int lane      = threadIdx.x & 31;   // lane within warp
    const int bc_idx    = blockIdx.x;         // (batch * C + channel)
    const int batch     = bc_idx / g.C;
    const int channel   = bc_idx % g.C;

    const int O = (g.I + 2 * g.P - g.K) / g.S + 1;  // output length

    for (int pos = lane; pos < O; pos += 32) {
        const int window_start = pos * g.S - g.P;

        float sum   = 0.0f;
        int   count = 0;

        for (int k = 0; k < g.K; ++k) {
            const int src_pos = window_start + k;
            if (src_pos >= 0 && src_pos < g.I) {
                const size_t src_idx =
                    (static_cast<size_t>(batch) * g.C + channel) * static_cast<size_t>(g.I) + src_pos;
                sum += __half2float(g.X.raw_ptr[src_idx]);
                ++count;
            }
        }

        const float  avg_f = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
        const __half avg_h = __float2half(avg_f);

        const size_t dst_idx =
            (static_cast<size_t>(batch) * g.C + channel) * static_cast<size_t>(O) + pos;
        g.Y.raw_ptr[dst_idx] = avg_h;
    }
}

__host__ void dispatch_micro(micro_globals g) {
    constexpr int smem_bytes = 0;
    cudaFuncSetAttribute(micro_tk,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
    micro_tk<<<g.grid(), g.block(), smem_bytes>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk",
        &micro_globals::X, &micro_globals::Y,
        &micro_globals::K, &micro_globals::S, &micro_globals::P,
        &micro_globals::B, &micro_globals::C, &micro_globals::I
    );

    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::X, &micro_globals::Y,
        &micro_globals::K, &micro_globals::S, &micro_globals::P,
        &micro_globals::B, &micro_globals::C, &micro_globals::I
    );
}