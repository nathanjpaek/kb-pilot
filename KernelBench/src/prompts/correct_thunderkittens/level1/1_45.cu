// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cstdint>

#define NUM_WORKERS 1
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct micro_globals {
    kittens::gl<kittens::half, -1, -1, -1, -1> Y;
    int64_t N;

    __host__ dim3 grid() const {
        const int elems_per_thread = 16;
        int64_t threads_needed = (N + elems_per_thread - 1) / elems_per_thread;
        int blocks = static_cast<int>((threads_needed + NUM_THREADS - 1) / NUM_THREADS);
        return dim3(blocks == 0 ? 1 : blocks, 1, 1);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    kittens::warp::sync();  // No-op placeholder kernel
}

void dispatch_micro(micro_globals g) {
    constexpr size_t shm = 0;
    cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
    micro_tk<<<g.grid(), g.block(), shm>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(m, "micro_tk",
        &micro_globals::Y,
        &micro_globals::N);

    kittens::py::bind_function<dispatch_micro, micro_globals>(m, "dispatch_micro",
        &micro_globals::Y,
        &micro_globals::N);
}