// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

#define NUM_WARPS 4
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

using dtype = float;

// Row/col layouts for tensor-core contracts
using a_rt_layout = ducks::rt_layout::row;
using b_rt_layout = ducks::rt_layout::col;
using c_rt_layout = ducks::rt_layout::row;

// Global-memory tensor views
using a_gl_t = gl<dtype, -1, -1, -1, -1, st<dtype, TILE_M, TILE_K>>;
using b_gl_t = gl<dtype, -1, -1, -1, -1, st<dtype, TILE_K, TILE_N>>;
using c_gl_t = gl<dtype, -1, -1, -1, -1, st<dtype, TILE_M, TILE_N>>;

struct micro_globals {
    a_gl_t A;
    b_gl_t B;
    c_gl_t C;
    int     M;
    int     K;
    int     N;

    dim3 grid()  const { return dim3((M + TILE_M - 1) / TILE_M,
                                     (N + TILE_N - 1) / TILE_N, 1); }
    dim3 block() const { return dim3(NUM_THREADS, 1, 1); }

    size_t dynamic_shared_memory() const {
        return shared_allocator::bytes<st<dtype, TILE_M, TILE_K>>() +
               shared_allocator::bytes<st<dtype, TILE_K, TILE_N>>();
    }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Shared tiles
    auto& a_s = *al.allocate<st<dtype, TILE_M, TILE_K>>();
    auto& b_s = *al.allocate<st<dtype, TILE_K, TILE_N>>();

    // Register accumulator
    rt<dtype, TILE_M, TILE_N, c_rt_layout> c_r;
    zero(c_r);

    const int tile_m = blockIdx.x * TILE_M;
    const int tile_n = blockIdx.y * TILE_N;

    for (int k_tile = 0; k_tile < g.K; k_tile += TILE_K) {
        // Global → shared
        load(a_s, g.A, {0, 0, tile_m, k_tile});
        load(b_s, g.B, {0, 0, k_tile, tile_n});
        __syncthreads();

        // Shared → registers
        rt<dtype, TILE_M, TILE_K, a_rt_layout> a_r;
        rt<dtype, TILE_K, TILE_N, b_rt_layout> b_r;
        load(a_r, a_s);
        load(b_r, b_s);

        // Compute
        mma_AB(c_r, a_r, b_r, c_r);
        __syncthreads();
    }

    // Store result
    store(g.C, c_r, {0, 0, tile_m, tile_n});
}

void dispatch_micro(micro_globals g) {
    size_t smem = g.dynamic_shared_memory();
    cudaFuncSetAttribute(micro_tk,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    micro_tk<<<g.grid(), g.block(), smem>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    BIND_KERNEL(m, "micro_tk", micro_tk, micro_globals,
                A, B, C, M, K, N);
    BIND_FUNCTION(m, "dispatch_micro", dispatch_micro, micro_globals,
                  A, B, C, M, K, N);
}