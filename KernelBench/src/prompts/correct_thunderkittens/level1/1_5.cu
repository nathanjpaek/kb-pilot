// tk_kernels.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define TILE_M 16
#define TILE_N 16
#define NUM_WORKERS 1
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct micro_globals {
    // Global tensor layouts: tiles of 16x16 half
    kittens::gl<kittens::half, -1, -1, -1, -1,
                kittens::st<kittens::half, TILE_M, TILE_N>> A;
    kittens::gl<kittens::half, -1, -1, -1, -1,
                kittens::st<kittens::half, TILE_M, TILE_N>> C;

    float s;   // scalar multiplier
    int   M;   // rows
    int   N;   // cols

    __host__ dim3 grid() const {
        const int tiles_x = (N + TILE_N - 1) / TILE_N;   // columns
        const int tiles_y = (M + TILE_M - 1) / TILE_M;   // rows
        return dim3(tiles_x, tiles_y, 1);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
    __host__ size_t dynamic_shared_memory() const { return 100000; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    using st_tile_t = kittens::st<kittens::half, TILE_M, TILE_N>;
    using rt_tile_t = kittens::rt<kittens::half, TILE_M, TILE_N,
                                  kittens::ducks::rt_layout::row>;

    // Shared allocator
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int *)&__shm[0]);

    st_tile_t &tile_s = al.allocate<st_tile_t>();

    // Tile indices (NOT element offsets)
    const int r_tile = blockIdx.y;  // which 16-row tile
    const int c_tile = blockIdx.x;  // which 16-col tile

    // 1) Global → Shared: use tile indices {b=0,d=0,r=r_tile,c=c_tile}
    kittens::warp::load(tile_s, g.A, {0, 0, r_tile, c_tile});
    __syncthreads();

    // 2) Shared → Registers
    rt_tile_t a_reg;
    kittens::warp::load(a_reg, tile_s);

    // 3) Multiply by scalar (broadcast)
    rt_tile_t c_reg;
    const kittens::half s_h = __float2half(g.s);
    kittens::warp::mul(c_reg, a_reg, s_h);  // scalar broadcast multiply

    // 4) Registers → Shared
    kittens::warp::store(tile_s, c_reg);
    __syncthreads();

    // 5) Shared → Global: store back to same tile coords
    kittens::warp::store(g.C, tile_s, {0, 0, r_tile, c_tile});
}

__host__ void dispatch_micro(micro_globals g) {
    const size_t shm_bytes = g.dynamic_shared_memory();
    cudaFuncSetAttribute(micro_tk,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_bytes);
    micro_tk<<<g.grid(), g.block(), shm_bytes>>>(g);
    cudaDeviceSynchronize();
}

/* ---------------- pybind11 bindings ---------------- */
PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk",
        &micro_globals::A,
        &micro_globals::C,
        &micro_globals::s,
        &micro_globals::M,
        &micro_globals::N);

    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::A,
        &micro_globals::C,
        &micro_globals::s,
        &micro_globals::M,
        &micro_globals::N);
}
