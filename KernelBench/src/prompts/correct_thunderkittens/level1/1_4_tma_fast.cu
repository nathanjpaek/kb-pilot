// tk_matvec_broadcast_reduce.cu
// ThunderKittens matrix–vector: Y[N] = W[NxK] @ X[K]
// Exact TK sequence inside the loop:
//   warp::load(weights_rt,  W_smem[warp])
//   warp::load(x_vec,       X_smem)
//   warp::broadcast_col(bcast_rt, x_vec)
//   warp::mul(bcast_rt, bcast_rt, weights_rt)
//   warp::row_sum(col_sum, bcast_rt)
//   warp::copy(partial, col_sum)
//   acc += partial
//
// Assumptions (like the VM example):
//   • N is a multiple of 16 (or ignore trailing rows)
//   • K is a multiple of 128 (or pad/ignore trailing cols)

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define K_TILE          128                 // columns processed per K-step
#define ROWS_PER_WARP    16                 // rows per warp (fixed TK warp tile)
#define NUM_WARPS         4                 // 4 warps => 64 rows per block
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

using dtype = kittens::bf16;

// Shared tiles / vectors
using W_st = kittens::st_bf<ROWS_PER_WARP, K_TILE>;  // 16 x 128 weights tile (shared)
using X_sv = kittens::sv_bf<K_TILE>;                 // 128-length vector chunk (shared)
using Y_sv = kittens::sv_bf<ROWS_PER_WARP>;          // 16-length output vector (typing the gl)

// Global views:
//  - W: rows (R) vary with N, cols (C) vary with K
//  - X: **rows (R) vary** with K (vector length on R), col fixed = 1
//  - Y: **rows (R) vary** with N, col fixed = 1
using W_gl = kittens::gl<dtype, 1, 1, -1, -1, W_st>;
using X_gl = kittens::gl<dtype, 1, 1, -1,  1, X_sv>;
using Y_gl = kittens::gl<dtype, 1, 1, -1,  1, Y_sv>;

struct micro_globals {
    W_gl W;   // (N x K) tiled as {r = m_tile, c = k_tile}
    X_gl X;   // (K)      tiled as {r = k_tile, c = 0}
    Y_gl Y;   // (N)      tiled as {r = m_tile, c = 0}
    int  N;   // rows of W / len(Y)
    int  K;   // cols of W / len(X)

    __host__ dim3 grid() const {
        const int rows_per_block = ROWS_PER_WARP * NUM_WARPS; // 64
        return dim3((N + rows_per_block - 1) / rows_per_block, 1, 1);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }

    // Simple SMEM size matching the allocator calls below
    __host__ size_t dynamic_shared_memory() const {
        return sizeof(W_st) * NUM_WARPS + sizeof(X_sv);
    }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void matvec_kernel(const micro_globals g) {
    static_assert(NUM_THREADS == kittens::WARP_THREADS * NUM_WARPS, "Expect NUM_WARPS full warps");

    // Aligned dynamic shared memory; allocator style that works in your setup
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);

    // Per-warp shared tiles for W
    W_st &W0 = al.allocate<W_st>();
    W_st &W1 = al.allocate<W_st>();
    W_st &W2 = al.allocate<W_st>();
    W_st &W3 = al.allocate<W_st>();
    W_st* W_smem[NUM_WARPS] = { &W0, &W1, &W2, &W3 };

    // One shared 128-chunk of X reused by the whole block per K-tile
    X_sv &X_smem = al.allocate<X_sv>();

    const int warp = threadIdx.x >> 5; // 0..NUM_WARPS-1

    // This block handles NUM_WARPS tiles along M (each tile = 16 rows)
    const int m_tile_group = blockIdx.x * NUM_WARPS;
    const int my_m_tile    = m_tile_group + warp;
    const int row_base     = my_m_tile * ROWS_PER_WARP;
    if (row_base >= g.N) return;

    const int k_tiles = (g.K + K_TILE - 1) / K_TILE;

    // Register tiles per the VM example
    kittens::rt_bf<ROWS_PER_WARP, K_TILE> weights_rt, bcast_rt;
    typename kittens::rt_bf<ROWS_PER_WARP, K_TILE>::row_vec x_vec;
    typename kittens::rt_bf<ROWS_PER_WARP, K_TILE>::col_vec col_sum;
    kittens::rv_bf<ROWS_PER_WARP> acc, partial;

    kittens::warp::zero(acc);

    for (int kt = 0; kt < k_tiles; ++kt) {
        // ---- Global -> Shared ----
        // Vector X length lives on ROWS → load with r = kt, c = 0
        if (warp == 0) {
            kittens::warp::load(X_smem, g.X, {0, 0, kt, 0});
        }
        // Weights W: r = my_m_tile (rows), c = kt (K chunk)
        kittens::warp::load(*W_smem[warp], g.W, {0, 0, my_m_tile, kt});
        __syncthreads();

        // ---- Shared -> Registers ----
        kittens::warp::load(weights_rt, *W_smem[warp]); // st_bf<16,128> -> rt_bf<16,128>
        kittens::warp::load(x_vec,      X_smem);        // sv_bf<128>    -> row_vec(128)

        // ---- Broadcast, elementwise multiply, row reduction ----
        kittens::warp::broadcast_col(bcast_rt, x_vec);  // 16x128
        kittens::warp::mul(bcast_rt, bcast_rt, weights_rt);
        kittens::warp::row_sum(col_sum, bcast_rt);      // -> 16x1
        kittens::warp::copy(partial, col_sum);          // -> rv_bf<16>

        // Accumulate across K tiles
        kittens::warp::add(acc, acc, partial);
        __syncthreads();
    }

    // ---- Store 16-row output vector for this warp ----
    // Y length lives on ROWS → store with r = my_m_tile, c = 0
    if (row_base + ROWS_PER_WARP <= g.N) {
        kittens::warp::store(g.Y, acc, {0, 0, my_m_tile, 0});
    }
}

/* --------------- Dispatcher --------------- */
void dispatch_micro(micro_globals g) {
    const size_t smem = g.dynamic_shared_memory();
    cudaFuncSetAttribute(
        matvec_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)
    );
    matvec_kernel<<<g.grid(), g.block(), smem>>>(g);
    cudaDeviceSynchronize();
}

/* ---------------- pybind ------------------ */
PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<matvec_kernel, micro_globals>(
        m, "matvec",
        &micro_globals::W, &micro_globals::X, &micro_globals::Y,
        &micro_globals::N, &micro_globals::K
    );
    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_matvec",
        &micro_globals::W, &micro_globals::X, &micro_globals::Y,
        &micro_globals::N, &micro_globals::K
    );
}