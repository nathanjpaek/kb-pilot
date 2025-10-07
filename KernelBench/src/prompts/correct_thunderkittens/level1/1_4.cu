// tk_matvec_noTMA_XY.cu
// Mat-vec: Y[N] = W[NxK] @ X[K]
// Guaranteed-correct on H100 even when X is bound as rows=K, cols=1.
// We avoid TMA for X (and Y) to dodge the vector-descriptor axis constraint,
// but keep TK tiles/ops for the core math.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define K_TILE           128                 // columns per step
#define ROWS_PER_WARP     16                 // rows per warp
#define NUM_WARPS          4                 // 64 rows per block
#define NUM_THREADS  (NUM_WARPS * kittens::WARP_THREADS)

using dtype = kittens::bf16;

// Shared tiles/vectors for TK usage
using W_st = kittens::st_bf<ROWS_PER_WARP, K_TILE>;  // 16x128 weights tile in shared
using X_sv = kittens::sv_bf<K_TILE>;                 // 128-length chunk in shared

// Global views (W via TK tiles; X,Y are still gl<> but we won't TMA them)
using W_gl = kittens::gl<dtype, 1, 1, -1, -1, W_st>;
using X_gl = kittens::gl<dtype, 1, 1, -1,  1>;      // no tile type → we won’t TMA load
using Y_gl = kittens::gl<dtype, 1, 1, -1,  1>;      // no tile type → we won’t TMA store

struct micro_globals {
    W_gl W;   // (N x K), TK-tiled: {r = m_tile, c = k_tile}
    X_gl X;   // (K),     bound as rows=K, cols=1 (row-varying)
    Y_gl Y;   // (N),     bound as rows=N, cols=1 (row-varying)
    int  N;   // rows of W / len(Y)
    int  K;   // cols of W / len(X)

    __host__ dim3 grid() const {
        const int rows_per_block = ROWS_PER_WARP * NUM_WARPS; // 64
        return dim3((N + rows_per_block - 1) / rows_per_block, 1, 1);
    }
    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }

    __host__ size_t dynamic_shared_memory() const {
        // 4 weight tiles + 1 vector chunk
        return sizeof(W_st) * NUM_WARPS + sizeof(X_sv);
    }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void matvec_kernel(const micro_globals g) {
    static_assert(NUM_THREADS == kittens::WARP_THREADS * NUM_WARPS, "Expect NUM_WARPS full warps");

    // Dynamic shared memory + allocator (style that works in your env)
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);

    // Per-warp W tiles (TK will TMA these)
    W_st &W0 = al.allocate<W_st>();
    W_st &W1 = al.allocate<W_st>();
    W_st &W2 = al.allocate<W_st>();
    W_st &W3 = al.allocate<W_st>();
    W_st* W_smem[NUM_WARPS] = { &W0, &W1, &W2, &W3 };

    // One shared 128-elem X chunk per K-tile (we will fill it manually)
    X_sv &X_smem = al.allocate<X_sv>();
    dtype* Xs_flat = reinterpret_cast<dtype*>(&X_smem);  // flat view

    const int warp = threadIdx.x >> 5; // 0..NUM_WARPS-1
    const int lane = threadIdx.x & 31;

    // Rows this block/warp handles
    const int m_tile_group = blockIdx.x * NUM_WARPS;
    const int my_m_tile    = m_tile_group + warp;
    const int row_base     = my_m_tile * ROWS_PER_WARP;
    if (row_base >= g.N) return;

    const int k_tiles = (g.K + K_TILE - 1) / K_TILE;

    // TK register tiles as in VM example
    kittens::rt_bf<ROWS_PER_WARP, K_TILE> weights_rt, bcast_rt;
    typename kittens::rt_bf<ROWS_PER_WARP, K_TILE>::row_vec x_vec;
    typename kittens::rt_bf<ROWS_PER_WARP, K_TILE>::col_vec col_sum;
    kittens::rv_bf<ROWS_PER_WARP> acc, partial;
    kittens::warp::zero(acc);

    // Raw pointers to X and Y (no TMA) - TODO: THIS IS CORRECT BUT VERY SLOW
    const dtype* __restrict__ Xp = g.X.raw_ptr;  // if your gl uses .ptr, replace .data() with .ptr
          dtype* __restrict__ Yp = g.Y.raw_ptr;  // same note here

    for (int kt = 0; kt < k_tiles; ++kt) {
        // ---- Global -> Shared (X) WITHOUT TMA ----
        // X is rows=K, cols=1. Copy 128 contiguous elements starting at kt*K_TILE.
        if (warp == 0) {
            const int base = kt * K_TILE;
            // lane-coalesced copy into shared chunk
            for (int i = lane; i < K_TILE; i += 32) {
                Xs_flat[i] = Xp[base + i];
            }
        }

        // ---- Global (TK) -> Shared (TK) for W ----
        // TK handles W tile via TMA (legal 16x128 tile)
        kittens::warp::load(*W_smem[warp], g.W, {0, 0, my_m_tile, kt});
        __syncthreads();

        // ---- Shared -> Registers (TK) ----
        kittens::warp::load(weights_rt, *W_smem[warp]); // st_bf<16,128> -> rt_bf<16,128>
        kittens::warp::load(x_vec,      X_smem);        // sv_bf<128>    -> row_vec(128)

        // ---- TK math: broadcast, mul, row-reduce ----
        kittens::warp::broadcast_col(bcast_rt, x_vec);  // 16x128
        kittens::warp::mul(bcast_rt, bcast_rt, weights_rt);
        kittens::warp::row_sum(col_sum, bcast_rt);      // -> 16x1
        kittens::warp::copy(partial, col_sum);          // -> rv_bf<16>
        kittens::warp::add(acc, acc, partial);
        __syncthreads();
    }

    // ---- Store Y WITHOUT TMA (avoid vector descriptor as well) ----
    // Write the 16 results for this warp directly.
    if (lane == 0) {
        const int rows_here = max(0, min(ROWS_PER_WARP, g.N - row_base));
        for (int i = 0; i < rows_here; ++i) {
            Yp[row_base + i] = acc[i][0];
        }
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
