// tk_kernels.cu -------------------------------------------------------------
// Swish activation ( Y = X * sigmoid(X) ) implemented with ThunderKittens.
//
// This revision fixes the original build failure and adds exhaustive,
// line-by-line commentary explaining both the *what* and the *why*.
//
// Root-cause summary of the previous errors:
//   (a) kittens::warp::rcp<…>() does not exist – the correct idiom is to
//       divide by using warp::div() with a *register tile* numerator.
//       Passing a scalar broke template matching.
//   (b) A few advanced template parameters were unnecessary and created
//       extra points of failure; they have been removed for robustness.

#include "kittens.cuh"          // ThunderKittens core API
#include "pyutils/pyutils.cuh"  // Automatic pybind11 helpers

// ---------------------------------------------------------------------------
//  Compile-time tuning knobs
// ---------------------------------------------------------------------------
constexpr int TILE_M           = 16;                          // tile height
constexpr int TILE_N           = 16;                          // tile width
constexpr int WARPS_PER_BLOCK  = 4;                           // 4 warps / block
constexpr int NUM_THREADS      = WARPS_PER_BLOCK *
                                 kittens::WARP_THREADS;       // 128 threads

// ---------------------------------------------------------------------------
//  Global-memory view of the tensors
// ---------------------------------------------------------------------------
// A ThunderKittens gl<…> object is a *typed descriptor* – not a real pointer.
//   - -1 strides mean “infer from contiguous row-major layout”.
//   - ST_h encodes the per-warp (16×16) tile geometry.
using ST_h = kittens::st_hf<TILE_M, TILE_N>;                  // shape tag
using GL_h = kittens::gl<kittens::half, -1, -1, -1, -1, ST_h>; // FP16 tensor

// ---------------------------------------------------------------------------
//  Kernel argument bundle – ThunderKittens emits the pybind glue directly
//  from this struct.  Everything must be POD-style.
// ---------------------------------------------------------------------------
struct micro_globals {
    GL_h X;         // Input  matrix (FP16)
    GL_h Y;         // Output matrix (FP16)
    int  M;         // Total #rows
    int  N;         // Total #cols

    // ---------------- CUDA launch shape (static, derived from M/N) ----------
    __host__ dim3 grid() const
    {
        // One *warp* computes a 16×16 tile; 4 warps = 2×2 layout per block.
        const int tiles_r = (M + TILE_M - 1) / TILE_M;  // #warp tiles (rows)
        const int tiles_c = (N + TILE_N - 1) / TILE_N;  // #warp tiles (cols)
        return dim3((tiles_c + 1) / 2,                  // ceil(tiles_c/2)
                    (tiles_r + 1) / 2,                  // ceil(tiles_r/2)
                    1);
    }

    __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }

    __host__ size_t dynamic_shared_memory() const { return 0; } // none
};

// ---------------------------------------------------------------------------
//  CUDA micro-kernel – one 32×32 macro-tile per thread block.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g)
{
    // -----------------------------------------------------------------------
    // 1. Warp identification – four warps arranged as
    //        (warp 0) (warp 1)
    //        (warp 2) (warp 3)
    //    Each warp owns exactly one 16×16 register tile.
    // -----------------------------------------------------------------------
    const int warp_id  = threadIdx.x / kittens::WARP_THREADS; // 0‥3
    const int warp_row = warp_id >> 1;                        // 0 or 1
    const int warp_col = warp_id &  1;                        // 0 or 1

    // Translate the in-block (row,col) to a *global* warp-tile coordinate.
    const int tile_r = blockIdx.y * 2 + warp_row;             // global row
    const int tile_c = blockIdx.x * 2 + warp_col;             // global col

    // -----------------------------------------------------------------------
    // 2. Fast coarse boundary check – drop any warp whose tile is entirely
    //    outside the logical matrix.  Partial tiles are handled internally
    //    by ThunderKittens’ predicates, so no extra code is needed.
    // -----------------------------------------------------------------------
    if (tile_r * TILE_M >= g.M || tile_c * TILE_N >= g.N) return;

    // -----------------------------------------------------------------------
    // 3. Declare all register tiles:
    //      * RTh  – FP16 tiles for I/O
    //      * RTf  – FP32 tiles for math (better accuracy, no perf loss on H100)
    // -----------------------------------------------------------------------
    using RTh = kittens::rt_hf<TILE_M, TILE_N>;               // half tile
    using RTf = kittens::rt_fl<TILE_M, TILE_N>;               // float tile

    RTh x_h, y_h;                     // on-chip FP16 tiles (load / store)
    RTf x_f, neg_f, e_f, den_f;       // working FP32 tiles
    RTf y_f;                          // final FP32 result

    // -----------------------------------------------------------------------
    // 4. Load 16×16 tile from global memory:   X → x_h
    //    The coordinate tuple is {N0, N1, row_tile_idx, col_tile_idx}.
    //    We leave the two leading (dummy) dimensions at 0.
    // -----------------------------------------------------------------------
    kittens::warp::load(x_h, g.X, {0, 0, tile_r, tile_c});

    // -----------------------------------------------------------------------
    // 5. Promote to FP32 for computation.
    // -----------------------------------------------------------------------
    kittens::warp::copy(x_f, x_h);

    // -----------------------------------------------------------------------
    // 6. Swish math:   y = x / (1 + exp(-x))
    // -----------------------------------------------------------------------
    kittens::warp::mul(neg_f, x_f, -1.0f);      // neg_f = −x
    kittens::warp::exp(e_f,  neg_f);            // e_f   = exp(−x)
    kittens::warp::add(den_f, e_f, 1.0f);       // den_f = 1 + e_f

    // IMPORTANT – TEMPLATE RULE:
    //   kittens::warp::div(out, lhs, rhs) requires *both* lhs & rhs
    //   to be register tiles; scalars are NOT allowed.  Using a scalar
    //   constant as the numerator was the original source of failure.
    kittens::warp::div(y_f, x_f, den_f);        // y_f = x / den_f

    // -----------------------------------------------------------------------
    // 7. Demote back to FP16 for storage.
    // -----------------------------------------------------------------------
    kittens::warp::copy(y_h, y_f);

    // -----------------------------------------------------------------------
    // 8. Store the tile back to global memory:  y_h → Y
    // -----------------------------------------------------------------------
    kittens::warp::store(g.Y, y_h, {0, 0, tile_r, tile_c});
}

// ---------------------------------------------------------------------------
//  User-friendly C++ launcher – sets kernel attributes & synchronises.
// ---------------------------------------------------------------------------
__host__ void dispatch_micro(micro_globals g)
{
    // (a) Inform CUDA that the kernel uses 0 bytes of dynamic shared memory.
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(g.dynamic_shared_memory()));

    // (b) Launch the kernel.
    micro_tk<<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);

    // (c) In a standalone demo it is convenient to block on completion;
    //     production code could remove this for better overlap.
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
//  pybind11 glue – auto-generated helpers remove virtually all boilerplate.
// ---------------------------------------------------------------------------
PYBIND11_MODULE(tk_kernels, m)
{
    // Expose the raw CUDA kernel (rarely called directly from Python).
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk",
        &micro_globals::X, &micro_globals::Y,
        &micro_globals::M, &micro_globals::N);

    // Expose the convenient C++ dispatcher used by model_new.py.
    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::X, &micro_globals::Y,
        &micro_globals::M, &micro_globals::N);
}