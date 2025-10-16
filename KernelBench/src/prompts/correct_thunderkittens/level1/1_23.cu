// tk_kernels.cu — Row-wise softmax with ThunderKittens (half in/out, float compute)
//
// Key fixes vs. your attempt:
//  • Use the right reduction output type: typename RT::col_vec (NOT rv_* directly).
//  • Remove TMA; use warp::load/store (no semaphores needed).
//  • Provide micro_globals::grid()/block()/dynamic_shared_memory() for py bindings.
//  • Do stable softmax: pass1 row_max, pass2 sum(exp(x - max)), pass3 write exp/sum.
//
// Assumptions:
//  • TILE_M = 16 rows, TILE_N = 128 cols per tile.
//  • For full correctness on ragged tails (N not multiple of 128), either pad inputs
//    or add masked loads/stores. (This version assumes N is a multiple of 128.)

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <pybind11/pybind11.h>

#define TILE_M      16
#define TILE_N      128
#define NUM_WORKERS 1
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

/* ---- Types ---- */
using ST = kittens::st<kittens::half, TILE_M, TILE_N>;               // 16x128 half tile
using GL = kittens::gl<kittens::half, -1, -1, -1, -1, ST>;           // (B,D,R,C) all runtime

struct micro_globals {
  GL X;               // input  M×N in tiles of 16×128
  GL Y;               // output M×N in tiles of 16×128
  int M;              // rows
  int N;              // cols

  // One block per 16-row tile; we iterate columns inside the kernel.
  __host__ dim3 grid() const  { return dim3((unsigned)((M + TILE_M - 1) / TILE_M), 1, 1); }
  __host__ dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
  __host__ size_t dynamic_shared_memory() const { return 2 * sizeof(ST); } // x_s + y_s
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
  // ---- Which 16-row tile do we own? (tile index, not element) ----
  const int r_tile = blockIdx.x;
  if (r_tile * TILE_M >= g.M) return;

  // ---- Shared tiles for staging ----
  extern __shared__ kittens::alignment_dummy __shm[];
  kittens::shared_allocator al((int*)&__shm[0]);
  ST &x_s = al.allocate<ST>();
  ST &y_s = al.allocate<ST>();

  // ---- Register tiles & vectors (float math for stability) ----
  using RT = kittens::rt_fl<TILE_M, TILE_N, kittens::ducks::rt_layout::row>; // 16×128 float
  RT x_rt_f, e_rt_f, tmp_rt;

  // For row-wise reductions, TK expects the tile's *column-vector* type:
  using CV = typename RT::col_vec;   // per-row (length 16), correct layout for row_* reductions
  CV row_max_cv, row_sum_cv, tile_vec_cv;

  // Convenience scalar constants
  constexpr float INV_LN2 = 1.4426950408889634f;    // 1/ln(2), to use exp2

  // ---------------- Pass 1: row-wise max over all column tiles ----------------
  // Initialize row_max to -inf
  kittens::warp::neg_infty(row_max_cv);


  const int n_tiles = (g.N + TILE_N - 1) / TILE_N;
  for (int c_tile = 0; c_tile < n_tiles; ++c_tile) {
    // Global -> shared -> register
    kittens::warp::load(x_s, g.X, {0, 0, r_tile, c_tile});
    kittens::warp::load(x_rt_f, x_s);        // half->float promotion handled by TK

    // tile_vec_cv = row-wise max of this 16×128 tile
    kittens::warp::row_max(tile_vec_cv, x_rt_f);   // <-- correct reduction type
    // row_max_cv = max(row_max_cv, tile_vec_cv)
    kittens::warp::max(row_max_cv, row_max_cv, tile_vec_cv);
  }

  // ---------------- Pass 2: row-wise sum of exp(x - row_max) ------------------
  kittens::warp::zero(row_sum_cv);  // start at 0

  for (int c_tile = 0; c_tile < n_tiles; ++c_tile) {
    kittens::warp::load(x_s, g.X, {0, 0, r_tile, c_tile});
    kittens::warp::load(x_rt_f, x_s);

    // Subtract row_max per row: broadcast the column-vector across columns
    kittens::warp::broadcast_row(tmp_rt, row_max_cv);    // tmp_rt = row_max replicated across columns
    kittens::warp::sub(x_rt_f, x_rt_f, tmp_rt);          // x_rt_f = x - max

    // Exponentiate: exp(x) = 2^(x * 1/ln2)
    kittens::warp::mul(x_rt_f, x_rt_f, INV_LN2);
    kittens::warp::exp2(e_rt_f, x_rt_f);

    // Accumulate row-wise sum into row_sum_cv
    kittens::warp::row_sum(tile_vec_cv, e_rt_f);         // 16-lane col_vec
    kittens::warp::add(row_sum_cv, row_sum_cv, tile_vec_cv);
  }

  // Compute reciprocal 1 / row_sum (in a col_vec to match later broadcast)
  CV ones_cv, inv_sum_cv;
  kittens::warp::one(ones_cv);
  kittens::warp::div(inv_sum_cv, ones_cv, row_sum_cv);

  // ---------------- Pass 3: write normalized probabilities --------------------
  for (int c_tile = 0; c_tile < n_tiles; ++c_tile) {
    kittens::warp::load(x_s, g.X, {0, 0, r_tile, c_tile});
    kittens::warp::load(x_rt_f, x_s);

    // x ← x - row_max
    kittens::warp::broadcast_row(tmp_rt, row_max_cv);
    kittens::warp::sub(x_rt_f, x_rt_f, tmp_rt);

    // e = exp(x)
    kittens::warp::mul(x_rt_f, x_rt_f, INV_LN2);
    kittens::warp::exp2(e_rt_f, x_rt_f);

    // e /= row_sum  (broadcast inverse sums across columns)
    kittens::warp::broadcast_row(tmp_rt, inv_sum_cv);
    kittens::warp::mul(e_rt_f, e_rt_f, tmp_rt);

    // Store probabilities (float->half handled by TK store to half ST)
    kittens::warp::store(y_s, e_rt_f);
    kittens::warp::store(g.Y, y_s, {0, 0, r_tile, c_tile});
  }
}

/* ---------------- Host launcher ---------------- */
__host__ void dispatch_micro(micro_globals g) {
  const size_t shmem = g.dynamic_shared_memory();
  cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
  micro_tk<<<g.grid(), g.block(), shmem>>>(g);
  cudaDeviceSynchronize();
}

/* ---------------- PyBind11 --------------------- */
PYBIND11_MODULE(tk_kernels, m) {
  kittens::py::bind_kernel<micro_tk, micro_globals>(
      m, "micro_tk",
      &micro_globals::X, &micro_globals::Y,
      &micro_globals::M, &micro_globals::N);

  kittens::py::bind_function<dispatch_micro, micro_globals>(
      m, "dispatch_micro",
      &micro_globals::X, &micro_globals::Y,
      &micro_globals::M, &micro_globals::N);
}
