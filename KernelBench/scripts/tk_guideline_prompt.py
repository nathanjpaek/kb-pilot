TK_GUIDELINE_PROMPT = """
You are writing CUDA kernels using ThunderKittens (TK). You MUST use the ThunderKittens API (kittens::…) to implement the kernel.

=====================
OUTPUT CONTRACT
=====================
Produce **exactly two code blocks** and nothing else:
1) A **C++/CUDA** file that builds a **pybind11** extension module named **tk_kernels** exporting **one kernel** and **one dispatcher**.
2) A **Python** file defining **torch.nn.Module** named **ModelNew** that imports **tk_kernels** and calls the **dispatcher** inside **forward**.

No prints, tests, timing, seeding, or commentary in either file.

=====================
GOLDEN RULES (must follow)
=====================
- Always call TK APIs with the **kittens::** prefix. For ops, use **kittens::warp::** or **kittens::warpgroup::**; never bare names.
- Do **NOT** use `using namespace kittens`.
- For global layouts, **never** put `0` in template dims. Use `-1` for runtime dims.
- Favor pipelining: Global → **TMA async** → Shared → Warp loads → Compute → Async store to Global.
- Use **semaphores** and `wait`/`arrive` to overlap producer/consumer work.
- For matvec: prefer broadcast + elementwise + reduction. For matmul: use **MMA**.
- Zero accumulators before use (`kittens::warp::zero`), avoid unnecessary syncs, and pick the lightest scope.
- Register tiles do compute (**rt_***), shared tiles stage data (**st_***), {r,s}v_* for vectors.
- Tensor-core MMAs: **bf16** inputs with **float** accumulators. Tile dims ≤ 64x64; chunk larger problems.
- Shared tile **width must match** register tile width for load/store group ops.
- Grid/block config must match warp layout (e.g., 4 warps = 2x2 macro-tile = 32x32).

- You MUST use ONLY the APIs outlined in this prompt, or used in the given examples. DO NOT ASSUME OR MAKE UP ANY OTHER APIs.

=====================
MENTAL MODEL / WORKFLOW
=====================
1) Define shared-memory allocator and tiles.
2) Define register tiles/vectors for compute.
3) Load from **global** to **shared** (TMA async) using tile indices {b, h, r_tile, c_tile}.
4) Warp-load from **shared** to **registers**.
5) Do compute (elementwise / reductions / MMA).
6) Store back: registers → shared, then async store to global.
7) Use semaphores to overlap stages; prefer warp scope syncs.

=====================
API CHEATSHEET (fully qualified)
=====================
Types & Layout
- Tiles/vectors (bf16 shown; analogous float/half types exist):
  - `kittens::rt_bf<H, W>` : register tile (compute)
  - `kittens::st_bf<H, W>` : shared tile (staging)
  - `kittens::rv_bf<N>`    : register vector
  - `kittens::sv_bf<N>`    : shared vector
- Global layouts (runtime dims = -1):
  - `kittens::gl<T, B, D, R, C, ...SharedTypes>`
  - Example: `kittens::gl<kittens::bf16, 1, -1, -1, 2048, kittens::st_bf<16,512>> W;`
  - Members: `T* raw_ptr;`
             `template<int axis> size_t shape() const;`
             `template<int axis> size_t stride() const;`
  - Indexing uses **tile indices**: `gW[kittens::coord<>{b, h, r_tile, c_tile}]`
  - For batched ops use **3D indexing** `{batch, row, col}` (not `{batch, 0, row, col}`).

Warp-scope Memory Ops
- Shared ↔ Registers:
  - `kittens::warp::load(kittens::rt_bf<H,W>& dst, const kittens::st_bf<H,W>& src)`
  - `kittens::warp::store(kittens::st_bf<H,W>& dst, const kittens::rt_bf<H,W>& src)`
  - `kittens::warp::load(kittens::rv_bf<N>& dst, const kittens::sv_bf<N>& src)`
  - `kittens::warp::store(kittens::sv_bf<N>& dst, const kittens::rv_bf<N>& src)`
- Direct small reads from global:
  - `kittens::warp::load(kittens::rt_bf<H,W>& dst, const kittens::gl<...>& g, kittens::coord<>{...})`
  - `kittens::warp::load(kittens::rv_bf<N>&  dst, const kittens::gl<...>& g, kittens::coord<>{offset})`

TMA (Async Global↔Shared)
- Declare/issue:
  - `kittens::tma::expect(kittens::semaphore& sem, kittens::st_bf<H,W>& tile_or_sv)`
  - `kittens::tma::load_async(kittens::st_bf<H,W>& dst_smem, kittens::gl<...>& src_gmem, kittens::coord<>{...}, kittens::semaphore& sem)`
  - `kittens::tma::store_async(kittens::gl<...>& dst_gmem, const kittens::sv_bf<N>& src_smem, kittens::coord<>{...})`
  - `kittens::tma::store_add_async(...)`
  - `kittens::tma::store_async_wait()`       // ensure visibility
  - `kittens::tma::store_async_read_wait()`  // ensure read-side hazard clear
- Pattern:
  - Producer: `tma::expect(...); tma::load_async(...);`
  - Consumer: `wait(sem, threshold); kittens::warp::load(...)` from shared
  - After consuming: `kittens::warp::arrive(done_sem, count);`

Sync & Semaphores
- `kittens::warp::sync()`
- `kittens::group<N>::sync(barrier_id)`  // N warps
- `kittens::warpgroup::sync()`
- `kittens::init_semaphore(kittens::semaphore&, int initial)`
- `kittens::wait(kittens::semaphore&, int threshold)`
- `kittens::warp::arrive(kittens::semaphore&, int delta=1)`
- Rare fence: `asm volatile("fence.acq_rel.gpu;");`

Math
- Set/fill:
  - `kittens::warp::zero(x), one(x), pos_infty(x), neg_infty(x)`
  - For setting to a specific scalar value: `kittens::warp::mul(tile, tile, scalar_value);  // after zeroing`
- Elementwise (tile/vector or scalar rhs):
  - `kittens::warp::add(dst, a, b)`, `sub`, `mul`, `div`
  - Scalars OK: `kittens::warp::mul(tile, tile, scalar)`
- Broadcast/layout:
  - `kittens::warp::broadcast_col(rt, row_vec)`
  - `kittens::warp::broadcast_row(rt, col_vec)`
  - `kittens::warp::transpose_inplace(rt)` → returns ref
  - `kittens::warp::swap_layout_inplace(rt)` → switch row/col view
- Apply lambdas:
  - `kittens::warp::apply(rv_dst, rv_src, Lambda)`
  - `kittens::warp::apply(rt_dst, rt_src, Lambda)`

Reductions
- Tile → vector:
  - `kittens::warp::row_sum(col_vec, rt)`  // sum across columns per row
  - `kittens::warp::col_sum(row_vec, rt)`
  - `row_max`, `col_max` (also on shared tiles)
- Vector → scalar:
  - `auto s = kittens::warp::sum(const kittens::rv_bf<N>&)`

MMA (Tensor Cores)
- Use bf16 inputs with float accumulators:
  - Inputs: `kittens::rt_bf<M,N, kittens::ducks::rt_layout::row|col>`
  - Accum:  `kittens::rt_fl<M,N, kittens::ducks::rt_layout::row>`
- Warp MMAs:
  - `kittens::warp::mma_AB(C, A, B, C)`
  - `kittens::warp::mma_ABt(C, A, B, C)`
  - `kittens::warp::mma_AtB(C, A, B, C)`
- Warpgroup MMAs:
  - `kittens::warpgroup::mma_AB(C, A, B)` and friends
  - `kittens::warpgroup::mma_async_wait()`
- Example TMA/compute/store:
```
// loader
kittens::tma::expect(inp_sem, weight_smem);
kittens::tma::load_async(weight_smem, g.W, kittens::coord<>{layer, col_block, tile_id}, inp_sem);
// consumer
kittens::wait(inp_sem, 0);
kittens::warp::load(Wt, weight_smem);
// ... compute ...
kittens::tma::store_async(g.O, out_smem_vec, kittens::coord<>{out_block});
kittens::tma::store_async_wait();
```

=====================
C++/CUDA FILE REQUIREMENTS
=====================
- Includes:
- `#include "kittens.cuh"`
- `#include "pyutils/pyutils.cuh"`
- No `using namespace kittens;`
- Launch config:
- e.g., `#define NUM_WORKERS (1)`
- `#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)`
- Tile dimensions are multiples of 16.
- micro_globals:
- Contains all inputs/outputs as TK global layouts + scalar params.
- Each tensor as: `kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, TILE_M, TILE_N>>`
  with runtime 4D shape (unused logical dims may be indexed with zeros).
- Kernel signature:
- `__global__ __launch_bounds__(NUM_THREADS, 1) void micro_tk(const __grid_constant__ micro_globals g)`
- Shared allocator (must use alignment_dummy):
- ```
  extern __shared__ kittens::alignment_dummy __shm[];
  kittens::shared_allocator al((int*)&__shm[0]);
  ```
- Allocate tiles:
- Shared: `auto& x_s = al.allocate<kittens::st<kittens::half, M, N>>();`
- Registers: `kittens::rt<kittens::half, M, N, kittens::ducks::rt_layout::row|col> x_rt;`
- **Match shared/register widths** for group load/store.
- Dispatcher:
- `void dispatch_micro(micro_globals g)`:
  - Optionally `cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);`
  - Launch: `micro_tk<<<g.grid(), g.block(), mem_size>>>(g);`
  - `cudaDeviceSynchronize();`
- PyBind11 binding (member pointers; order matches struct fields):
- ```
  PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(m, "micro_tk",
      &micro_globals::A, &micro_globals::B, &micro_globals::C, &micro_globals::M, &micro_globals::N);
    kittens::py::bind_function<dispatch_micro, micro_globals>(m, "dispatch_micro",
      &micro_globals::A, &micro_globals::B, &micro_globals::C, &micro_globals::M, &micro_globals::N);
  }
  ```

=====================
PYTHON FILE REQUIREMENTS
=====================
- `import tk_kernels` and standard PyTorch imports at top.
- Define:
- `class ModelNew(torch.nn.Module):`
  - `def forward(self, ...):`
    - Ensure inputs on CUDA (`.cuda()` as needed).
    - Allocate outputs on CUDA with correct `dtype`/shape.
    - Call `tk_kernels.dispatch_micro(...)` with args in the **exact** order as in the PyBind signature.
    - Return the output tensor.
- No printing, checks, seeding, timing, or tests.

=====================
COMMON PITFALLS / FIXES (strict)
=====================
1) Never declare `using dtype = fp16;` (causes compile errors).
2) Never use `0` in global layout template dims—use `-1` for runtime dims.
3) For tensor cores: use `kittens::rt_bf<>` inputs with `kittens::rt_fl<>` accumulators.
4) Always call ops via `kittens::warp::…` (or `kittens::warpgroup::…`), never unqualified.
5) Use `kittens::alignment_dummy __shm[]` for shared memory (not `int __shm[]`).
6) For batched ops, index `{batch, row, col}` (3D), not `{batch, 0, row, col}`.
7) Use `__host__` (not `KITTENS_HOST_DEVICE`) for host functions.
8) For scalar→half convert, use `__float2half()` (not `kittens::to_half()`).
9) In pybind, pass **member pointers** (`&Class::member`), not string names.
10) Use a fixed integer (e.g., `100000`) for `dynamic_shared_memory()` sizing; avoid template-size expressions.
11) For scalar broadcasting to tiles: `kittens::warp::zero(tile); kittens::warp::add(tile, tile, scalar);`
12) Allocate shared tiles with `auto& x_s = al.allocate<...>();` (not `*al.allocate`).
13) For matmul use `kittens::warp::mma_AB(accum, A, B, accum)` (not elementwise mul+add).
14) Global layout coordinates are **tile indices** `{b, h, r_tile, c_tile}` (not element offsets).
15) Match shared/register **widths** to avoid “Group load/store requires tile widths to match”.
16) Align grid/block with actual warp count (e.g., 4 warps = 2×2 layout = 32×32).

=====================
CHECKLIST BEFORE YOU FINISH
=====================
- Exactly two code blocks produced: (1) C++/CUDA pybind module **tk_kernels** with `micro_tk` and `dispatch_micro`; (2) Python `ModelNew` calling the dispatcher in `forward`.
- All TK calls are fully qualified with `kittens::…`.
- No extra text, prints, tests, or timing code.
"""


