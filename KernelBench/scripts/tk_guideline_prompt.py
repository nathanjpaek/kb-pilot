TK_GUIDELINE_PROMPT = """
We are providing an API with simple programming primitives, called ThunderKittens (TK), to simplify the coding task.

Output Requirements: Provide exactly two code blocks with no additional commentary, tests, or print statements. First, a C++/CUDA file building a pybind11 extension named tk_kernels that exports one kernel and one dispatcher. Second, a Python file defining a torch.nn.Module named ModelNew that imports tk_kernels and calls the dispatcher inside forward.

TK API Overview:

The general CUDA workflow with TK follows these steps: define shared memory allocator and tiles, define register memory, load from global to shared using {b, h, d, w} indexing, load from shared to register, perform tile operations, store from register to shared, and store from shared to global.

TK provides tile primitives at two memory levels. Register tiles are declared as rt<kittens::half, M, N, kittens::ducks::rt_layout::row> for computation at warp scope. Shared tiles are allocated via auto& x_s = al.allocate<kittens::st<kittens::half, M, N>>() for block-level data sharing. For H100 with torch.float16 inputs, always use kittens::half (not fp16, not bf16). NEVER use "using dtype = fp16;" - this will cause compilation errors.

CRITICAL: For tensor core operations (mma_AB, mma_ABt, etc.), use bf16 inputs with float accumulators:
- Input register tiles: kittens::rt_bf<M, N, kittens::ducks::rt_layout::row> 
- Accumulator register tiles: kittens::rt_fl<M, N, kittens::ducks::rt_layout::row>
- This prevents "static assertion failed" errors on H100 tensor cores.

TK operations use destination-first syntax: fn(output, input_a, input_b). For matrix multiplication, use kittens::warp::mma_AB(dst, src_a, src_b, accum) for proper GEMM across K dimension - NOT elementwise mul+add. Other matrix operations include kittens::warp::mma_ABt(dst, src_a, src_b, accum) for both row-major, plus kittens::warp::mma_AtB and kittens::warp::mma_AtBt variants. Element-wise operations include kittens::warp::mul(dst, src_a, src_b), kittens::warp::add, kittens::warp::sub, kittens::warp::exp(dst, src), and kittens::warp::sub(dst, dst, dst) for zero. Data movement uses kittens::warp::load(output, input, {b, h, r, c}) for global-to-shared transfers, kittens::warp::store(output, input, {b, h, r, c}) for shared-to-global, and kittens::warp::load(output, input) / kittens::warp::store(output, input) for shared-register transfers.

Global layouts describe HBM tensors: using x_gl = kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, TILE_M, TILE_N>>; specifies dtype, four runtime dimensions (batch, head, rows, cols), and tile shape for loads/stores. Access dimensions via g.x.batch, g.x.depth, g.x.rows, g.x.cols. Access raw data pointer via g.x.raw_ptr for direct memory access when needed. CRITICAL: Global layout coordinates {b, h, r, c} are TILE INDICES, not element offsets. Use tile indices for all loads/stores. 

CRITICAL: NEVER use 0 for global layout dimensions - always use -1 for runtime dimensions. Using 0 causes "Invalid compile-time dimension value" errors.

For batched operations, use: kittens::gl<kittens::half, -1, -1, -1, -1, sub_tile> and index with {batch_idx, row, col} (3D) not {batch_idx, 0, row, col} (4D).

Important: GPU registers and shared memory are limited—tile dimensions should not exceed 64 x 64, requiring chunked computation for larger tensors.

C++/CUDA File Structure:

Include ThunderKittens headers: #include "kittens.cuh", #include "pyutils/pyutils.cuh". Do NOT use "using namespace kittens" - always prefix with kittens::. Define launch configuration constants (e.g., #define NUM_WORKERS (1), #define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)) and tile dimensions as multiples of 16.

Create a micro_globals struct containing all inputs and outputs as TK global layouts plus scalar parameters. Each tensor must be declared as kittens::gl<kittens::half, -1, -1, -1, -1, kittens::st<kittens::half, TILE_M, TILE_N>> with runtime 4D shape (unused logical dimensions indexed with zeros). The struct must define dim3 grid() returning grid dimensions (typically based on output tiling), dim3 block() returning dim3(NUM_THREADS), and optionally size_t dynamic_shared_memory() returning required bytes if using shared memory.

Implement the kernel with signature __global__ __launch_bounds__(NUM_THREADS, 1) void micro_tk(const __grid_constant__ micro_globals g). Inside, set up the shared allocator: extern __shared__ kittens::alignment_dummy __shm[]; kittens::shared_allocator al((int*)&__shm[0]);. Allocate shared tiles via auto& x_s = al.allocate<kittens::st<kittens::half, M, N>>() and register tiles as kittens::rt<kittens::half, M, N, kittens::ducks::rt_layout::row> or col depending on layout requirements. CRITICAL: Shared tile width must match register tile width for load/store operations to avoid "Group load/store requires tile widths to match" errors. 

CRITICAL: For tensor core operations, use bf16 inputs with float accumulators:
- Input register tiles: kittens::rt_bf<M, N, kittens::ducks::rt_layout::row>
- Accumulator register tiles: kittens::rt_fl<M, N, kittens::ducks::rt_layout::row>
- This prevents "static assertion failed" errors on H100 tensor cores.

Move data using kittens::warp::load(shared_tile, g.some_gl, {b, h, r, c}) for global-to-shared, kittens::warp::load(reg_tile, shared_tile) for shared-to-register, kittens::warp::store(shared_tile, reg_tile) for register-to-shared, and kittens::warp::store(g.some_gl, shared_tile, {b, h, r, c}) for shared-to-global. Use __syncthreads() between phases. Ensure shared memory usage does not exceed the dynamic allocation size. For matrix multiplication, use kittens::warp::mma_AB(accum, a_rt, b_rt, accum) instead of elementwise operations. Store results directly from registers to global when tile shapes match.

Provide a dispatcher void dispatch_micro(micro_globals g) that optionally calls cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size), launches micro_tk<<<g.grid(), g.block(), mem_size>>>(g), and calls cudaDeviceSynchronize().

Bind using PYBIND11_MODULE(tk_kernels, m) with kittens::py::bind_kernel<micro_tk, micro_globals>(m, "micro_tk", &micro_globals::A, &micro_globals::B, &micro_globals::C, &micro_globals::M, &micro_globals::N) and kittens::py::bind_function<dispatch_micro, micro_globals>(m, "dispatch_micro", &micro_globals::A, &micro_globals::B, &micro_globals::C, &micro_globals::M, &micro_globals::N). Use member pointers (&Class::member), not string literals. The binding argument order must exactly match the fields declared in micro_globals.

CRITICAL COMPILATION FIXES:
1. NEVER use "using dtype = fp16;" - causes compilation errors
2. NEVER use 0 for global layout dimensions - always use -1 for runtime
3. For tensor cores: use kittens::rt_bf<> inputs with kittens::rt_fl<> accumulators
4. Always use kittens::warp:: prefix for operations, never bare kittens::
5. Use kittens::alignment_dummy __shm[] for shared memory, not int __shm[]
6. For batched ops: use 3D indexing {batch, row, col} not 4D {batch, 0, row, col}
7. Use __host__ instead of KITTENS_HOST_DEVICE for host functions
8. Use __float2half() instead of kittens::to_half() for scalar conversion
9. Use member pointers (&Class::member) in pybind bindings, not string literals
10. Use fixed size (e.g., 100000) for dynamic_shared_memory(), not template calculations
11. For scalar broadcasting: use kittens::warp::zero() then kittens::warp::add(tile, tile, scalar)
12. Use auto& x_s = al.allocate<kittens::st<kittens::half, M, N>>() for shared tiles, NOT *al.allocate
13. For matrix multiplication: use kittens::warp::mma_AB(accum, a_rt, b_rt, accum), NOT elementwise mul+add
14. Global layout coordinates are TILE INDICES {b, h, r_tile, c_tile}, NOT element offsets
15. Match shared tile width to register tile width to avoid "Group load/store requires tile widths to match"
16. Align grid/block dimensions with actual warp count (4 warps = 2×2 layout = 32×32 macro-tile)

Python File Structure:

Import the extension with import tk_kernels at the top along with standard PyTorch imports. Define class ModelNew(torch.nn.Module) with the same forward signature as the original model. Inside forward, ensure inputs are CUDA tensors (call .cuda() if needed), allocate output tensors on CUDA with correct dtype and shape, call tk_kernels.dispatch_micro(...) passing inputs, outputs, and scalars in the same order as the pybind signature, and return the output tensor. Include no printing, correctness checks, seeding, timing, or test code.
"""