TK_GUIDELINE_PROMPT = """
We are providing an API with simple programming primitives, called ThunderKittens (TK), to simplify the coding task.

Output Requirements: Provide exactly two code blocks with no additional commentary, tests, or print statements. First, a C++/CUDA file building a pybind11 extension named tk_kernels that exports one kernel and one dispatcher. Second, a Python file defining a torch.nn.Module named ModelNew that imports tk_kernels and calls the dispatcher inside forward.

TK API Overview:

The general CUDA workflow with TK follows these steps: define shared memory allocator and tiles, define register memory, load from global to shared using {b, h, d, w} indexing, load from shared to register, perform tile operations, store from register to shared, and store from shared to global.

TK provides tile primitives at two memory levels. Register tiles are declared as rt<bf16, M, N, ducks::rt_layout::row> for computation at warp scope. Shared tiles are allocated via st<bf16, M, N> (&x_s) = al.allocate<st<bf16, M, N>>() for block-level data sharing.

TK operations use destination-first syntax: fn(output, input_a, input_b). Matrix operations include mma_AB(dst, src_a, src_b, accum) for row-major A and col-major B, mma_ABt(dst, src_a, src_b, accum) for both row-major, plus mma_AtB and mma_AtBt variants. Element-wise operations include mul(dst, src_a, src_b), add, sub, exp(dst, src), and zero(dst). Data movement uses load(output, input, {b, h, r, c}) for global-to-shared transfers, store(output, input, {b, h, r, c}) for shared-to-global, and load(output, input) / store(output, input) for shared-register transfers.

Global layouts describe HBM tensors: using x_gl = gl<bf16, -1, -1, -1, -1, st<bf16, TILE_M, TILE_N>>; specifies dtype, four runtime dimensions (batch, head, rows, cols), and tile shape for loads/stores. Access dimensions via g.x.batch, g.x.depth, g.x.rows, g.x.cols. Important: GPU registers and shared memory are limited—tile dimensions should not exceed 64×64, requiring chunked computation for larger tensors.

C++/CUDA File Structure:

Include ThunderKittens headers and namespace: #include "kittens.cuh", #include "pyutils/pyutils.cuh", using namespace kittens;. Define launch configuration constants (e.g., #define NUM_WORKERS (1), #define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)) and tile dimensions as multiples of 16.

Create a micro_globals struct containing all inputs and outputs as TK global layouts plus scalar parameters. Each tensor must be declared as gl<dtype, -1, -1, -1, -1, st<dtype, TILE_M, TILE_N>> with runtime 4D shape (unused logical dimensions indexed with zeros). The struct must define dim3 grid() returning grid dimensions (typically based on output tiling), dim3 block() returning dim3(NUM_THREADS), and optionally size_t dynamic_shared_memory() returning required bytes if using shared memory.

Implement the kernel with signature __global__ __launch_bounds__(NUM_THREADS, 1) void micro_tk(const __grid_constant__ micro_globals g). Inside, set up the shared allocator: extern __shared__ alignment_dummy __shm[]; shared_allocator al((int*)&__shm[0]);. Allocate shared tiles via al.allocate<st<dtype, M, N>>() and register tiles as rt<dtype, M, N, ducks::rt_layout::row> or col depending on layout requirements. Move data using load(shared_tile, g.some_gl, {b, h, r, c}) for global-to-shared, load(reg_tile, shared_tile) for shared-to-register, store(shared_tile, reg_tile) for register-to-shared, and store(g.some_gl, shared_tile, {b, h, r, c}) for shared-to-global. Use __syncthreads() between phases. Ensure shared memory usage does not exceed the dynamic allocation size.

Provide a dispatcher void dispatch_micro(micro_globals g) that optionally calls cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size), launches micro_tk<<<g.grid(), g.block(), mem_size>>>(g), and calls cudaDeviceSynchronize().

Bind using PYBIND11_MODULE(tk_kernels, m) with BIND_KERNEL(m, "micro_tk", micro_tk, micro_globals, /* list all globals/scalars */) and BIND_FUNCTION(m, "dispatch_micro", dispatch_micro, micro_globals, /* same ordered list */). The binding argument order must exactly match the fields declared in micro_globals.

Python File Structure:

Import the extension with import tk_kernels at the top along with standard PyTorch imports. Define class ModelNew(torch.nn.Module) with the same forward signature as the original model. Inside forward, ensure inputs are CUDA tensors (call .cuda() if needed), allocate output tensors on CUDA with correct dtype and shape, call tk_kernels.dispatch_micro(...) passing inputs, outputs, and scalars in the same order as the pybind signature, and return the output tensor. Include no printing, correctness checks, seeding, timing, or test code.
"""