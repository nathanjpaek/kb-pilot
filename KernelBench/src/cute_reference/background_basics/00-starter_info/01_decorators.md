---
topic: "decorators"
difficulty: "beginner"
related_topics: ["overview", "quick_start", "code_generation"]
---

# CuTe DSL Decorators

## Overview

CuTe DSL provides two main Python decorators for generating optimized code via dynamic compilation:

1. **@cute.jit** — Host-side JIT-compiled functions
2. **@cute.kernel** — GPU kernel functions

Both decorators transform Python functions into optimized executable code, but they serve different purposes and have different execution contexts.

---

## @cute.jit Decorator

### Purpose
Declares JIT-compiled functions that can be invoked from Python or from other CuTe DSL functions. These functions typically run on the CPU (host) and orchestrate GPU execution.

### Basic Usage

Example usage of @cute.jit decorator for host-side function.

### Decorator Parameters

#### preprocessor (bool, default=True)

Controls how Python control flow is handled:

**preprocessor=True (default)** — Automatic expansion
- Loops and conditionals are automatically converted to IR
- Works correctly with dynamic control flow
- Recommended for most use cases

**preprocessor=False** — Manual control (tracing only)
- Only straight-line code works reliably
- Loops/branches may not work as expected
- Use only for simple arithmetic operations

**When to use preprocessor=False:**
- Simple arithmetic-only functions
- When you need fastest possible compilation
- When you're certain there's no control flow

**When to use preprocessor=True (default):**
- Any function with loops or conditionals
- When correctness is more important than compile time
- Most use cases (recommended default)

### Call-Site Parameters

#### no_cache (bool, default=False)

Controls JIT caching behavior. First call compiles and caches. Second call reuses cached version. Use no_cache=True to force recompilation.

**Use no_cache=True when:**
- Debugging compilation issues
- Testing different optimization flags
- Measuring compilation time
- Working with rapidly changing code

### Examples

**Simple host function:**
Function that prepares tensors for GEMM operation by checking dimensions and creating output tensor.

**With explicit caching control:**
Function that selects different kernels based on tensor size threshold.

---

## @cute.kernel Decorator

### Purpose
Defines GPU kernel functions that are compiled as specialized GPU code and launched with explicit grid/block configuration.

### Basic Usage

Example kernel that performs vector addition using thread and block indices.

### Decorator Parameters

#### preprocessor (bool, default=True)

Same behavior as @cute.jit - controls whether loops and control flow are preserved in the generated IR.

### Kernel Launch Parameters

kernel_function.launch(
grid=[grid_x, grid_y, grid_z],
block=[block_x, block_y, block_z],
cluster=(cluster_x, cluster_y, cluster_z),  # Optional
smem=shared_memory_bytes,  # Optional
stream=cuda_stream,  # Optional
**compile_time_args  # Optional static arguments
)

#### Required Parameters

**grid** (list[int, int, int])
- Specifies the grid dimensions (number of blocks)
- Format: [blocks_x, blocks_y, blocks_z]

**block** (list[int, int, int])
- Specifies the block dimensions (threads per block)
- Format: [threads_x, threads_y, threads_z]

#### Optional Parameters

**cluster** (tuple[int, int, int], optional)
- Cluster dimensions for Hopper+ architectures
- Enables thread block clusters and additional features
- Format: (cluster_x, cluster_y, cluster_z)

**smem** (int, optional)
- Shared memory size in bytes
- Dynamically allocated shared memory

**stream** (cuda.CUstream, optional)
- CUDA stream for asynchronous execution
- Default: uses default stream

**Compile-time arguments** (kwargs)
- Any parameters marked with cutlass.Constexpr in the kernel signature
- These are baked into the compiled kernel

### Examples

**Simple vector add:**
Kernel launched with 1D grid and blocks to perform vector addition.

**Matrix multiplication with shared memory:**
Kernel launched with 2D grid, specifying shared memory size and compile-time tile size.

**Using clusters (Hopper+):**
Kernel using 2x2 cluster of blocks for Hopper architecture features.

---

## Calling Conventions

The following table shows which calling patterns are allowed:

| Caller | Callee | Allowed | Compilation/Runtime |
|--------|--------|---------|---------------------|
| Python function | @jit | ✅ | DSL runtime |
| Python function | @kernel | ❌ | N/A (error raised) |
| @jit | @jit | ✅ | Compile-time call, inlined |
| @jit | Python function | ✅ | Compile-time call, inlined |
| @jit | @kernel | ✅ | Dynamic call via GPU driver or runtime |
| @kernel | @jit | ✅ | Compile-time call, inlined |
| @kernel | Python function | ✅ | Compile-time call, inlined |
| @kernel | @kernel | ❌ | N/A (error raised) |

**Key points:**
- Python can call @jit functions directly
- Python CANNOT call @kernel functions directly (must use .launch())
- @jit functions can launch @kernel functions
- @kernel functions CANNOT call other @kernel functions

---

## Choosing Between @jit and @kernel

| Use @cute.jit when: | Use @cute.kernel when: |
|----------------------|-------------------------|
| Orchestrating GPU work from CPU | Writing GPU computation |
| Preparing/validating data | Performing parallel computation |
| Making decisions about which kernel to launch | Implementing the actual kernel |
| Interfacing with Python code | Need explicit thread/block control |

### Typical Pattern

A common pattern involves a @cute.kernel doing the actual GPU computation and a @cute.jit host function that orchestrates the kernel launch and manages memory.

---

## Advanced: Combining Decorators

You can nest @jit functions and launch multiple kernels from within them to create complex pipelines.

---

## Common Pitfalls

### ❌ Forgetting .launch() on kernels

Wrong: Calling kernel directly without .launch()
Correct: Using .launch() with grid and block parameters

### ❌ Using @kernel for host code

Wrong: Using @kernel decorator for CPU work like reshaping
Correct: Use @jit for host work

### ❌ Control flow without preprocessor

May not work correctly: Using preprocessor=False with loops
Works correctly: Use default preprocessor=True for control flow

---

## Next Steps

- [Quick Start](./quick_start.md) - Write your first kernel
- [Code Generation](../01_fundamentals/code_generation.md) - Understand how decorators generate code
- [Control Flow](../02_control_flow/loops.md) - Learn about loops and conditionals

## Related Documentation

- [JIT Function Arguments](../01_fundamentals/jit_arguments.md)
- [JIT Caching](../06_compilation/jit_caching.md)
- [Kernel Launch Examples](../10_examples/)