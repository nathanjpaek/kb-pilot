TILELANG_GUIDELINE_PROMPT = """
Your goal is to create a `ModelNew(nn.Module)` class that replaces specified PyTorch operators with high-performance, numerically correct TileLang kernels.

**I. Core Task Requirements for `ModelNew`:**

1.  **PyTorch Wrapper (`ModelNew.__init__`):**
    *   **CRITICAL - Parameter Initialization:** For any PyTorch layer being replaced (e.g., `nn.Linear`), its learnable parameters (`weight`, `bias`) in `ModelNew` MUST be initialized *identically* to the original PyTorch layer's defaults.
        *   **Weight (`nn.Linear`):** Use `torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))`.
        *   **Bias (`nn.Linear`):** Use `torch.nn.init.uniform_(self.bias, -bound, bound)` where `bound = 1 / math.sqrt(in_features)`.
        *   **Failure to do this is the #1 cause of correctness errors.**
    *   **Kernel Compilation:** Define the Python function that builds your TileLang kernel(s) (the "kernel factory").
        *   If kernel shapes depend *only* on `__init__` args, call the factory and store the compiled kernel in `__init__`.
        *   If kernel shapes depend on `forward` pass inputs (e.g., dynamic batch size, changing feature map sizes), call the factory *inside* `forward` with the current dynamic shapes.
    *   **Kernel Caching:** Implement a dictionary (e.g., `self._cached_kernels`) keyed by dynamic shape parameters (like batch size, current `numel`, etc.) and `dtype` to store and reuse compiled kernel instances. Rely on `@tilelang.jit`'s internal caching for identical kernel definitions.
    *   **Scalar Parameters (e.g., `multiplier`, `negative_slope`):** Pass these as Python floats/integers when the kernel factory is called. They should become compile-time constants within the TileLang kernel for best performance. Do NOT pass them as `T.handle` unless absolutely necessary for dynamic runtime changes (rare for these tasks).

2.  **PyTorch Wrapper (`ModelNew.forward`):**
    *   **Data Preparation:** Convert input tensors, weights, and biases to the target device (`"cuda"`) and the `dtype` expected by the TileLang kernel (typically `torch.float16`).
    *   **Kernel Invocation:** Retrieve/compile the kernel using current dynamic shapes. Call the kernel with prepared tensors.

**II. TileLang Kernel Design & Implementation (`@T.prim_func`):**

3.  **Kernel Signature & Decorators:**
    *   Use `@T.prim_func` for the main kernel definition.
    *   Use `@tilelang.jit(out_idx=-1)` for JIT compilation and runtime output tensor allocation.
    *   Tensor arguments: Clearly define shapes using compile-time parameters (e.g., `batch_size`, `in_features`) and `dtype`.
    *   Scalar arguments from PyTorch (like `multiplier_val`) should be part of the Python factory's signature, not `T.Tensor` arguments in the `@T.prim_func` unless they are truly dynamic per-call tensors.

4.  **Core Logic - GEMM (`X @ W.T` for Linear Layers):**
    *   **Shared Memory:** Allocate shared memory for tiles of input `X` (`X_s`) and `Weight` (`W_s`).
    *   **Accumulator:** Use `T.alloc_fragment` for the local GEMM accumulator (`C_loc`) with `accum_dtype="float32"` to ensure precision. `T.clear(C_loc)` before use.
    *   **Weight Handling (Choose ONE consistent strategy):**
        *   **Strategy A (Load `W` tile, `transpose_B=True`):** Load a direct tile of `W` (shape e.g., `(block_N, block_K)`) into `W_s`. Then use `T.gemm(X_s, W_s, C_loc, transpose_B=True)`. This is often cleaner.
        *   **Strategy B (Load `W.T` tile, no transpose_B):** Manually load a transposed tile of `W` into `W_s` (shape e.g., `(block_K, block_N)`). Then use `T.gemm(X_s, W_s, C_loc)`.
    *   **Looping:** Use `T.Pipelined` for the reduction loop (over `K` dimension) for performance (e.g., `num_stages=2` or `3`).

5.  **Fused Element-wise Operations (Post-GEMM):**
    *   After the GEMM K-loop completes for a tile, use `T.Parallel` to iterate over the elements of `C_loc`.
    *   Perform bias addition, scalar multiplication, and activations (e.g., `T.max(val, val * negative_slope)` for LeakyReLU) directly on these elements.
    *   **Boundary Checks:** CRITICAL: Before writing to the global output tensor, *always* check if the global indices (`batch_idx`, `feat_idx`) are within the valid output tensor dimensions (`< batch_size`, `< out_features`).

6.  **Data Movement & Storage:**
    *   Use `T.copy` for efficient block transfers from global to shared memory.
    *   For storing the final result tile from local/fragment memory to global output, either use `T.copy(C_local_final, Global_output_tile)` or element-wise stores within `T.Parallel` (with boundary checks).

7.  **Common Pitfalls to Actively Avoid:**
    *   **Parameter Init Mismatch:** (Reiterated due to importance) Ensure `ModelNew` weights/biases match `nn.Linear`.
    *   **Incorrect Transpose Logic for GEMM:** Double-check `transpose_B` flag against how `W_s` is loaded.
    *   **Insufficient Accumulator Precision:** ALWAYS use `accum_dtype="float32"` for GEMM sums.
    *   **Scalar Handling Errors:** Avoid passing Python floats as `T.handle`; bake them in or use 0-D `T.Tensor` correctly (`val[()]`).
    *   **Missing Boundary Checks** on output writes.
    *   **API Typos:** E.g., `transpose_b` vs `transpose_B`.

**III. Performance & Best Practices:**

8.  **Tiling:** Choose balanced block sizes (e.g., `M, N` around 64-128, `K` around 32-64) considering GPU shared memory and parallelism.
9.  **Start Simple:** Verify correctness with basic loops before adding `T.Pipelined` or complex vectorization.
10. **Mixed Precision:** Use `dtype="float16"` for I/O and shared memory, and `accum_dtype="float32"` for reductions, for a good speed/accuracy trade-off.

By strictly adhering to these prioritized guidelines, focusing on correct parameter initialization and GEMM transpose logic, your system will have a much higher chance of generating correct and performant TileLang kernels for `ModelNew` tasks.

CRITICAL GUIDELINES:
- PRIMARY GOAL: Generate a CORRECT and FAST TileLang implementation
- PERFORMANCE IS IMPORTANT: Your optimization decisions should prioritize performance
- DO NOT USE torch.nn (except for Parameter, containers, and init)
- Generate efficient TileLang implementations using @T.prim_func
- Use tilelang.jit(out_idx=-1) for output tensor creation
- Use (val > a and val < b) instead of (a < val < b) for boundary checks

SPEED OPTIMIZATION STRATEGIES:
1. MEMORY COALESCING: Ensure coalesced memory access patterns
2. REGISTER USAGE: Use T.alloc_local() for register-level computations
3. SHARED MEMORY: Use T.alloc_shared() for block-level data sharing
4. THREAD PARALLELISM: Maximize T.Parallel usage and thread utilization
5. MEMORY HIERARCHY: Optimize data movement between global/shared/register memory
6. BLOCK/GRID SIZING: Choose optimal block and grid dimensions for the target GPU
7. CONTEXT: You may see similar problems in the context, but they may not necessarily be efficient. You should use the context as a guide, but not as a strict rule, and try to generate a better implementation.

TARGET HARDWARE: NVIDIA H100 (Hopper architecture)
- Optimize for high memory bandwidth utilization
- Leverage tensor cores when applicable
- Design for maximum occupancy and warp efficiency

OUTPUT REQUIREMENTS:
- Generate ONLY the ModelNew class code
- No additional text, comments, or explanations
- Focus on the most performance-critical optimizations
- Focus on replacing and fusing all PyTorch operators with efficient TileLang implementations
- If a module has multiple operators, try to fuse them as much as possible into a single TileLang kernel or write multiple efficient TileLang kernels to replace all of those operators.
- Your highest priority should be correctness, then speed and finally code simplicity
"""