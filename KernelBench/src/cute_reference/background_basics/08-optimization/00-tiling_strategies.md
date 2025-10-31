---
topic: "tiling_strategies"
difficulty: "intermediate"
related_topics: ["thread_layouts", "pipelining_patterns", "shared_memory"]
---

# Tiling Strategies in CuTe DSL

## Overview

**Tiling** (or blocking) is the fundamental optimization technique in GPU computing. It divides large problems into smaller "tiles" that fit in fast shared memory, enabling data reuse and reducing global memory traffic.

**Key concepts:**
- What is tiling and why it matters
- Tile size selection
- Hierarchical tiling
- Memory hierarchy mapping
- Performance impact
- Common tiling patterns

---

## What is Tiling?

### The Memory Problem

**Without tiling (naive):**
```python
@cute.kernel
def naive_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    """
    C[M, N] = A[M, K] @ B[K, N]
    Each thread computes one element of C
    """
    tx = cute.arch.thread_idx()[0]
    bx = cute.arch.block_idx()[0]
    
    i = bx * 256 + tx  # Row of C
    
    # Each element requires K loads from A and K loads from B
    for j in range(N):
        acc = 0.0
        for k in range(K):
            acc += A[i, k] * B[k, j]  # 2K global memory accesses!
        C[i, j] = acc
    
    # Total: 2MNK global memory accesses
    # Very slow! Each element loaded many times.
```

**Problem:**
- Every element loaded from global memory many times
- Global memory is slow (~400 GB/s)
- 80-90% of time spent waiting on memory

### With Tiling

**Tiled version:**
```python
@cute.kernel
def tiled_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    """
    Divide matrices into tiles that fit in shared memory
    Load tiles once, reuse many times
    """
    # Shared memory tiles (fast!)
    smem_A = cute.make_smem_tensor(cute.make_shape(TILE_M, TILE_K), dtype)
    smem_B = cute.make_smem_tensor(cute.make_shape(TILE_K, TILE_N), dtype)
    
    # Register accumulator
    acc = cute.make_rmem_tensor(cute.make_shape(THREAD_M, THREAD_N), dtype)
    acc.store(0.0)
    
    # Loop over K dimension in tiles
    for k_tile in range(num_k_tiles):
        # 1. Load tile to shared memory (once)
        load_tile_A(smem_A, A, k_tile)
        load_tile_B(smem_B, B, k_tile)
        cute.arch.syncthreads()
        
        # 2. Compute using tile (many reuses!)
        compute_tile(acc, smem_A, smem_B)
        cute.arch.syncthreads()
    
    # 3. Write result
    store_tile(C, acc)
    
    # Total: Much fewer global memory accesses!
    # Each element loaded once per tile, reused many times.
```

**Benefits:**
- Each element loaded once per tile
- Reused many times from shared memory
- Shared memory is fast (~19 TB/s on Ampere)
- 10-100× speedup!

---

## Tile Size Selection

### Factors Affecting Tile Size

**Competing constraints:**

1. **Shared memory capacity:**
   - Ampere: 48-164 KB per block
   - Hopper: 228 KB per block
   - Must fit A tile + B tile + other data

2. **Register pressure:**
   - Larger tiles → more registers per thread
   - Too many registers → lower occupancy

3. **Occupancy:**
   - Larger blocks → fewer blocks per SM
   - Need enough blocks to hide latency

4. **Compute intensity:**
   - Larger tiles → more reuse → better perf
   - But diminishing returns

### Common Tile Sizes

**GEMM tile sizes:**
```python
# Small tiles (memory-bound, high occupancy)
TILE_M, TILE_N, TILE_K = 64, 64, 16

# Medium tiles (balanced)
TILE_M, TILE_N, TILE_K = 128, 128, 32

# Large tiles (compute-bound, tensor cores)
TILE_M, TILE_N, TILE_K = 128, 256, 64
TILE_M, TILE_N, TILE_K = 256, 128, 64

# Very large (Hopper, specialized)
TILE_M, TILE_N, TILE_K = 256, 256, 64
```

**Shared memory usage:**
```python
def calculate_smem_usage(
    tile_m: int, tile_n: int, tile_k: int,
    dtype_size: int = 2  # Float16 = 2 bytes
) -> int:
    """Calculate shared memory needed for GEMM tiles"""
    
    smem_A = tile_m * tile_k * dtype_size
    smem_B = tile_k * tile_n * dtype_size
    
    total = smem_A + smem_B
    
    print(f"Tile A ({tile_m}×{tile_k}): {smem_A/1024:.1f} KB")
    print(f"Tile B ({tile_k}×{tile_n}): {smem_B/1024:.1f} KB")
    print(f"Total: {total/1024:.1f} KB")
    
    return total

# Example: 128×128×32 tile with Float16
calculate_smem_usage(128, 128, 32)
# Output:
# Tile A (128×32): 8.0 KB
# Tile B (32×128): 8.0 KB
# Total: 16.0 KB
```

### Choosing Tile Size

**Decision tree:**
```python
def choose_tile_size(
    M: int, N: int, K: int,
    architecture: str,
    memory_bound: bool
):
    """Choose appropriate tile size"""
    
    if architecture == "sm80":  # Ampere
        max_smem = 164 * 1024  # 164 KB
        
        if memory_bound:
            # High occupancy more important
            return 64, 64, 32
        else:
            # Compute intensity more important
            return 128, 128, 64
    
    elif architecture == "sm90":  # Hopper
        max_smem = 228 * 1024  # 228 KB
        
        if memory_bound:
            return 128, 128, 64
        else:
            return 256, 128, 64
    
    else:
        # Conservative default
        return 128, 128, 32
```

---

## Hierarchical Tiling

### Multi-Level Tiling

**Grid → Block → Warp → Thread hierarchy:**
```python
# Level 1: Grid tiles (distribute work across SMs)
GRID_TILE_M = 256  # Multiple blocks process this
GRID_TILE_N = 256

# Level 2: Block tiles (shared memory level)
BLOCK_TILE_M = 128
BLOCK_TILE_N = 128
BLOCK_TILE_K = 32

# Level 3: Warp tiles (warp-level operations)
WARP_TILE_M = 32
WARP_TILE_N = 64

# Level 4: Thread tiles (register level)
THREAD_TILE_M = 8
THREAD_TILE_N = 8
```

**Relationship:**
```
BLOCK_TILE_M = NUM_WARPS_M * WARP_TILE_M
WARP_TILE_M = NUM_THREADS_M * THREAD_TILE_M

Example:
128 = 4 warps × 32
32 = 4 threads × 8
```

### Implementing Hierarchical Tiling
```python
@cute.kernel
def hierarchical_gemm(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor
):
    # Block tile configuration
    TILE_M, TILE_N, TILE_K = 128, 128, 32
    
    # Warp tile configuration
    WARP_M, WARP_N = 32, 64
    NUM_WARPS_M = TILE_M // WARP_M  # 4
    NUM_WARPS_N = TILE_N // WARP_N  # 2
    
    # Thread tile configuration
    THREAD_M, THREAD_N = 8, 8
    
    # Get thread/warp/block IDs
    tid = cute.arch.thread_idx()[0]
    warp_id = tid // 32
    lane_id = tid % 32
    
    warp_m = warp_id // NUM_WARPS_N
    warp_n = warp_id % NUM_WARPS_N
    
    bid_m = cute.arch.block_idx()[0]
    bid_n = cute.arch.block_idx()[1]
    
    # Allocate shared memory (block level)
    smem_A = cute.make_smem_tensor(
        cute.make_shape(TILE_M, TILE_K),
        cutlass.Float16
    )
    smem_B = cute.make_smem_tensor(
        cute.make_shape(TILE_K, TILE_N),
        cutlass.Float16
    )
    
    # Allocate registers (thread level)
    acc = cute.make_rmem_tensor(
        cute.make_shape(THREAD_M, THREAD_N),
        cutlass.Float32
    )
    acc.store(0.0)
    
    # K-dimension loop (tile over K)
    for k_tile in range(num_k_tiles):
        # Load to shared memory
        load_tile_collective(smem_A, smem_B, A, B, k_tile)
        cute.arch.syncthreads()
        
        # Each warp processes its portion
        warp_compute(acc, smem_A, smem_B, warp_m, warp_n)
        cute.arch.syncthreads()
    
    # Store results
    store_tile(C, acc, bid_m, bid_n, warp_m, warp_n)
```

---

## Tiling Patterns

### Pattern 1: Square Tiles

**Equal M and N dimensions:**
```python
# Good for square matrices
TILE_M = TILE_N = 128
TILE_K = 32

# Example: 4096×4096 matrix
# Grid: (32, 32, 1) blocks
# Each block: 128×128 tile
```

**When to use:**
- Square or nearly square matrices
- Balanced memory access
- General-purpose kernels

### Pattern 2: Rectangular Tiles

**Different M and N dimensions:**
```python
# Wide tiles (more columns)
TILE_M, TILE_N = 64, 256

# Tall tiles (more rows)
TILE_M, TILE_N = 256, 64
```

**When to use:**
- Non-square matrices
- Match matrix aspect ratio
- Optimize for specific dimensions

### Pattern 3: K-Slicing

**Small K dimension for pipelining:**
```python
# Small K for better pipelining
TILE_M, TILE_N = 128, 128
TILE_K = 16  # Small K → more iterations → better overlap

# vs

# Large K for fewer iterations
TILE_M, TILE_N = 128, 128
TILE_K = 64  # Large K → fewer iterations
```

**Trade-off:**
- Small K: More pipelining opportunities, more overhead
- Large K: Fewer iterations, less pipelining

### Pattern 4: Register Blocking

**Tile within registers:**
```python
@cute.kernel
def register_blocked_gemm():
    # Thread computes 8×8 tile in registers
    THREAD_M, THREAD_N = 8, 8
    
    # Allocate register tile
    frag_C = cute.make_rmem_tensor(
        cute.make_shape(THREAD_M, THREAD_N),
        cutlass.Float32
    )
    
    # Compute outer product into register tile
    for k in range(TILE_K):
        frag_A = load_fragment_A(k)  # 8 elements
        frag_B = load_fragment_B(k)  # 8 elements
        
        # Outer product: 8×8 updates
        for i in range(THREAD_M):
            for j in range(THREAD_N):
                frag_C[i, j] += frag_A[i] * frag_B[j]
```

---

## Memory Hierarchy Mapping

### Three-Level Memory Hierarchy
```
Global Memory (GMEM)     → Block Tiles    → smem
    ↓                                          ↓
Shared Memory (SMEM)     → Warp Tiles     → registers
    ↓                                          ↓
Registers (RMEM)         → Thread Elements → compute
```

**Tile size guidelines:**
```python
# GMEM → SMEM (block tile)
# Size: Limited by shared memory capacity
BLOCK_TILE_M = 128  # Fits in shared memory
BLOCK_TILE_N = 128
BLOCK_TILE_K = 32

# SMEM → RMEM (thread tile)
# Size: Limited by register file
THREAD_TILE_M = 8   # Fits in registers
THREAD_TILE_N = 8

# Threads per block
THREADS = (BLOCK_TILE_M // THREAD_TILE_M) * \
          (BLOCK_TILE_N // THREAD_TILE_N)
        = 16 * 16 = 256
```

### Calculating Tile Hierarchy
```python
def design_tile_hierarchy(
    block_tile_m: int,
    block_tile_n: int,
    thread_tile_m: int,
    thread_tile_n: int
):
    """Calculate thread layout from tile sizes"""
    
    threads_m = block_tile_m // thread_tile_m
    threads_n = block_tile_n // thread_tile_n
    total_threads = threads_m * threads_n
    
    print(f"Block tile: {block_tile_m}×{block_tile_n}")
    print(f"Thread tile: {thread_tile_m}×{thread_tile_n}")
    print(f"Thread layout: {threads_m}×{threads_n}")
    print(f"Total threads: {total_threads}")
    
    if total_threads > 1024:
        print("⚠️  Too many threads per block!")
    elif total_threads % 32 != 0:
        print("⚠️  Threads not multiple of warp size!")
    else:
        print("✓ Valid configuration")
    
    return threads_m, threads_n

# Example
design_tile_hierarchy(128, 128, 8, 8)
# Output:
# Block tile: 128×128
# Thread tile: 8×8
# Thread layout: 16×16
# Total threads: 256
# ✓ Valid configuration
```

---

## Tiling for Different Operations

### Tiling Element-wise Operations

**No data reuse, but still benefit from coalescing:**
```python
@cute.kernel
def tiled_elementwise(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    """Element-wise with tiling for coalescing"""
    
    TILE_SIZE = 256  # Process 256 elements at a time
    
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    # Tile offset
    tile_start = bid * TILE_SIZE
    
    # Process tile
    idx = tile_start + tid
    if idx < cute.size(C):
        C[idx] = A[idx] + B[idx]
```

### Tiling Convolution

**2D spatial tiles:**
```python
@cute.kernel
def tiled_conv2d(
    input: cute.Tensor,   # [C, H, W]
    weight: cute.Tensor,  # [K, C, R, S]
    output: cute.Tensor   # [K, H', W']
):
    """Tiled 2D convolution"""
    
    # Output tile
    TILE_H, TILE_W = 16, 16
    
    # Input tile (includes halo for kernel)
    R, S = cute.shape(weight)[2], cute.shape(weight)[3]
    INPUT_TILE_H = TILE_H + R - 1
    INPUT_TILE_W = TILE_W + S - 1
    
    # Shared memory for input tile
    smem_input = cute.make_smem_tensor(
        cute.make_shape(INPUT_TILE_H, INPUT_TILE_W),
        cutlass.Float16
    )
    
    # Load input tile (with halo)
    load_input_tile(smem_input, input)
    cute.arch.syncthreads()
    
    # Compute convolution on tile
    compute_conv_tile(output, smem_input, weight)
```

### Tiling Reduction

**Tree-based reduction with tiling:**
```python
@cute.kernel
def tiled_reduction(data: cute.Tensor, output: cute.Tensor):
    """Reduction with tiling"""
    
    TILE_SIZE = 1024
    
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    # Shared memory for partial reduction
    smem = cute.make_smem_tensor(
        cute.make_shape(TILE_SIZE),
        cutlass.Float32
    )
    
    # Each thread loads multiple elements (tiled)
    tile_start = bid * TILE_SIZE
    thread_sum = 0.0
    
    for i in range(tid, TILE_SIZE, cute.arch.block_dim_x()):
        idx = tile_start + i
        if idx < cute.size(data):
            thread_sum += data[idx]
    
    # Store to shared memory
    smem[tid] = thread_sum
    cute.arch.syncthreads()
    
    # Tree reduction within block
    stride = cute.arch.block_dim_x() // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        cute.arch.syncthreads()
        stride //= 2
    
    # Write block result
    if tid == 0:
        output[bid] = smem[0]
```

---

## Performance Impact

### Measuring Tiling Benefit

**Benchmark:**
```python
import torch
import time

def benchmark_gemm(M, N, K, tile_size, num_iters=100):
    """Benchmark GEMM with different tile sizes"""
    
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    
    # Warm up
    for _ in range(10):
        gemm_kernel.launch(
            grid=calculate_grid(M, N, tile_size),
            block=calculate_block(tile_size)
        )(A, B, C, tile_size=tile_size)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        gemm_kernel.launch(
            grid=calculate_grid(M, N, tile_size),
            block=calculate_block(tile_size)
        )(A, B, C, tile_size=tile_size)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Calculate performance
    flops = 2 * M * N * K * num_iters
    tflops = flops / elapsed / 1e12
    
    print(f"Tile {tile_size:3d}: {tflops:6.2f} TFLOPS")
    
    return tflops

# Test different tile sizes
print("GEMM Performance (4096×4096×4096):")
for tile_size in [64, 96, 128, 192, 256]:
    benchmark_gemm(4096, 4096, 4096, tile_size)

# Typical output:
# GEMM Performance (4096×4096×4096):
# Tile  64:  12.34 TFLOPS
# Tile  96:  18.45 TFLOPS
# Tile 128:  24.67 TFLOPS  ← Best
# Tile 192:  22.15 TFLOPS
# Tile 256:  19.23 TFLOPS
```

### Roofline Analysis

**Understanding performance limits:**
```python
def roofline_analysis(
    M: int, N: int, K: int,
    tile_m: int, tile_n: int, tile_k: int,
    dtype_size: int = 2,  # Float16
    peak_tflops: float = 312.0,  # A100
    memory_bw: float = 1555.0  # GB/s, A100
):
    """Analyze if tiling is compute or memory bound"""
    
    # Arithmetic intensity (FLOPs per byte)
    flops_per_tile = 2 * tile_m * tile_n * tile_k
    
    bytes_loaded = (tile_m * tile_k + tile_k * tile_n) * dtype_size
    bytes_stored = tile_m * tile_n * dtype_size * 2  # Accumulator
    total_bytes = bytes_loaded + bytes_stored
    
    arithmetic_intensity = flops_per_tile / total_bytes
    
    # Performance limits
    compute_limit = peak_tflops  # TFLOPS
    memory_limit = memory_bw * arithmetic_intensity / 1000  # TFLOPS
    
    print(f"Tile: {tile_m}×{tile_n}×{tile_k}")
    print(f"Arithmetic Intensity: {arithmetic_intensity:.2f} FLOP/byte")
    print(f"Compute Limit: {compute_limit:.1f} TFLOPS")
    print(f"Memory Limit: {memory_limit:.1f} TFLOPS")
    
    if memory_limit < compute_limit:
        print(f"⚠️  Memory Bound (need AI > {compute_limit / memory_bw * 1000:.1f})")
        print(f"   Recommendation: Increase tile size")
    else:
        print(f"✓ Compute Bound")
    
    return min(compute_limit, memory_limit)

# Example
roofline_analysis(4096, 4096, 4096, 128, 128, 32)
```

---

## Advanced Tiling Techniques

### Swizzled Tiling

**Reduce bank conflicts:**
```python
# Instead of linear tile layout
# [0][1][2][3]
# [4][5][6][7]

# Use swizzled layout
# [0][1][2][3]
# [5][4][7][6]  ← Swizzled

smem_layout = cute.make_layout(
    cute.make_shape(TILE_M, TILE_K),
    cute.make_swizzled_stride(TILE_K, 1, swizzle_bits=3)
)
```

### Multi-Buffered Tiling

**Double/triple buffering:**
```python
@cute.kernel
def multi_buffered_gemm():
    """Use multiple buffers for pipelining"""
    
    NUM_STAGES = 3
    
    # Allocate multiple buffers
    smem_A = [
        cute.make_smem_tensor(cute.make_shape(TILE_M, TILE_K), dtype)
        for _ in range(NUM_STAGES)
    ]
    smem_B = [
        cute.make_smem_tensor(cute.make_shape(TILE_K, TILE_N), dtype)
        for _ in range(NUM_STAGES)
    ]
    
    # Pipeline loop
    for k_tile in range(num_k_tiles):
        stage = k_tile % NUM_STAGES
        
        # Load next tile while computing current
        load_tile_async(smem_A[stage], smem_B[stage], k_tile + 1)
        compute_tile(smem_A[stage], smem_B[stage])
```

---

## Best Practices

### ✅ DO

**Start with proven tile sizes:**
```python
# GEMM: 128×128×32 is a good starting point
TILE_M, TILE_N, TILE_K = 128, 128, 32
```

**Measure performance:**
```python
for tile_size in [64, 128, 256]:
    benchmark(tile_size)
```

**Consider shared memory limits:**
```python
smem_usage = (TILE_M * TILE_K + TILE_K * TILE_N) * dtype_size
assert smem_usage < max_smem
```

**Align to warp size:**
```python
threads = (TILE_M // THREAD_M) * (TILE_N // THREAD_N)
assert threads % 32 == 0
```

### ❌ DON'T

**Don't use arbitrarily sized tiles:**
```python
# ✗ BAD: Random sizes
TILE_M, TILE_N = 73, 91  # Inefficient
```

**Don't exceed shared memory:**
```python
# ✗ BAD: Too large
TILE_M, TILE_N, TILE_K = 512, 512, 128  # > 164KB
```

**Don't forget bounds checking:**
```python
# ✗ BAD: No bounds check
# Could access out of bounds
```

---

## Summary

**Tiling:**
- Divides problem into smaller tiles
- Tiles fit in fast shared memory
- Enable data reuse
- 10-100× speedup for memory-bound ops

**Tile size selection:**
- Limited by shared memory capacity
- Balanced with occupancy
- Common GEMM: 128×128×32
- Profile to find optimal

**Hierarchical tiling:**
- Grid tiles (SMs)
- Block tiles (shared memory)
- Warp tiles (warp operations)
- Thread tiles (registers)

**Key insight:** Tiling is the single most important optimization for GPU performance. Get tiling right, and everything else follows.

---

## Next Steps

- [Pipelining Patterns](./pipelining_patterns.md) - Overlap computation and memory
- [Thread Layouts](./thread_layouts.md) - Organize threads for tiling
- [Performance Tips](./performance_tips.md) - Additional optimizations

---

## Further Reading

- [CUTLASS GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)
- [Cache Blocking](https://en.wikipedia.org/wiki/Loop_nest_optimization)