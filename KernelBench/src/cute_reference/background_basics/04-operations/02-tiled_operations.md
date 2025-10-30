---
topic: "tiled_operations"
difficulty: "advanced"
related_topics: ["mma_atoms", "copy_atoms", "partition_operations"]
---

# Tiled Operations in CuTe DSL

## Overview

**Tiled operations** extend single-thread atoms (copy atoms, MMA atoms) to work cooperatively across multiple threads. They describe how a group of threads (warp, thread block) collectively operates on larger tiles of data.

**Key concept:** A tiled operation takes a small atom (e.g., 16×8×16 MMA) and replicates it across threads to process much larger tiles (e.g., 128×128×32).

---

## Why Tiling?

### The Problem with Single Atoms

**MMA atom:** 16×8×16 tile
- Too small for efficient GPU utilization
- Need to process larger matrices (1024×1024 or bigger)
- Single thread/warp can't handle large workloads alone

### The Solution: Tiled Operations

**Tiled MMA:** 128×128×32 tile (using many 16×8×16 atoms)
- Many threads cooperate to process larger tile
- Each thread handles its portion using atoms
- Efficient GPU utilization

---

## Tiled Copy Operations

### make_tiled_copy()

**Purpose:** Create a cooperative copy operation across multiple threads.

**Syntax:**
```
tiled_copy = cute.make_tiled_copy(
    copy_atom,           # Base copy operation
    thread_layout,       # How threads are arranged
    value_layout         # How data values are arranged
)
```

### Basic Tiled Copy Example
```
@cute.kernel
def basic_tiled_copy(gmem_src: cute.Tensor, smem_dst: cute.Tensor):
    # Create tiled copy operation
    # 256 threads cooperatively copy 128×64 tile
    tiled_copy = cute.make_tiled_copy(
        copy_atom=cute.UniversalCopy(),
        thread_layout=cute.make_shape(16, 16),  # 16×16 = 256 threads
        value_layout=cute.make_shape(8, 4)      # Each thread copies 8×4 values
    )
    # Total tile: (16*8) × (16*4) = 128×64
    
    # Get this thread's slice of the copy operation
    tid = cute.arch.thread_idx()[0]
    thread_copy = tiled_copy.get_slice(tid)
    
    # Partition source and destination for this thread
    thread_src = thread_copy.partition_S(gmem_src)
    thread_dst = thread_copy.partition_D(smem_dst)
    
    # Copy this thread's portion
    cute.copy(tiled_copy, thread_src, thread_dst)
    
    # Synchronize after cooperative copy
    cute.arch.syncthreads()
```

### What Happens

**Without tiling (each thread independent):**
```
Thread 0: Copies gmem[0:64]
Thread 1: Copies gmem[64:128]
Thread 2: Copies gmem[128:192]
...
# Inefficient, no coordination
```

**With tiling (cooperative):**
```
Threads 0-15:   Copy rows 0-7,   columns 0-3
Threads 16-31:  Copy rows 8-15,  columns 0-3
Threads 32-47:  Copy rows 16-23, columns 0-3
...
# Efficient, coordinated, coalesced memory access
```

---

## Tiled MMA Operations

### make_tiled_mma()

**Purpose:** Create a cooperative MMA operation across multiple threads/warps.

**Syntax:**
```
tiled_mma = cute.make_tiled_mma(
    mma_atom,        # Base MMA operation (e.g., 16×8×16)
    thread_layout,   # Optional: how threads arranged
    value_layout     # Optional: how values arranged
)
```

### Basic Tiled MMA Example
```
@cute.kernel
def basic_tiled_mma():
    # Base MMA atom: 16×8×16
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Create tiled MMA: 128×128×32 tile
    # Uses multiple warps, each with multiple atoms
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        thread_layout=cute.make_shape(2, 2, 1),  # 2×2 arrangement of atoms
        value_layout=cute.make_shape(8, 8, 2)    # Each position has 8×8×2 repetition
    )
    # Total: (2*8*16) × (2*8*8) × (1*2*16) = 256×128×32
    
    # Get this thread's slice
    tid = cute.arch.thread_idx()[0]
    thread_mma = tiled_mma.get_slice(tid)
    
    # Create fragments for this thread
    frag_A = tiled_mma.make_fragment_A(cute.make_shape(256, 32))
    frag_B = tiled_mma.make_fragment_B(cute.make_shape(32, 128))
    frag_C = tiled_mma.make_fragment_C(cute.make_shape(256, 128))
    
    # Initialize accumulator
    frag_C.store(0.0)
    
    # Perform tiled MMA
    cute.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
```

---

## Understanding Tiled Layouts

### Thread Layout

**Describes how threads are arranged to cover the operation.**
```
# 1D thread layout (256 threads in a line)
thread_layout = cute.make_shape(256)

# 2D thread layout (16×16 = 256 threads in a grid)
thread_layout = cute.make_shape(16, 16)

# 3D thread layout (4×4×16 = 256 threads)
thread_layout = cute.make_shape(4, 4, 16)
```

**Example interpretation (2D):**
```
thread_layout = (16, 16)

Thread arrangement:
Row 0:  T0  T1  T2  ... T15
Row 1:  T16 T17 T18 ... T31
...
Row 15: T240 T241 ... T255
```

### Value Layout

**Describes how many values each thread position processes.**
```
# Each thread handles 8×4 values
value_layout = cute.make_shape(8, 4)

# Each thread handles 16 values (1D)
value_layout = cute.make_shape(16)
```

### Total Tile Size

**Total tile = thread_layout × value_layout × atom_size**

**Example:**
```
mma_atom = 16×8×16 (M×N×K)
thread_layout = (2, 4, 1)
value_layout = (4, 2, 2)

Total tile:
M = 2 * 4 * 16 = 128
N = 4 * 2 * 8  = 64
K = 1 * 2 * 16 = 32

Result: 128×64×32 tile
```

---

## Partitioning with Tiled Operations

### partition_S() and partition_D()

**For tiled copy:**
```
@cute.kernel
def partition_example(gmem: cute.Tensor, smem: cute.Tensor):
    tiled_copy = cute.make_tiled_copy(...)
    
    # Get this thread's slice
    tid = cute.arch.thread_idx()[0]
    thread_copy = tiled_copy.get_slice(tid)
    
    # Partition source (S) for this thread
    thread_src = thread_copy.partition_S(gmem)
    # thread_src contains only the elements this thread should copy from
    
    # Partition destination (D) for this thread
    thread_dst = thread_copy.partition_D(smem)
    # thread_dst contains only the elements this thread should copy to
    
    # Copy this thread's portion
    cute.copy(tiled_copy, thread_src, thread_dst)
```

### partition_A(), partition_B(), partition_C()

**For tiled MMA:**
```
@cute.kernel
def partition_mma_example(smem_A: cute.Tensor, smem_B: cute.Tensor):
    tiled_mma = cute.make_tiled_mma(...)
    
    # Get this thread's MMA slice
    tid = cute.arch.thread_idx()[0]
    thread_mma = tiled_mma.get_slice(tid)
    
    # Partition A matrix for this thread
    thread_A = thread_mma.partition_A(smem_A)
    # thread_A contains only the A elements this thread needs
    
    # Partition B matrix for this thread
    thread_B = thread_mma.partition_B(smem_B)
    # thread_B contains only the B elements this thread needs
    
    # Create accumulator fragment
    thread_C = thread_mma.partition_C(smem_C.shape)
    
    # Load into register fragments
    frag_A = tiled_mma.make_fragment_A(thread_A.shape)
    frag_B = tiled_mma.make_fragment_B(thread_B.shape)
    frag_C = tiled_mma.make_fragment_C(thread_C.shape)
    
    cute.copy(thread_A, frag_A)
    cute.copy(thread_B, frag_B)
    
    # MMA
    cute.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
```

---

## Complete Example: Tiled GEMM

### Full Kernel with Tiled Copy and Tiled MMA
```
@cute.kernel
def tiled_gemm_complete(
    gA: cute.Tensor,  # Global A (M×K)
    gB: cute.Tensor,  # Global B (K×N)
    gC: cute.Tensor   # Global C (M×N)
):
    # Tile sizes
    tile_m, tile_n, tile_k = 128, 128, 32
    
    # ========================================
    # 1. Create Tiled Copy for Global → Shared
    # ========================================
    tiled_copy = cute.make_tiled_copy(
        copy_atom=cute.UniversalCopy(),
        thread_layout=cute.make_shape(32, 8),  # 256 threads
        value_layout=cute.make_shape(4, 4)
    )
    
    # Get thread's copy slice
    tid = cute.arch.thread_idx()[0]
    thread_copy = tiled_copy.get_slice(tid)
    
    # ========================================
    # 2. Create Tiled MMA
    # ========================================
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        thread_layout=cute.make_shape(4, 8, 1),
        value_layout=cute.make_shape(2, 2, 2)
    )
    
    # Get thread's MMA slice
    thread_mma = tiled_mma.get_slice(tid)
    
    # ========================================
    # 3. Allocate Shared Memory
    # ========================================
    smem_A = cute.make_smem_tensor(
        cute.make_shape(tile_m, tile_k),
        cutlass.Float16
    )
    smem_B = cute.make_smem_tensor(
        cute.make_shape(tile_k, tile_n),
        cutlass.Float16
    )
    
    # ========================================
    # 4. Create Register Fragments
    # ========================================
    frag_A = tiled_mma.make_fragment_A(smem_A.shape)
    frag_B = tiled_mma.make_fragment_B(smem_B.shape)
    frag_C = tiled_mma.make_fragment_C(cute.make_shape(tile_m, tile_n))
    
    # Initialize accumulator
    frag_C.store(0.0)
    
    # ========================================
    # 5. Get Block's Tile
    # ========================================
    block_m = cute.arch.block_idx()[0]
    block_n = cute.arch.block_idx()[1]
    
    # Partition global tensors for this block
    block_A = gA[block_m]
    block_B = gB[:, block_n]
    
    # ========================================
    # 6. Main Loop over K
    # ========================================
    num_k_tiles = cute.size(gA, mode=1) // tile_k
    
    for k_tile in range(num_k_tiles):
        # ---- Copy Global → Shared ----
        # Partition for this thread (copy)
        thread_gA = thread_copy.partition_S(block_A[:, k_tile])
        thread_sA = thread_copy.partition_D(smem_A)
        
        thread_gB = thread_copy.partition_S(block_B[k_tile, :])
        thread_sB = thread_copy.partition_D(smem_B)
        
        # Copy
        cute.copy(tiled_copy, thread_gA, thread_sA)
        cute.copy(tiled_copy, thread_gB, thread_sB)
        
        cute.arch.syncthreads()
        
        # ---- Copy Shared → Registers ----
        # Partition for this thread (MMA)
        thread_smem_A = thread_mma.partition_A(smem_A)
        thread_smem_B = thread_mma.partition_B(smem_B)
        
        # Copy to fragments
        cute.copy(thread_smem_A, frag_A)
        cute.copy(thread_smem_B, frag_B)
        
        # ---- MMA ----
        cute.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
        
        cute.arch.syncthreads()
    
    # ========================================
    # 7. Store Result to Global Memory
    # ========================================
    block_C = gC[block_m, block_n]
    thread_C = thread_mma.partition_C(block_C)
    
    cute.copy(frag_C, thread_C)
```

---

## Thread Cooperation Patterns

### Warp-Level Cooperation

**Single warp (32 threads) processes a tile:**
```
@cute.kernel
def warp_tiled_mma():
    # Warp-level tiled MMA
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        thread_layout=cute.make_shape(32, 1, 1)  # Single warp
    )
    
    # Warp cooperates on 16×8×16 tile
    # No cross-warp synchronization needed for MMA
```

### Block-Level Cooperation

**Multiple warps (e.g., 256 threads = 8 warps) process larger tile:**
```
@cute.kernel
def block_tiled_mma():
    # Block-level tiled MMA
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        thread_layout=cute.make_shape(128, 2, 1)  # 256 threads
    )
    
    # Multiple warps cooperate on 128×128 tile
    # Synchronization needed for shared memory
```

---

## Common Tiling Strategies

### Strategy 1: Square Tiles (M = N)

**Good for general matrix multiplication:**
```
# 128×128×32 tile
tile_m = tile_n = 128
tile_k = 32

tiled_mma = cute.make_tiled_mma(
    mma_atom,
    # Arrange for square output tile
    thread_layout=cute.make_shape(4, 4, 1)
)
```

**Benefits:**
- Balanced workload
- Good for square matrices
- Simplifies logic

### Strategy 2: Rectangular Tiles (M ≠ N)

**Good for non-square matrices or specific optimizations:**
```
# 256×64×32 tile (wider in M)
tile_m = 256
tile_n = 64
tile_k = 32

tiled_mma = cute.make_tiled_mma(
    mma_atom,
    # Arrange for rectangular output tile
    thread_layout=cute.make_shape(8, 2, 1)
)
```

**Use when:**
- Input matrices are non-square
- Want more reuse of A or B
- Memory bandwidth constraints

### Strategy 3: Large K Tiles

**Good for compute-bound cases:**
```
# 128×128×64 tile (larger K)
tile_m = tile_n = 128
tile_k = 64

tiled_mma = cute.make_tiled_mma(
    mma_atom,
    thread_layout=cute.make_shape(4, 4, 2)  # More K repetition
)
```

**Benefits:**
- More computation per data load
- Better for compute-bound workloads
- Hides memory latency

---

## Tiling and Register Pressure

### Register Usage

**Larger tiles = more registers per thread:**
```
# Small tile: ~64 registers per thread
tiled_mma_small = cute.make_tiled_mma(
    mma_atom,
    thread_layout=cute.make_shape(2, 2, 1)
)

# Large tile: ~200 registers per thread
tiled_mma_large = cute.make_tiled_mma(
    mma_atom,
    thread_layout=cute.make_shape(8, 8, 2)
)
```

**Trade-off:**
- More registers = larger tiles = better efficiency
- But too many registers = lower occupancy

### Finding the Balance
```
@cute.jit
def find_optimal_tile_size():
    # Try different tile sizes
    tile_sizes = [
        (64, 64, 32),
        (128, 128, 32),
        (256, 256, 32)
    ]
    
    for tile_m, tile_n, tile_k in tile_sizes:
        # Create tiled MMA
        tiled_mma = cute.make_tiled_mma(...)
        
        # Profile performance
        performance = benchmark_kernel(tiled_mma)
        
        print(f"Tile {tile_m}×{tile_n}×{tile_k}: {performance} TFLOPS")
    
    # Choose tile size with best performance
```

---

## Tiling with Different Atom Types

### FP16 Tiled MMA
```
@cute.kernel
def fp16_tiled_mma():
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        thread_layout=cute.make_shape(4, 4, 1)
    )
    # Processes 64×32×16 tile with FP16 inputs
```

### FP8 Tiled MMA (Hopper)
```
@cute.kernel
def fp8_tiled_mma():
    mma_atom = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)
    
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        thread_layout=cute.make_shape(4, 4, 1)
    )
    # Processes 64×32×32 tile with FP8 inputs
    # Note: K dimension is larger (32 vs 16) for FP8
```

### INT8 Tiled MMA
```
@cute.kernel
def int8_tiled_mma():
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x32_I8I8I32I32_TN)
    
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        thread_layout=cute.make_shape(4, 4, 1)
    )
    # Processes 64×32×32 tile with INT8 inputs
```

---

## Advanced: Multi-Stage Tiling

### Hierarchical Tiling

**Tile at multiple levels:**
```
@cute.kernel
def hierarchical_tiling():
    # Level 1: Block-level tile (128×128×32)
    block_tile_m, block_tile_n, block_tile_k = 128, 128, 32
    
    # Level 2: Warp-level tile (32×32×32)
    warp_tile_m, warp_tile_n, warp_tile_k = 32, 32, 32
    
    # Level 3: MMA atom (16×8×16)
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Each warp handles 32×32×32 using multiple MMA atoms
    # Each block has 4×4=16 warps, handling 128×128×32 total
```

---

## Debugging Tiled Operations

### Verify Tile Coverage
```
@cute.kernel
def debug_tile_coverage():
    tiled_mma = cute.make_tiled_mma(...)
    
    tid = cute.arch.thread_idx()[0]
    thread_mma = tiled_mma.get_slice(tid)
    
    # Print which elements this thread handles
    cute.printf("Thread %d handles: ...\n", tid)
    
    # Verify all threads cover the entire tile without overlap
```

### Check Fragment Sizes
```
@cute.jit
def check_fragment_sizes():
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    tiled_mma = cute.make_tiled_mma(mma_atom, ...)
    
    frag_A = tiled_mma.make_fragment_A(cute.make_shape(128, 32))
    frag_B = tiled_mma.make_fragment_B(cute.make_shape(32, 128))
    frag_C = tiled_mma.make_fragment_C(cute.make_shape(128, 128))
    
    print(f"Fragment A: {frag_A.shape} elements")
    print(f"Fragment B: {frag_B.shape} elements")
    print(f"Fragment C: {frag_C.shape} elements")
```

---

## Best Practices

### ✅ DO

**Start with standard tile sizes:**
```
# ✓ GOOD: Common, well-tested sizes
tile_m, tile_n, tile_k = 128, 128, 32
```

**Profile different tile sizes:**
```
# ✓ GOOD: Find optimal for your problem
for tile_size in [64, 128, 256]:
    benchmark(tile_size)
```

**Match tiling to problem size:**
```
# ✓ GOOD: Adjust based on matrix dimensions
if m > 4096:
    tile_m = 256  # Larger tiles for large matrices
else:
    tile_m = 128  # Smaller tiles for small matrices
```

**Use power-of-2 sizes when possible:**
```
# ✓ GOOD: 64, 128, 256 (simplifies indexing)
tile_m = 128
```

### ❌ DON'T

**Don't use tiles that are too small:**
```
# ✗ BAD: Too small, poor efficiency
tile_m, tile_n = 16, 16  # Underutilizes GPU
```

**Don't use tiles that are too large:**
```
# ✗ BAD: Too large, register pressure
tile_m, tile_n = 512, 512  # May reduce occupancy
```

**Don't forget synchronization:**
```
# ✗ BAD: Missing sync after shared memory write
cute.copy(tiled_copy, gmem, smem)
cute.copy(smem, frag)  # Race condition!

# ✓ GOOD: Sync before reading
cute.copy(tiled_copy, gmem, smem)
cute.arch.syncthreads()
cute.copy(smem, frag)
```

---

## Summary

**Tiled operations:**
- Extend atoms to work across multiple threads
- Enable processing of large tiles cooperatively
- Critical for efficient GPU utilization

**Key functions:**
- `make_tiled_copy()` - Cooperative data movement
- `make_tiled_mma()` - Cooperative matrix multiplication
- `.get_slice(tid)` - Get thread's portion
- `.partition_X()` - Partition tensors for thread

**Typical tile sizes:**
- Copy: 128×64, 256×128
- MMA: 128×128×32, 256×128×32
- Larger tiles = better efficiency (up to register limits)

**Best practices:**
- Start with standard sizes (128×128×32)
- Profile to find optimal for your problem
- Balance tile size with register pressure
- Always synchronize shared memory access

---

## Next Steps

- [Partition Operations](./partition_operations.md) - Deep dive into partitioning
- [MMA Atoms](./mma_atoms.md) - Understanding base MMA operations
- [Performance Tuning](../08_optimization/performance_tips.md) - Optimizing tile sizes

---

## Further Reading

- [CuTe Tiling Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md)
- [GEMM Optimization Guide](https://docs.nvidia.com/cuda/cublas/index.html)