---
topic: "partition_operations"
difficulty: "advanced"
related_topics: ["tiled_operations", "mma_atoms", "layouts"]
---

# Partition Operations in CuTe DSL

## Overview

**Partitioning** is the process of dividing tensors among threads, warps, or thread blocks so that each computational unit processes its assigned portion of data. Partitioning is fundamental to parallel GPU programming in CuTe DSL.

**Key concept:** A large tensor is partitioned into smaller views, with each thread/warp/block getting a unique slice to work on independently.

---

## Why Partitioning?

### The Problem

**Large tensor:** 1024×1024 matrix
- Too large for one thread
- Need 256 threads to process efficiently
- How do we divide the work?

### The Solution

**Partition the tensor:**
- Divide into 16×16 = 256 tiles of 64×64 each
- Each thread gets one 64×64 tile
- All threads work in parallel

---

## Types of Partitioning

### 1. Spatial Partitioning (Coordinate-Based)

**Divide by coordinates:** Each thread gets different spatial region.
```
@cute.kernel
def spatial_partition(data: cute.Tensor):
    # 256 threads, each gets 4 elements
    # data is 1024 elements total
    tid = cute.arch.thread_idx()[0]
    
    # Each thread processes 4 contiguous elements
    start = tid * 4
    for i in range(4):
        process(data[start + i])
    
    # Thread 0: data[0:4]
    # Thread 1: data[4:8]
    # Thread 2: data[8:12]
    # ...
```

### 2. Strided Partitioning

**Divide by stride:** Each thread gets every Nth element.
```
@cute.kernel
def strided_partition(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    num_threads = cute.arch.block_dim_x()
    
    # Each thread processes every num_threads-th element
    for i in range(tid, cute.size(data), num_threads):
        process(data[i])
    
    # Thread 0: data[0, 256, 512, 768, ...]
    # Thread 1: data[1, 257, 513, 769, ...]
    # Thread 2: data[2, 258, 514, 770, ...]
    # ...
```

### 3. Hierarchical Partitioning

**Divide at multiple levels:** Block → Warp → Thread.
```
@cute.kernel
def hierarchical_partition(data: cute.Tensor):
    # Level 1: Partition among blocks
    bid = cute.arch.block_idx()[0]
    block_data = data[bid]
    
    # Level 2: Partition among warps within block
    wid = cute.arch.thread_idx()[0] // 32
    warp_data = block_data[wid]
    
    # Level 3: Partition among threads within warp
    lane = cute.arch.thread_idx()[0] % 32
    thread_data = warp_data[lane]
    
    process(thread_data)
```

---

## CuTe Partition Functions

### partition_S() - Partition Source

**Use for copy operations - partition the source tensor.**
```
@cute.kernel
def partition_source_example(gmem: cute.Tensor, smem: cute.Tensor):
    tiled_copy = cute.make_tiled_copy(...)
    
    tid = cute.arch.thread_idx()[0]
    thread_copy = tiled_copy.get_slice(tid)
    
    # Partition source (global memory)
    thread_src = thread_copy.partition_S(gmem)
    # thread_src is a VIEW of gmem containing only this thread's elements
    
    thread_dst = thread_copy.partition_D(smem)
    
    # Copy this thread's portion
    cute.copy(tiled_copy, thread_src, thread_dst)
```

**What it does:**
- Takes full tensor (e.g., 128×64)
- Returns view containing only this thread's source elements
- Different threads get different, non-overlapping views

### partition_D() - Partition Destination

**Use for copy operations - partition the destination tensor.**
```
@cute.kernel
def partition_destination_example(gmem: cute.Tensor, smem: cute.Tensor):
    tiled_copy = cute.make_tiled_copy(...)
    
    tid = cute.arch.thread_idx()[0]
    thread_copy = tiled_copy.get_slice(tid)
    
    thread_src = thread_copy.partition_S(gmem)
    
    # Partition destination (shared memory)
    thread_dst = thread_copy.partition_D(smem)
    # thread_dst is a VIEW of smem containing only this thread's destination
    
    cute.copy(tiled_copy, thread_src, thread_dst)
```

### partition_A() - Partition A Matrix

**Use for MMA operations - partition the A (left) matrix.**
```
@cute.kernel
def partition_a_example(smem_A: cute.Tensor):
    tiled_mma = cute.make_tiled_mma(...)
    
    tid = cute.arch.thread_idx()[0]
    thread_mma = tiled_mma.get_slice(tid)
    
    # Partition A matrix for this thread
    thread_A = thread_mma.partition_A(smem_A)
    # thread_A contains only the A elements this thread needs for MMA
    
    # Load into register fragment
    frag_A = tiled_mma.make_fragment_A(thread_A.shape)
    cute.copy(thread_A, frag_A)
```

### partition_B() - Partition B Matrix

**Use for MMA operations - partition the B (right) matrix.**
```
@cute.kernel
def partition_b_example(smem_B: cute.Tensor):
    tiled_mma = cute.make_tiled_mma(...)
    
    tid = cute.arch.thread_idx()[0]
    thread_mma = tiled_mma.get_slice(tid)
    
    # Partition B matrix for this thread
    thread_B = thread_mma.partition_B(smem_B)
    # thread_B contains only the B elements this thread needs for MMA
    
    frag_B = tiled_mma.make_fragment_B(thread_B.shape)
    cute.copy(thread_B, frag_B)
```

### partition_C() - Partition C Matrix (Accumulator)

**Use for MMA operations - partition the C (output/accumulator) matrix.**
```
@cute.kernel
def partition_c_example(gmem_C: cute.Tensor):
    tiled_mma = cute.make_tiled_mma(...)
    
    tid = cute.arch.thread_idx()[0]
    thread_mma = tiled_mma.get_slice(tid)
    
    # Partition C matrix for this thread
    thread_C = thread_mma.partition_C(gmem_C)
    # thread_C contains only the C elements this thread computes
    
    # Create accumulator fragment
    frag_C = tiled_mma.make_fragment_C(thread_C.shape)
    frag_C.store(0.0)
    
    # After MMA, store back
    cute.copy(frag_C, thread_C)
```

---

## Understanding Partitioned Views

### What is a Partitioned View?

**Not a copy - it's a view (reference) to a subset of the original tensor.**
```
@cute.kernel
def partition_is_view():
    # Original tensor: 1024 elements
    tensor = cute.make_tensor(cute.make_shape(1024), cutlass.Float32)
    
    # Partition for thread 5
    tiled_copy = cute.make_tiled_copy(...)
    thread_copy = tiled_copy.get_slice(5)
    thread_view = thread_copy.partition_S(tensor)
    
    # thread_view is NOT a copy!
    # It's a view that references elements [20:24] of tensor (example)
    
    # Writing to thread_view modifies original tensor
    thread_view[0] = 42.0
    # This writes to tensor[20] (not a separate copy)
```

### Visualization of Partitioning
```
Original tensor (128 elements):
[0, 1, 2, 3, 4, 5, ... 127]

4 threads, each gets 32 elements:

Thread 0 partition: [0:32]
[0, 1, 2, ... 31]

Thread 1 partition: [32:64]
[32, 33, 34, ... 63]

Thread 2 partition: [64:96]
[64, 65, 66, ... 95]

Thread 3 partition: [96:128]
[96, 97, 98, ... 127]
```

---

## Partition Patterns

### Pattern 1: Block-Level Partitioning

**Each block processes a tile of the full tensor.**
```
@cute.kernel
def block_partition(gmem_data: cute.Tensor):
    # Tensor is 4096 elements
    # 16 blocks, each processes 256 elements
    
    bid = cute.arch.block_idx()[0]
    
    # Each block gets its tile
    block_tile = gmem_data[bid * 256 : (bid + 1) * 256]
    
    # Block 0: elements [0:256]
    # Block 1: elements [256:512]
    # ...
    
    process_block_tile(block_tile)
```

### Pattern 2: Thread-Level Partitioning Within Block

**Within each block, threads further partition the data.**
```
@cute.kernel
def thread_partition_within_block(gmem_data: cute.Tensor):
    # Each block processes 256 elements
    # 256 threads per block
    
    bid = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    
    # Block tile
    block_tile = gmem_data[bid * 256 : (bid + 1) * 256]
    
    # Each thread gets 1 element
    thread_element = block_tile[tid]
    
    process(thread_element)
```

### Pattern 3: 2D Partitioning

**Partition a 2D tensor among 2D thread layout.**
```
@cute.kernel
def partition_2d(gmem_matrix: cute.Tensor):
    # Matrix is 1024×1024
    # Grid is 16×16 blocks
    # Each block is 16×16 threads
    
    # Block coordinates
    block_x = cute.arch.block_idx()[0]
    block_y = cute.arch.block_idx()[1]
    
    # Thread coordinates
    thread_x = cute.arch.thread_idx()[0]
    thread_y = cute.arch.thread_idx()[1]
    
    # Each block processes 64×64 tile
    block_tile = gmem_matrix[
        block_x * 64 : (block_x + 1) * 64,
        block_y * 64 : (block_y + 1) * 64
    ]
    
    # Each thread within block processes 4×4 subtile
    thread_tile = block_tile[
        thread_x * 4 : (thread_x + 1) * 4,
        thread_y * 4 : (thread_y + 1) * 4
    ]
    
    process_2d_tile(thread_tile)
```

### Pattern 4: Cyclic Partitioning

**Distribute work cyclically for load balancing.**
```
@cute.kernel
def cyclic_partition(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    num_threads = cute.arch.block_dim_x()
    
    # Process every num_threads-th element
    for i in range(tid, cute.size(data), num_threads):
        process(data[i])
    
    # Good for irregular workloads
    # Ensures balanced load across threads
```

---

## Partitioning for Copy Operations

### GMEM → SMEM Copy
```
@cute.kernel
def gmem_to_smem_partition(gmem: cute.Tensor, smem: cute.Tensor):
    # Create tiled copy
    tiled_copy = cute.make_tiled_copy(
        copy_atom=cute.UniversalCopy(),
        thread_layout=cute.make_shape(32, 8),  # 256 threads
        value_layout=cute.make_shape(4, 4)     # Each thread: 4×4 values
    )
    
    tid = cute.arch.thread_idx()[0]
    thread_copy = tiled_copy.get_slice(tid)
    
    # Partition source (global)
    thread_gmem = thread_copy.partition_S(gmem)
    # Thread 0 gets gmem[0:16] (example)
    # Thread 1 gets gmem[16:32]
    # ...
    
    # Partition destination (shared)
    thread_smem = thread_copy.partition_D(smem)
    # Thread 0 gets smem[0:16]
    # Thread 1 gets smem[16:32]
    # ...
    
    # Copy: each thread copies its portion
    cute.copy(tiled_copy, thread_gmem, thread_smem)
    
    cute.arch.syncthreads()
```

### SMEM → RMEM Copy
```
@cute.kernel
def smem_to_rmem_partition(smem: cute.Tensor):
    tiled_copy = cute.make_tiled_copy(...)
    
    tid = cute.arch.thread_idx()[0]
    thread_copy = tiled_copy.get_slice(tid)
    
    # Partition shared memory
    thread_smem = thread_copy.partition_S(smem)
    
    # Create register fragment
    rmem_fragment = cute.make_rmem_tensor(thread_smem.shape, cutlass.Float32)
    
    # Copy shared → registers
    cute.copy(thread_smem, rmem_fragment)
    
    # Each thread now has its portion in registers
```

---

## Partitioning for MMA Operations

### Complete MMA Partitioning Example
```
@cute.kernel
def mma_partition_complete(
    smem_A: cute.Tensor,  # 128×32
    smem_B: cute.Tensor,  # 32×128
    gmem_C: cute.Tensor   # 128×128
):
    # Create tiled MMA
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        thread_layout=cute.make_shape(4, 4, 1)
    )
    
    tid = cute.arch.thread_idx()[0]
    thread_mma = tiled_mma.get_slice(tid)
    
    # ========================================
    # Partition A matrix
    # ========================================
    thread_A = thread_mma.partition_A(smem_A)
    # Thread 0 gets A[0:32, 0:8] (example)
    # Thread 1 gets A[0:32, 8:16]
    # ...different threads get different rows/cols
    
    frag_A = tiled_mma.make_fragment_A(thread_A.shape)
    cute.copy(thread_A, frag_A)
    
    # ========================================
    # Partition B matrix
    # ========================================
    thread_B = thread_mma.partition_B(smem_B)
    # Thread 0 gets B[0:8, 0:32] (example)
    # Thread 1 gets B[8:16, 0:32]
    # ...
    
    frag_B = tiled_mma.make_fragment_B(thread_B.shape)
    cute.copy(thread_B, frag_B)
    
    # ========================================
    # Partition C matrix (accumulator)
    # ========================================
    thread_C = thread_mma.partition_C(gmem_C)
    # Thread 0 gets C[0:32, 0:32] (example)
    # Thread 1 gets C[0:32, 32:64]
    # ...
    
    frag_C = tiled_mma.make_fragment_C(thread_C.shape)
    frag_C.store(0.0)
    
    # ========================================
    # MMA: Each thread computes its portion
    # ========================================
    cute.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
    
    # ========================================
    # Store result: Each thread stores its portion
    # ========================================
    cute.copy(frag_C, thread_C)
```

---

## Hierarchical Partitioning Example

### Three-Level Hierarchy: Grid → Block → Thread
```
@cute.kernel
def three_level_partition(gmem_data: cute.Tensor):
    # Data is 1M elements (1024×1024)
    
    # ========================================
    # Level 1: Grid-level (blocks)
    # ========================================
    # 256 blocks, each handles 4096 elements
    bid_x = cute.arch.block_idx()[0]
    bid_y = cute.arch.block_idx()[1]
    
    # Block gets 64×64 tile
    block_tile = gmem_data[
        bid_x * 64 : (bid_x + 1) * 64,
        bid_y * 64 : (bid_y + 1) * 64
    ]
    # 4096 elements per block
    
    # ========================================
    # Level 2: Warp-level
    # ========================================
    # 256 threads = 8 warps
    # Each warp handles 32×16 subtile
    tid = cute.arch.thread_idx()[0]
    warp_id = tid // 32
    warp_row = warp_id // 4
    warp_col = warp_id % 4
    
    warp_tile = block_tile[
        warp_row * 16 : (warp_row + 1) * 16,
        warp_col * 16 : (warp_col + 1) * 16
    ]
    # 256 elements per warp
    
    # ========================================
    # Level 3: Thread-level
    # ========================================
    # 32 threads per warp
    # Each thread handles 8 elements
    lane = tid % 32
    thread_row = lane // 4
    thread_col = lane % 4
    
    thread_tile = warp_tile[
        thread_row * 2 : (thread_row + 1) * 2,
        thread_col * 2 : (thread_col + 1) * 2
    ]
    # 4 elements per thread (2×2)
    
    # Process thread's portion
    for i in range(2):
        for j in range(2):
            process(thread_tile[i, j])
```

---

## Partition Alignment and Coalescing

### Aligned Partitioning

**Ensure partitions align with memory boundaries:**
```
@cute.kernel
def aligned_partition(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # ✓ GOOD: Each thread gets 4 floats (16 bytes)
    # Aligned to 16-byte boundary
    start = tid * 4
    for i in range(4):
        val = data[start + i]
    
    # Threads 0-31 access consecutive memory
    # Coalesced into single 128-byte transaction
```

### Coalesced Access Pattern

**Adjacent threads access adjacent memory:**
```
@cute.kernel
def coalesced_partition(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # ✓ GOOD: Coalesced access
    # Thread 0: data[0]
    # Thread 1: data[1]
    # Thread 2: data[2]
    # ...all in one memory transaction
    val = data[tid]
    
    # ✗ BAD: Strided access
    # Thread 0: data[0]
    # Thread 1: data[128]
    # Thread 2: data[256]
    # ...32 separate memory transactions!
    bad_val = data[tid * 128]
```

---

## Dynamic vs Static Partitioning

### Static Partitioning (Compile-Time)

**Partition based on compile-time constants:**
```
@cute.kernel
def static_partition(data: cute.Tensor, tile_size: cutlass.Constexpr):
    tid = cute.arch.thread_idx()[0]
    
    # tile_size is compile-time constant
    start = tid * tile_size
    
    # Compiler knows exact partition layout
    # Can optimize aggressively
    for i in cutlass.range_constexpr(tile_size):
        process(data[start + i])
```

### Dynamic Partitioning (Runtime)

**Partition based on runtime values:**
```
@cute.kernel
def dynamic_partition(data: cute.Tensor, tile_size: cutlass.Int32):
    tid = cute.arch.thread_idx()[0]
    
    # tile_size is runtime value
    start = tid * tile_size
    
    # Compiler can't optimize as much
    # More flexible
    for i in range(tile_size):
        process(data[start + i])
```

---

## Partition Verification

### Checking Coverage

**Ensure all elements are covered exactly once:**
```
@cute.jit
def verify_partition_coverage():
    import torch
    
    # Create test tensor
    data = torch.arange(1024, device="cuda")
    
    # Mark elements as processed
    processed = torch.zeros(1024, dtype=torch.bool, device="cuda")
    
    # Run kernel that marks processed elements
    partition_kernel.launch(...)(data, processed)
    
    # Check coverage
    assert torch.all(processed), "Not all elements processed!"
    print("✓ All elements covered")
```

### Checking for Overlaps

**Ensure no two threads process same element:**
```
@cute.kernel
def debug_partition_overlap():
    tid = cute.arch.thread_idx()[0]
    
    # Partition
    tiled_copy = cute.make_tiled_copy(...)
    thread_copy = tiled_copy.get_slice(tid)
    thread_view = thread_copy.partition_S(data)
    
    # Mark which indices this thread accesses
    for idx in thread_view.indices():
        cute.printf("Thread %d accesses index %d\n", tid, idx)
    
    # Manually verify no overlaps in output
```

---

## Common Partitioning Mistakes

### Mistake 1: Incorrect Partition Size
```
@cute.kernel
def wrong_partition_size(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # ✗ WRONG: 256 threads × 4 elements = 1024
    # But data has 1000 elements!
    start = tid * 4
    for i in range(4):
        data[start + i] = 0  # Out of bounds for some threads!
    
    # ✓ CORRECT: Check bounds
    start = tid * 4
    for i in range(4):
        idx = start + i
        if idx < cute.size(data):
            data[idx] = 0
```

### Mistake 2: Non-Coalesced Partitioning
```
@cute.kernel
def non_coalesced_partition(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # ✗ BAD: Strided access (poor coalescing)
    for i in range(4):
        val = data[tid + i * 256]
    
    # ✓ GOOD: Contiguous access (coalesced)
    start = tid * 4
    for i in range(4):
        val = data[start + i]
```

### Mistake 3: Unbalanced Partitions
```
@cute.kernel
def unbalanced_partition(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # ✗ BAD: First threads do more work
    # Thread 0: processes 100 elements
    # Thread 255: processes 1 element
    elements_per_thread = (cute.size(data) - tid * 100) // 256
    
    for i in range(elements_per_thread):
        process(data[tid * 100 + i])
    
    # ✓ GOOD: Balanced
    elements_per_thread = cute.size(data) // 256
    start = tid * elements_per_thread
    for i in range(elements_per_thread):
        process(data[start + i])
```

---

## Best Practices

### ✅ DO

**Partition for coalesced access:**
```
# ✓ GOOD: Adjacent threads → adjacent memory
tid = cute.arch.thread_idx()[0]
val = data[tid]
```

**Use tiled operations for complex partitioning:**
```
# ✓ GOOD: Let CuTe handle partitioning
tiled_copy = cute.make_tiled_copy(...)
thread_view = tiled_copy.get_slice(tid).partition_S(data)
```

**Verify partition coverage:**
```
# ✓ GOOD: Test that all elements processed
assert all_elements_processed(data)
```

**Balance workload:**
```
# ✓ GOOD: Equal work per thread
elements_per_thread = total_elements // num_threads
```

### ❌ DON'T

**Don't use strided access:**
```
# ✗ BAD: Poor coalescing
val = data[tid * stride]  # if stride is large
```

**Don't partition without bounds checking:**
```
# ✗ BAD: May access out of bounds
data[tid * 4] = 0  # What if tid * 4 >= size?
```

**Don't create unbalanced partitions:**
```
# ✗ BAD: Irregular partition sizes
# Some threads idle while others work
```

---

## Summary

**Partitioning:**
- Divides data among threads/warps/blocks
- Essential for parallel GPU execution
- Must ensure coverage and no overlaps

**Key partition functions:**
- `partition_S()` - Source for copy
- `partition_D()` - Destination for copy
- `partition_A/B/C()` - Matrices for MMA
- `.get_slice(tid)` - Get thread's slice

**Best practices:**
- Coalesce memory accesses
- Balance workload
- Check bounds
- Verify coverage
- Use tiled operations for complex partitioning

**Common patterns:**
- Block → Warp → Thread hierarchy
- 2D spatial partitioning
- Cyclic for load balancing

---

## Next Steps

- [Tiled Operations](./tiled_operations.md) - Using tiled operations for partitioning
- [Copy Atoms](./copy_atoms.md) - Partitioned data movement
- [MMA Atoms](./mma_atoms.md) - Partitioned matrix operations

---

## Further Reading

- [CuTe Partitioning Guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md)
- [CUDA Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)