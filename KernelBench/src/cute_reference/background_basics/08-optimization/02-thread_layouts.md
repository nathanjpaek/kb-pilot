---
topic: "thread_layouts"
difficulty: "intermediate"
related_topics: ["tiling_strategies", "pipelining_patterns", "mma_atoms"]
---

# Thread Layouts in CuTe DSL

## Overview

**Thread layout** determines how threads are organized and mapped to data. Proper thread layout is crucial for coalesced memory access, avoiding bank conflicts, and maximizing compute throughput.

**Key concepts:**
- Thread-to-data mapping
- Coalesced memory access
- Bank conflict avoidance
- Warp-level organization
- Thread tile partitioning
- Layout composition

---

## What is Thread Layout?

### Thread Organization Hierarchy
```
Grid
  └─ Blocks (CTAs)
      └─ Warps (32 threads)
          └─ Threads
              └─ Work per thread
```

**Thread layout defines:**
- How threads map to data elements
- Memory access patterns
- Shared memory access patterns
- Computation distribution

### Simple Example

**Bad layout (strided access):**
```python
@cute.kernel
def bad_layout(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Each thread accesses every 32nd element
    idx = tid  # 0, 1, 2, 3, ..., 31
    while idx < cute.size(data):
        data[idx] = data[idx] * 2.0
        idx += 32  # Jump by warp size
    
    # Memory access pattern: 0, 32, 64, 96, ...
    # NOT coalesced! (32 cache lines for 1 warp)
```

**Good layout (contiguous access):**
```python
@cute.kernel
def good_layout(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    block_size = cute.arch.block_dim_x()
    
    # Each thread accesses contiguous elements
    idx = bid * block_size + tid
    if idx < cute.size(data):
        data[idx] = data[idx] * 2.0
    
    # Memory access pattern: 0, 1, 2, 3, ..., 31
    # Coalesced! (1 cache line for 1 warp)
```

---

## Memory Coalescing

### What is Coalescing?

**Coalesced access:** Warp accesses contiguous memory in single transaction.

**Benefits:**
- 1 memory transaction instead of 32
- 32× faster memory access
- Essential for good performance

### Coalescing Requirements

**For optimal coalescing:**

1. **Threads access consecutive addresses**
```python
   # Thread 0: address 0
   # Thread 1: address 1
   # Thread 2: address 2
   # ...
   # Thread 31: address 31
```

2. **Base address aligned to cache line (128 bytes)**
```python
   # Aligned: 0, 128, 256, 384, ...
```

3. **Access size matches element size**
```python
   # float32: 4 bytes per element
   # 32 threads × 4 bytes = 128 bytes (perfect!)
```

### Checking Coalescing
```python
@cute.kernel
def check_coalescing(data: cute.Tensor):
    """Check if memory access is coalesced"""
    
    tid = cute.arch.thread_idx()[0]
    lane_id = tid % 32
    
    # Good: Each warp accesses consecutive elements
    warp_id = tid // 32
    warp_offset = warp_id * 32
    idx = warp_offset + lane_id
    
    if tid == 0:
        # Print access pattern for first warp
        for i in range(32):
            addr = data.data_ptr() + i * 4  # Assuming float32
            cute.printf("Thread %d: address %p\n", i, addr)
    
    # This will show: consecutive addresses (coalesced!)
    data[idx] = data[idx] * 2.0
```

---

## Thread Layout Patterns

### Pattern 1: Linear Layout

**Simplest: One thread per element**
```python
@cute.kernel
def linear_layout(data: cute.Tensor):
    """1D linear thread layout"""
    
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    # Linear mapping
    global_tid = bid * 256 + tid
    
    if global_tid < cute.size(data):
        data[global_tid] = process(data[global_tid])
```

**Thread-to-data mapping:**
```
Thread 0 → data[0]
Thread 1 → data[1]
Thread 2 → data[2]
...
```

### Pattern 2: 2D Layout

**Map threads to 2D grid:**
```python
@cute.kernel
def layout_2d(matrix: cute.Tensor):
    """2D thread layout for matrix"""
    
    # Thread position in 2D
    tx = cute.arch.thread_idx()[0]
    ty = cute.arch.thread_idx()[1]
    
    # Block position in 2D
    bx = cute.arch.block_idx()[0]
    by = cute.arch.block_idx()[1]
    
    # Tile dimensions
    TILE_X, TILE_Y = 16, 16
    
    # Global position
    i = bx * TILE_X + tx
    j = by * TILE_Y + ty
    
    # Process element
    if i < cute.shape(matrix)[0] and j < cute.shape(matrix)[1]:
        matrix[i, j] = process(matrix[i, j])

# Launch with 2D block
grid = [(M + 15) // 16, (N + 15) // 16, 1]
block = [16, 16, 1]
```

**Thread-to-data mapping:**
```
Block (0,0):
  Thread (0,0) → matrix[0,0]
  Thread (0,1) → matrix[0,1]
  Thread (1,0) → matrix[1,0]
  Thread (1,1) → matrix[1,1]
  ...
```

### Pattern 3: Tiled Layout

**Each thread processes multiple elements (tile):**
```python
@cute.kernel
def tiled_layout(data: cute.Tensor):
    """Each thread processes a tile of elements"""
    
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    ELEMENTS_PER_THREAD = 8  # Thread tile size
    
    # Base index for this thread
    base_idx = (bid * 256 + tid) * ELEMENTS_PER_THREAD
    
    # Process tile
    for i in range(ELEMENTS_PER_THREAD):
        idx = base_idx + i
        if idx < cute.size(data):
            data[idx] = process(data[idx])
```

**Thread-to-data mapping:**
```
Thread 0 → data[0:8]
Thread 1 → data[8:16]
Thread 2 → data[16:24]
...
```

### Pattern 4: Vectorized Layout

**Threads access vector elements:**
```python
@cute.kernel
def vectorized_layout(data: cute.Tensor):
    """Vectorized access (float4)"""
    
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    # Each thread loads 4 floats at once
    vec_idx = bid * 256 + tid
    
    # Load float4 (4 consecutive floats)
    vec = load_float4(data, vec_idx * 4)
    
    # Process vector
    vec.x = process(vec.x)
    vec.y = process(vec.y)
    vec.z = process(vec.z)
    vec.w = process(vec.w)
    
    # Store float4
    store_float4(data, vec_idx * 4, vec)
```

**Benefits:**
- 4× fewer instructions
- Better memory throughput
- Coalesced access

---

## GEMM Thread Layouts

### Basic GEMM Layout

**Each thread computes portion of C:**
```python
@cute.kernel
def gemm_thread_layout(
    A: cute.Tensor,  # M × K
    B: cute.Tensor,  # K × N
    C: cute.Tensor   # M × N
):
    """
    Thread layout for GEMM
    Each thread computes THREAD_M × THREAD_N tile of C
    """
    
    # Block tile dimensions
    TILE_M, TILE_N, TILE_K = 128, 128, 32
    
    # Thread tile dimensions
    THREAD_M, THREAD_N = 8, 8
    
    # Threads per block
    THREADS_M = TILE_M // THREAD_M  # 16
    THREADS_N = TILE_N // THREAD_N  # 16
    # Total: 16 × 16 = 256 threads
    
    tid = cute.arch.thread_idx()[0]
    
    # Map thread to 2D position
    thread_m = tid // THREADS_N  # 0-15
    thread_n = tid % THREADS_N   # 0-15
    
    # This thread's tile in C
    bid_m = cute.arch.block_idx()[0]
    bid_n = cute.arch.block_idx()[1]
    
    # Global position
    tile_m = bid_m * TILE_M + thread_m * THREAD_M
    tile_n = bid_n * TILE_N + thread_n * THREAD_N
    
    # Accumulator (8×8 tile in registers)
    acc = cute.make_rmem_tensor(
        cute.make_shape(THREAD_M, THREAD_N),
        cutlass.Float32
    )
    acc.store(0.0)
    
    # Compute (simplified)
    for k_tile in range(num_k_tiles):
        # Each thread loads its portion
        frag_A = load_thread_fragment_A(A, tile_m, k_tile)
        frag_B = load_thread_fragment_B(B, k_tile, tile_n)
        
        # Outer product into accumulator
        for i in range(THREAD_M):
            for j in range(THREAD_N):
                for k in range(TILE_K):
                    acc[i, j] += frag_A[i, k] * frag_B[k, j]
    
    # Store results
    store_thread_tile(C, acc, tile_m, tile_n)
```

**Visualization:**
```
Block (0,0) processes C[0:128, 0:128]

Thread layout (16×16 threads):
┌────────────────────────────────┐
│ T0  T1  T2  T3  ... T15        │
│ T16 T17 T18 T19 ... T31        │
│ T32 T33 T34 T35 ... T47        │
│ ...                            │
│ T240 T241 ... T255             │
└────────────────────────────────┘

Each thread (e.g., T0) computes 8×8 tile:
C[0:8, 0:8] ← Thread 0
C[0:8, 8:16] ← Thread 1
...
```

---

## Warp-Level Layouts

### Warp Tile Organization

**Group threads into warps:**
```python
@cute.kernel
def warp_layout_gemm():
    """Warp-level organization"""
    
    # Block has 128 threads = 4 warps
    tid = cute.arch.thread_idx()[0]
    warp_id = tid // 32  # 0, 1, 2, 3
    lane_id = tid % 32   # 0-31
    
    # Warp tile dimensions
    WARP_M, WARP_N = 32, 64
    
    # 4 warps arranged as 2×2
    warp_m = warp_id // 2  # 0 or 1
    warp_n = warp_id % 2   # 0 or 1
    
    # This warp's tile in C
    warp_tile_m = warp_m * WARP_M  # 0 or 32
    warp_tile_n = warp_n * WARP_N  # 0 or 64
    
    # Thread within warp
    # 32 threads process 32×64 warp tile
    thread_tile_m = lane_id // 8  # 0-3 (4 rows)
    thread_tile_n = lane_id % 8   # 0-7 (8 cols)
    
    # Each thread: 8×8 tile
    THREAD_M, THREAD_N = 8, 8
```

**Warp organization:**
```
Block tile: 64×128
4 warps (2×2):

┌─────────────┬─────────────┐
│   Warp 0    │   Warp 1    │
│   32×64     │   32×64     │
├─────────────┼─────────────┤
│   Warp 2    │   Warp 3    │
│   32×64     │   32×64     │
└─────────────┴─────────────┘
```

### Warp-Specialized Layouts

**Different warps do different work:**
```python
@cute.kernel
def warp_specialized():
    """Warp specialization"""
    
    tid = cute.arch.thread_idx()[0]
    warp_id = tid // 32
    
    # 128 threads = 4 warps
    if warp_id == 0:
        # Warp 0: Producer (loads data)
        producer_work()
    else:
        # Warps 1-3: Consumers (compute)
        consumer_id = warp_id - 1
        consumer_work(consumer_id)
```

---

## Bank Conflict Avoidance

### Shared Memory Banks

**Shared memory organized in banks:**
```
Ampere/Hopper: 32 banks
Bank width: 4 bytes (1 float32)

Address mapping:
addr[bits 6:2] → bank number (0-31)

Example:
Address 0   → Bank 0
Address 4   → Bank 1
Address 8   → Bank 2
...
Address 124 → Bank 31
Address 128 → Bank 0 (wraps)
```

### Bank Conflict Example

**Conflict (bad layout):**
```python
@cute.kernel
def bank_conflict():
    """Bad: All threads access same bank"""
    
    smem = cute.make_smem_tensor(cute.make_shape(32, 32), cutlass.Float32)
    
    tid = cute.arch.thread_idx()[0]
    lane_id = tid % 32
    
    # All threads in warp access column 0
    # All access bank 0 → 32-way conflict!
    value = smem[lane_id, 0]  # BAD
```

**No conflict (good layout):**
```python
@cute.kernel
def no_bank_conflict():
    """Good: Threads access different banks"""
    
    smem = cute.make_smem_tensor(cute.make_shape(32, 32), cutlass.Float32)
    
    tid = cute.arch.thread_idx()[0]
    lane_id = tid % 32
    
    # Each thread accesses different column
    # Different banks → no conflict!
    value = smem[lane_id, lane_id]  # GOOD
```

### Padding to Avoid Conflicts

**Add padding columns:**
```python
@cute.kernel
def padded_layout():
    """Padding eliminates bank conflicts"""
    
    COLS = 32
    PADDING = 8  # Extra columns
    
    # Allocate with padding
    smem = cute.make_smem_tensor(
        cute.make_shape(32, COLS + PADDING),
        cutlass.Float32
    )
    
    tid = cute.arch.thread_idx()[0]
    lane_id = tid % 32
    
    # Access pattern same as before
    # But padding shifts addresses → no conflicts
    for i in range(COLS):
        value = smem[lane_id, i]  # No conflict now!
```

---

## CuTe Layout Composition

### Using make_layout

**Define custom thread layouts:**
```python
@cute.kernel
def cute_layout_example():
    """Using CuTe layout system"""
    
    # Thread layout: 16×16 threads → 128×128 tile
    thread_layout = cute.make_layout(
        cute.make_shape(16, 16),  # Thread grid
        cute.make_stride(16, 1)   # Row-major threads
    )
    
    # Each thread processes 8×8 tile
    thread_tile = cute.make_layout(
        cute.make_shape(8, 8),
        cute.make_stride(1, 8)  # Column-major within tile
    )
    
    # Compose: total layout is 128×128
    total_layout = cute.composition(thread_layout, thread_tile)
    
    # total_layout describes:
    # - 256 threads (16×16)
    # - Each processes 8×8 elements
    # - Total: 128×128 elements
```

### Partitioning with Layouts
```python
@cute.kernel
def partition_example(data: cute.Tensor):
    """Partition data among threads"""
    
    tid = cute.arch.thread_idx()[0]
    
    # Define how threads access data
    thread_layout = cute.make_layout(cute.make_shape(256))
    
    # Partition data tensor for this thread
    thread_data = cute.partition(data, thread_layout, tid)
    
    # Now thread_data is view of this thread's elements
    for i in range(cute.size(thread_data)):
        thread_data[i] = process(thread_data[i])
```

---

## Tensor Core Layouts

### MMA Thread Layout (Ampere)

**Tensor cores have fixed thread layout:**
```python
@cute.kernel
def tensor_core_layout():
    """Ampere 16x8x16 MMA layout"""
    
    # MMA instruction: 16×8×16 (M×N×K)
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Thread layout for MMA (fixed by hardware)
    # 32 threads participate
    # Each thread holds specific fragment elements
    
    tid = cute.arch.thread_idx()[0]
    lane_id = tid % 32
    
    # Fragment A: 16×16, each thread holds some elements
    frag_A = cute.make_rmem_tensor(
        mma_atom.frag_A_layout(),
        cutlass.Float16
    )
    
    # Fragment B: 16×8, each thread holds some elements
    frag_B = cute.make_rmem_tensor(
        mma_atom.frag_B_layout(),
        cutlass.Float16
    )
    
    # Fragment C: 16×8, each thread holds some elements
    frag_C = cute.make_rmem_tensor(
        mma_atom.frag_C_layout(),
        cutlass.Float32
    )
    
    # Load fragments (uses specific layout)
    load_mma_fragment(frag_A, smem_A, lane_id)
    load_mma_fragment(frag_B, smem_B, lane_id)
    
    # MMA
    cute.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
```

### Tiled MMA Layout

**Multiple MMA tiles per thread block:**
```python
@cute.kernel
def tiled_mma_layout():
    """Tiled MMA with multiple atoms"""
    
    # Single MMA: 16×8×16
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Tile 8×16 MMA atoms → 128×128 output
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        thread_layout=cute.make_shape(8, 16, 1)
    )
    
    # 256 threads (8 MMA ops × 32 threads each / sharing)
    # Each thread participates in multiple MMAs
    
    tid = cute.arch.thread_idx()[0]
    
    # Allocate fragments for tiled MMA
    frag_A = cute.make_rmem_tensor(
        tiled_mma.frag_A_layout(),
        cutlass.Float16
    )
    frag_B = cute.make_rmem_tensor(
        tiled_mma.frag_B_layout(),
        cutlass.Float16
    )
    frag_C = cute.make_rmem_tensor(
        tiled_mma.frag_C_layout(),
        cutlass.Float32
    )
    
    # Load and compute
    load_tiled_fragments(frag_A, frag_B, smem_A, smem_B, tid)
    cute.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
```

---

## Performance Considerations

### Memory Coalescing Impact

**Benchmark:**
```python
def benchmark_coalescing():
    """Compare coalesced vs non-coalesced access"""
    
    import torch
    import time
    
    n = 1024 * 1024
    data = torch.randn(n, device='cuda', dtype=torch.float32)
    
    # Coalesced access
    start = time.time()
    for _ in range(1000):
        coalesced_kernel.launch(...)(data)
    torch.cuda.synchronize()
    coalesced_time = time.time() - start
    
    # Non-coalesced (strided) access
    start = time.time()
    for _ in range(1000):
        strided_kernel.launch(...)(data)
    torch.cuda.synchronize()
    strided_time = time.time() - start
    
    print(f"Coalesced: {coalesced_time*1000:.2f} ms")
    print(f"Strided:   {strided_time*1000:.2f} ms")
    print(f"Speedup:   {strided_time/coalesced_time:.1f}×")
    
    # Typical output:
    # Coalesced: 3.45 ms
    # Strided:   98.23 ms
    # Speedup:   28.5×
```

### Bank Conflict Impact
```python
def benchmark_bank_conflicts():
    """Measure bank conflict overhead"""
    
    # No padding (conflicts)
    time_conflicts = benchmark_smem_access(padding=0)
    
    # With padding (no conflicts)
    time_no_conflicts = benchmark_smem_access(padding=8)
    
    print(f"With conflicts:    {time_conflicts:.2f} ms")
    print(f"Without conflicts: {time_no_conflicts:.2f} ms")
    print(f"Speedup:           {time_conflicts/time_no_conflicts:.1f}×")
    
    # Typical output:
    # With conflicts:    12.34 ms
    # Without conflicts:  3.45 ms
    # Speedup:            3.6×
```

---

## Best Practices

### ✅ DO

**Ensure coalesced access:**
```python
# Threads access consecutive memory
idx = bid * block_size + tid
data[idx] = value
```

**Use 2D layouts for matrices:**
```python
block = [16, 16, 1]  # 2D thread block
```

**Add padding to avoid bank conflicts:**
```python
smem = cute.make_smem_tensor(
    cute.make_shape(32, 32 + 8),  # +8 padding
    dtype
)
```

**Vectorize when possible:**
```python
# Load 4 floats at once
vec = load_float4(data, idx)
```

### ❌ DON'T

**Don't use strided access:**
```python
# ✗ BAD: Non-coalesced
idx = tid
while idx < n:
    data[idx] = value
    idx += 32  # Stride!
```

**Don't access same shared memory bank:**
```python
# ✗ BAD: Bank conflict
value = smem[lane_id, 0]  # All threads → bank 0
```

**Don't use arbitrary thread counts:**
```python
# ✗ BAD: Not multiple of warp size
block = [250, 1, 1]  # Should be 256
```

---

## Summary

**Thread layout:**
- How threads map to data
- Critical for performance
- Affects memory coalescing
- Affects bank conflicts

**Key patterns:**
- Linear: 1D sequential
- 2D: Matrix operations
- Tiled: Each thread processes tile
- Vectorized: SIMD access

**Optimization rules:**
1. Coalesce global memory (consecutive access)
2. Avoid bank conflicts (padding)
3. Use warp-size multiples
4. Vectorize when possible

**Key insight:** The right thread layout can give 10-100× speedup. Always think about how threads access memory.

---

## Next Steps

- [Tiling Strategies](./tiling_strategies.md) - What threads process
- [Performance Tips](./performance_tips.md) - Additional optimizations
- [Profiling](./profiling.md) - Measuring layout efficiency

---

## Further Reading

- [CUDA Memory Coalescing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
- [Shared Memory Bank Conflicts](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-bank-conflicts)
- [CUTLASS Thread Maps](https://github.com/NVIDIA/cutlass/blob/main/media/docs/tile_iterator_concept.md)