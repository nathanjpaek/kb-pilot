---
topic: "memory_hierarchy"
difficulty: "intermediate"
related_topics: ["tensors", "layouts", "copy_atoms"]
---

# GPU Memory Hierarchy in CuTe DSL

## Overview

Understanding GPU memory hierarchy is critical for writing high-performance CuTe DSL kernels. Different memory spaces have vastly different capacities, latencies, and access patterns.

**Key principle:** Move data through the memory hierarchy efficiently - load from slow memory once, reuse from fast memory many times.

---

## GPU Memory Spaces

### Memory Hierarchy Pyramid
```
                    [Registers (RMEM)]
                    Fastest, Smallest
                    ~256KB per SM
                    Private per thread
                         |
                    [Shared Memory (SMEM)]
                    Very Fast, Small
                    48KB - 256KB per block
                    Shared within thread block
                         |
                    [L2 Cache]
                    Fast, Medium
                    40MB - 60MB (varies by GPU)
                    Shared across SMs
                         |
                    [Global Memory (GMEM)]
                    Slow, Largest
                    16GB - 80GB+ (varies by GPU)
                    Shared across entire GPU
                         |
                    [Host Memory (CPU)]
                    Very Slow, Very Large
                    Access via PCIe
```

---

## Global Memory (GMEM)

### Characteristics

- **Size:** 16GB - 80GB+ (largest memory space)
- **Latency:** ~400-800 cycles
- **Bandwidth:** 1-3 TB/s (varies by GPU)
- **Scope:** Visible to all threads, all blocks, all SMs
- **Lifetime:** Persistent across kernel launches

### In CuTe DSL

**Creating global memory tensors:**
```
import torch
from cutlass.cute.runtime import from_dlpack

# PyTorch tensor in global memory
torch_tensor = torch.randn(1024, 1024, device="cuda")

# Convert to CuTe tensor (zero-copy, still in GMEM)
gmem_tensor = from_dlpack(torch_tensor)

# Or allocate within @cute.jit
@cute.jit
def allocate_gmem():
    shape = cute.make_shape(1024, 1024)
    gmem_tensor = cute.make_tensor(shape, cutlass.Float32)
    return gmem_tensor
```

### Best Practices for GMEM

**✅ DO:**

**Coalesce memory accesses:**
```
@cute.kernel
def coalesced_access(data: cute.Tensor):
    # Adjacent threads access adjacent memory
    idx = cute.arch.block_idx()[0] * cute.arch.block_dim_x() + cute.arch.thread_idx()[0]
    val = data[idx]  # ✓ Coalesced - good!
```

**Use vectorized loads when possible:**
```
@cute.kernel
def vectorized_load(data: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    # Load 4 floats at once (if aligned)
    val = data[idx * 4 : idx * 4 + 4]
```

**Minimize GMEM accesses:**
```
@cute.kernel
def minimize_gmem():
    # ✓ Load once, reuse many times
    val = gmem_tensor[idx]
    result = val * val + val * 2.0 + val * 3.0
    gmem_output[idx] = result
    
    # ✗ Bad: Multiple loads
    gmem_output[idx] = gmem_tensor[idx] * gmem_tensor[idx] + gmem_tensor[idx] * 2.0
```

**❌ DON'T:**

**Strided access patterns:**
```
@cute.kernel
def bad_strided_access(data: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    # ✗ Stride of 1024 - poor coalescing
    val = data[idx * 1024]
```

**Unaligned accesses:**
```
@cute.kernel
def bad_unaligned(data: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    # ✗ Misaligned - poor performance
    val = data[idx * 3 + 1]
```

---

## Shared Memory (SMEM)

### Characteristics

- **Size:** 48KB (Volta), 164KB (Ampere), 256KB (Hopper+)
- **Latency:** ~20-30 cycles
- **Bandwidth:** ~10-20 TB/s (much faster than GMEM)
- **Scope:** Shared within thread block only
- **Lifetime:** Kernel execution only

### In CuTe DSL

**Allocating shared memory:**
```
@cute.kernel
def use_shared_memory(gmem_tensor: cute.Tensor):
    # Allocate shared memory for this thread block
    # Note: Actual allocation syntax may vary - check documentation
    smem_size = cute.make_shape(128, 64)
    smem_tensor = cute.make_smem_tensor(smem_size, cutlass.Float32)
    
    # Copy from global to shared
    cute.copy(copy_atom, gmem_tensor, smem_tensor)
    
    # Synchronize - ensure all threads see the data
    cute.arch.syncthreads()
    
    # Now all threads in block can access smem_tensor
    val = smem_tensor[cute.arch.thread_idx()[0], cute.arch.thread_idx()[1]]
```

**Dynamic shared memory allocation:**
```
@cute.jit
def launch_with_shared_memory(gmem_data: cute.Tensor):
    # Calculate shared memory needed
    tile_size = 128 * 64
    bytes_per_element = 4  # Float32
    smem_bytes = tile_size * bytes_per_element
    
    # Launch kernel with shared memory allocation
    my_kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[128, 1, 1],
        smem=smem_bytes  # Request shared memory
    )(gmem_data)
```

### Shared Memory Bank Conflicts

**What are bank conflicts?**

Shared memory is divided into 32 banks. When multiple threads in a warp access different addresses in the same bank simultaneously, accesses are serialized.

**Example of bank conflict:**
```
@cute.kernel
def bad_bank_conflicts():
    smem = cute.make_smem_tensor(cute.make_shape(32, 32), cutlass.Float32)
    tid = cute.arch.thread_idx()[0]
    
    # ✗ BAD: All threads access same bank (stride of 32)
    val = smem[tid, 0]  # tid=0,1,2,... all access column 0
    # 32-way bank conflict! Serialized execution
```

**Avoiding bank conflicts:**
```
@cute.kernel
def good_no_conflicts():
    # Add padding to avoid conflicts
    smem = cute.make_smem_tensor(cute.make_shape(32, 33), cutlass.Float32)  # 33, not 32!
    tid = cute.arch.thread_idx()[0]
    
    # ✓ GOOD: Padding eliminates bank conflicts
    val = smem[tid, 0]  # Different banks due to padding
```

**Or use different access pattern:**
```
@cute.kernel
def good_coalesced_smem():
    smem = cute.make_smem_tensor(cute.make_shape(32, 32), cutlass.Float32)
    tid = cute.arch.thread_idx()[0]
    
    # ✓ GOOD: Adjacent threads access adjacent elements (stride 1)
    val = smem[0, tid]  # tid=0,1,2,... access different banks
```

### Best Practices for SMEM

**✅ DO:**

**Use as staging buffer:**
```
@cute.kernel
def staging_pattern(gmem_in: cute.Tensor, gmem_out: cute.Tensor):
    smem = cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float32)
    
    # Load tile from GMEM to SMEM
    cute.copy(copy_atom, gmem_in[block_tile], smem)
    cute.arch.syncthreads()
    
    # Process from SMEM (fast, reusable)
    for i in range(10):
        result = process(smem)
    
    cute.arch.syncthreads()
    
    # Write back to GMEM
    cute.copy(copy_atom, smem, gmem_out[block_tile])
```

**Pad to avoid bank conflicts:**
```
# Add 1 extra column to avoid conflicts
smem = cute.make_smem_tensor(cute.make_shape(tile_m, tile_n + 1), dtype)
```

**Synchronize before and after SMEM access:**
```
# After writing to SMEM
cute.copy(copy_atom, gmem, smem)
cute.arch.syncthreads()  # ✓ Ensure writes complete

# Read and process
val = smem[...]

cute.arch.syncthreads()  # ✓ Ensure reads complete before next write
```

**❌ DON'T:**

**Forget synchronization:**
```
# ✗ WRONG: Race condition!
cute.copy(copy_atom, gmem, smem)
val = smem[...]  # May read stale data!
```

**Exceed shared memory limits:**
```
# ✗ WRONG: Too much shared memory
smem = cute.make_smem_tensor(cute.make_shape(1024, 1024), cutlass.Float32)
# 1024*1024*4 = 4MB > 256KB limit!
```

---

## Register Memory (RMEM)

### Characteristics

- **Size:** ~256KB per SM (limited per thread)
- **Latency:** 1 cycle (fastest!)
- **Bandwidth:** Highest (on-chip)
- **Scope:** Private per thread
- **Lifetime:** Thread execution only

### In CuTe DSL

**Register variables:**
```
@cute.kernel
def use_registers(gmem: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    
    # Simple variables live in registers
    val = gmem[idx]  # val in register
    result = val * 2.0 + 1.0  # Intermediate values in registers
    
    gmem[idx] = result
```

**Register fragments for MMA:**
```
@cute.kernel
def mma_with_registers(smem_A, smem_B, gmem_C):
    tiled_mma = cute.make_tiled_mma(...)
    
    # Allocate register fragments
    frag_A = tiled_mma.make_fragment_A(smem_A.shape)
    frag_B = tiled_mma.make_fragment_B(smem_B.shape)
    frag_C = tiled_mma.make_fragment_C(gmem_C.shape)
    
    # Initialize accumulator (in registers)
    frag_C.store(0.0)
    
    # Load from SMEM to registers
    cute.copy(copy_atom, smem_A, frag_A)
    cute.copy(copy_atom, smem_B, frag_B)
    
    # MMA in registers (very fast!)
    cute.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
    
    # Store from registers to GMEM
    cute.copy(copy_atom, frag_C, gmem_C)
```

**Explicit register tensors:**
```
@cute.kernel
def explicit_register_tensor():
    # Create register-resident tensor
    reg_tensor = cute.make_rmem_tensor(cute.make_shape(8, 8), cutlass.Float32)
    
    # Use for thread-private computation
    for i in range(8):
        for j in range(8):
            reg_tensor[i, j] = compute(i, j)
```

### Register Pressure

**Too many registers per thread reduces occupancy:**
```
# ✗ BAD: Too many registers
@cute.kernel
def high_register_pressure():
    # 100 Float32 values = 400 registers!
    temp1 = cute.make_rmem_tensor(cute.make_shape(100), cutlass.Float32)
    temp2 = cute.make_rmem_tensor(cute.make_shape(100), cutlass.Float32)
    # ... may reduce occupancy
```

**Solution: Use shared memory for large temporary data:**
```
# ✓ GOOD: Keep large data in SMEM
@cute.kernel
def balanced_memory():
    # Small frequently-used data in registers
    accumulator = 0.0
    
    # Large temporary data in shared memory
    smem_temp = cute.make_smem_tensor(cute.make_shape(100), cutlass.Float32)
```

---

## L2 Cache

### Characteristics

- **Size:** 40MB - 60MB (varies by GPU)
- **Latency:** ~200 cycles
- **Scope:** Shared across all SMs
- **Management:** Automatic (hardware-managed)

### In CuTe DSL

**L2 cache is transparent** - you don't explicitly allocate or manage it.

**Tips for L2 efficiency:**

**Temporal locality:**
```
@cute.kernel
def temporal_locality(data: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    
    # ✓ Access same data multiple times (stays in L2)
    val = data[idx]
    result1 = process1(val)
    result2 = process2(val)
    result3 = process3(val)
```

**Spatial locality:**
```
@cute.kernel
def spatial_locality(data: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    
    # ✓ Access nearby elements (likely in same cache line)
    val1 = data[idx]
    val2 = data[idx + 1]
    val3 = data[idx + 2]
```

---

## Tensor Memory (TMEM) - Blackwell Only

### Characteristics

- **Size:** Architecture-specific (Blackwell)
- **Latency:** Very low
- **Purpose:** Specialized for tensor core operations
- **Scope:** SM-level

### In CuTe DSL

**TMEM usage (Blackwell-specific):**
```
@cute.kernel
def blackwell_tmem_example(gmem_data: cute.Tensor):
    # Allocate TMEM (if supported)
    # Exact API TBD - check Blackwell documentation
    
    # TMEM used for large MMA accumulators
    # Offloads register pressure
    pass
```

---

## Memory Access Patterns

### Pattern 1: GMEM → SMEM → Registers → SMEM → GMEM

**Standard pattern for compute-heavy kernels:**
```
@cute.kernel
def standard_pattern(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    # 1. Allocate shared memory
    smem_A = cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float32)
    smem_B = cute.make_smem_tensor(cute.make_shape(64, 128), cutlass.Float32)
    
    # 2. Load from GMEM to SMEM
    cute.copy(copy_atom, gA[block_tile], smem_A)
    cute.copy(copy_atom, gB[block_tile], smem_B)
    cute.arch.syncthreads()
    
    # 3. Load from SMEM to Registers
    frag_A = cute.make_rmem_tensor(cute.make_shape(16, 8), cutlass.Float32)
    frag_B = cute.make_rmem_tensor(cute.make_shape(8, 16), cutlass.Float32)
    cute.copy(copy_atom, smem_A[thread_tile], frag_A)
    cute.copy(copy_atom, smem_B[thread_tile], frag_B)
    
    # 4. Compute in Registers (fast!)
    result = compute(frag_A, frag_B)
    
    # 5. Store Registers to SMEM
    cute.copy(copy_atom, result, smem_A)
    cute.arch.syncthreads()
    
    # 6. Store SMEM to GMEM
    cute.copy(copy_atom, smem_A, gC[block_tile])
```

### Pattern 2: Direct GMEM Access (Simple Kernels)

**For element-wise operations:**
```
@cute.kernel
def elementwise_pattern(input: cute.Tensor, output: cute.Tensor):
    idx = cute.arch.block_idx()[0] * cute.arch.block_dim_x() + cute.arch.thread_idx()[0]
    
    # Direct GMEM access (no SMEM needed)
    if idx < cute.size(input):
        val = input[idx]
        output[idx] = val * 2.0 + 1.0
```

### Pattern 3: Reduction (GMEM → Registers → SMEM → GMEM)

**Block-level reduction:**
```
@cute.kernel
def reduction_pattern(input: cute.Tensor, output: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # 1. Load to registers and accumulate
    thread_sum = 0.0
    for i in range(elements_per_thread):
        thread_sum += input[tid * elements_per_thread + i]
    
    # 2. Store to SMEM for block reduction
    smem = cute.make_smem_tensor(cute.make_shape(256), cutlass.Float32)
    smem[tid] = thread_sum
    cute.arch.syncthreads()
    
    # 3. Reduce in SMEM (tree reduction)
    for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        cute.arch.syncthreads()
    
    # 4. Write result to GMEM
    if tid == 0:
        output[cute.arch.block_idx()[0]] = smem[0]
```

---

## Memory Bandwidth Utilization

### Theoretical Bandwidth

**Modern GPU memory bandwidth:**
- A100: 1.5 TB/s
- H100: 3.0 TB/s
- B100/B200: 8+ TB/s

### Measuring Achieved Bandwidth
```
@cute.jit
def measure_bandwidth(data: cute.Tensor, num_iterations: cutlass.Int32):
    import time
    
    # Warm up
    my_kernel.launch(...)(data)
    
    # Measure
    start = time.time()
    for i in range(num_iterations):
        my_kernel.launch(...)(data)
    cute.arch.device_synchronize()
    end = time.time()
    
    # Calculate bandwidth
    bytes_transferred = cute.size(data) * 4 * num_iterations  # Float32 = 4 bytes
    elapsed = end - start
    bandwidth_gbs = (bytes_transferred / elapsed) / 1e9
    
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
```

---

## Best Practices Summary

### Memory Movement Strategy

1. **Load from GMEM as infrequently as possible**
2. **Stage through SMEM for data reuse**
3. **Compute in registers for maximum speed**
4. **Synchronize at memory space boundaries**
5. **Coalesce GMEM accesses**
6. **Avoid SMEM bank conflicts**
7. **Manage register pressure**

### Optimization Checklist

**For GMEM:**
- ✓ Coalesced access patterns
- ✓ Vectorized loads/stores when aligned
- ✓ Minimize accesses (load once, use many times)

**For SMEM:**
- ✓ Proper synchronization (syncthreads)
- ✓ Padding to avoid bank conflicts
- ✓ Stay within size limits

**For Registers:**
- ✓ Keep frequently-used data in registers
- ✓ Monitor register pressure (affects occupancy)
- ✓ Use for accumulation and intermediate values

---

## Common Mistakes

**Mistake 1: Forgetting syncthreads**
```
# ✗ WRONG
cute.copy(copy_atom, gmem, smem)
val = smem[...]  # Race condition!

# ✓ CORRECT
cute.copy(copy_atom, gmem, smem)
cute.arch.syncthreads()
val = smem[...]
```

**Mistake 2: Uncoalesced GMEM access**
```
# ✗ WRONG: Strided access
idx = tid * 1024
val = gmem[idx]

# ✓ CORRECT: Coalesced access
idx = tid
val = gmem[idx]
```

**Mistake 3: Too much shared memory**
```
# ✗ WRONG: Exceeds limit
smem = cute.make_smem_tensor(cute.make_shape(2048, 2048), cutlass.Float32)
# 2048 * 2048 * 4 = 16MB > 256KB limit!

# ✓ CORRECT: Reasonable size
smem = cute.make_smem_tensor(cute.make_shape(128, 128), cutlass.Float32)
# 128 * 128 * 4 = 64KB < 256KB limit
```

---

## Summary

**Memory hierarchy (fastest to slowest):**
1. Registers (RMEM) - 1 cycle, private per thread
2. Shared Memory (SMEM) - ~20 cycles, shared per block
3. L2 Cache - ~200 cycles, automatic
4. Global Memory (GMEM) - ~400 cycles, shared across GPU
5. Host Memory - milliseconds, avoid!

**Key strategies:**
- Load from GMEM → stage in SMEM → compute in registers
- Coalesce GMEM accesses (adjacent threads → adjacent memory)
- Avoid SMEM bank conflicts (use padding)
- Synchronize when crossing memory boundaries
- Manage register pressure (affects occupancy)

---

## Next Steps

- [Copy Operations](../04_operations/copy_atoms.md) - Efficient data movement
- [Tiling Strategies](../08_optimization/tiling_strategies.md) - Using memory hierarchy effectively
- [Architecture Features](../05_architecture_specific/) - Memory features by GPU

---

## Further Reading

- [CUDA Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Memory Coalescing](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)