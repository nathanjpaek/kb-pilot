---
topic: "swizzling"
difficulty: "advanced"
related_topics: ["memory_hierarchy", "layouts", "bank_conflicts"]
---

# Memory Swizzling in CuTe DSL

## Overview

**Swizzling** is a technique to rearrange how data is stored in shared memory to avoid **bank conflicts**. Bank conflicts occur when multiple threads in a warp access different addresses in the same memory bank, causing serialization and performance loss.

**Key concept:** By "swizzling" (permuting) the address mapping, we can distribute accesses across banks more evenly, eliminating conflicts.

---

## The Bank Conflict Problem

### Shared Memory Banks

**Shared memory is divided into 32 banks:**
```
Bank 0:  Address 0, 32, 64, 96, 128, ...
Bank 1:  Address 1, 33, 65, 97, 129, ...
Bank 2:  Address 2, 34, 66, 98, 130, ...
...
Bank 31: Address 31, 63, 95, 127, 159, ...
```

**Rule:** Address `addr` is in bank `(addr / 4) % 32` (for 4-byte elements like Float32).

### When Conflicts Occur

**Conflict example:**
```
@cute.kernel
def bank_conflict_example():
    smem = cute.make_smem_tensor(cute.make_shape(32, 32), cutlass.Float32)
    tid = cute.arch.thread_idx()[0]  # 0-31
    
    # ✗ BAD: All threads access column 0
    val = smem[tid, 0]
    # Thread 0: smem[0, 0]  -> Bank 0
    # Thread 1: smem[1, 0]  -> Bank 0
    # Thread 2: smem[2, 0]  -> Bank 0
    # ...
    # All threads access Bank 0! 32-way conflict!
```

**What happens:**
- All 32 threads try to access Bank 0 simultaneously
- Hardware serializes the accesses: 1st thread, wait, 2nd thread, wait, ...
- 32× slower than conflict-free access!

### Conflict-Free Access

**No conflict example:**
```
@cute.kernel
def no_conflict_example():
    smem = cute.make_smem_tensor(cute.make_shape(32, 32), cutlass.Float32)
    tid = cute.arch.thread_idx()[0]  # 0-31
    
    # ✓ GOOD: Adjacent threads access adjacent columns
    val = smem[0, tid]
    # Thread 0: smem[0, 0]  -> Bank 0
    # Thread 1: smem[0, 1]  -> Bank 1
    # Thread 2: smem[0, 2]  -> Bank 2
    # ...
    # Each thread accesses different bank - no conflict!
```

---

## Simple Solution: Padding

### Adding Padding to Avoid Conflicts

**Problem:** Matrix with stride equal to power of 2 causes conflicts.
```
@cute.kernel
def conflict_with_padding():
    # ✗ BAD: 32 columns = stride 32 = bank conflict pattern
    smem = cute.make_smem_tensor(cute.make_shape(128, 32), cutlass.Float32)
    
    # Accessing column causes conflict
    # smem[0, 0] -> Bank 0
    # smem[1, 0] -> Bank (32/4) % 32 = Bank 8
    # smem[2, 0] -> Bank 16
    # smem[3, 0] -> Bank 24
    # smem[4, 0] -> Bank 0  <- Conflict with smem[0,0]!
```

**Solution:** Add one extra column (padding).
```
@cute.kernel
def no_conflict_with_padding():
    # ✓ GOOD: 33 columns = stride 33 = no conflict pattern
    smem = cute.make_smem_tensor(cute.make_shape(128, 33), cutlass.Float32)
    
    # Use only first 32 columns, ignore column 32 (padding)
    # Now stride is 33, which doesn't align with bank pattern
    
    # Accessing column:
    # smem[0, 0] -> Bank 0
    # smem[1, 0] -> Bank (33/4) % 32 = Bank 8
    # smem[2, 0] -> Bank 16
    # smem[3, 0] -> Bank 25
    # smem[4, 0] -> Bank 1  <- Different bank!
```

**Key insight:** Stride of 33 is coprime with 32, so conflicts are avoided.

### Padding Pattern
```
# General rule: Add padding to avoid power-of-2 strides
original_columns = 32  # or 64, 128, etc.
padded_columns = original_columns + 1  # or +8, +16 depending on layout

smem = cute.make_smem_tensor(
    cute.make_shape(rows, padded_columns),
    cutlass.Float32
)

# Use only original_columns for actual data
# Extra column(s) are wasted but avoid conflicts
```

---

## Advanced Solution: Swizzling

### What is Swizzling?

**Swizzling** permutes address bits to distribute accesses across banks without wasting memory.

**Concept:**
```
Normal addressing:    [row bits | col bits]
Swizzled addressing:  [row bits XOR col bits | col bits]
```

The XOR operation "mixes" row and column bits, spreading accesses across banks.

### Swizzling in CuTe

**CuTe provides swizzle functions (exact API may vary - check documentation):**
```
@cute.kernel
def swizzled_layout_example():
    # Create base layout
    base_shape = cute.make_shape(128, 64)
    base_stride = cute.make_stride(1, 128)
    
    # Apply swizzle (API example - actual syntax may differ)
    # This is conceptual - CuTe Python may have different API
    swizzled_layout = apply_swizzle(base_shape, base_stride, swizzle_bits=3)
    
    smem = cute.make_smem_tensor_with_layout(swizzled_layout, cutlass.Float32)
    
    # Access pattern now automatically distributes across banks
```

**Note:** Swizzling API in CuTe Python may differ from C++. Check current documentation for exact syntax.

---

## Swizzle Patterns

### XOR-Based Swizzling

**Common pattern:** XOR row index with column index.
```
# Conceptual swizzle function
def swizzled_address(row, col, swizzle_bits):
    # XOR high bits of row with low bits of col
    swizzled_col = col ^ (row >> swizzle_bits)
    return row * stride + swizzled_col
```

**Effect:**
- Adjacent rows access different banks for same column
- Eliminates conflicts for common access patterns

### Swizzle Bit Selection

**Swizzle parameter determines mixing:**
```
swizzle_bits = 3:  Mix 3 bits (good for 8-element groups)
swizzle_bits = 4:  Mix 4 bits (good for 16-element groups)
swizzle_bits = 5:  Mix 5 bits (good for 32-element groups)
```

**Rule of thumb:** Match swizzle_bits to your access pattern granularity.

---

## Practical Swizzling Examples

### Example 1: Tile for GEMM
```
@cute.kernel
def gemm_with_swizzled_smem(gA, gB, gC):
    # Allocate swizzled shared memory tiles
    # (Exact API may vary - this is conceptual)
    
    tile_shape = cute.make_shape(128, 64)
    
    # Create swizzled layout for tile A
    # This avoids bank conflicts when loading columns
    smem_A = allocate_swizzled_smem(tile_shape, cutlass.Float16)
    
    # Create swizzled layout for tile B  
    smem_B = allocate_swizzled_smem(tile_shape, cutlass.Float16)
    
    # Load from global to swizzled shared memory
    cute.copy(tma_A, gA[block_tile], smem_A)
    cute.copy(tma_B, gB[block_tile], smem_B)
    
    cute.arch.syncthreads()
    
    # Access shared memory - swizzling handles bank conflicts automatically
    cute.gemm(tiled_mma, acc, smem_A, smem_B, acc)
```

### Example 2: Transpose with Swizzling
```
@cute.kernel
def transpose_swizzled(input: cute.Tensor, output: cute.Tensor):
    tile_size = 32
    
    # Swizzled shared memory avoids conflicts during transpose
    smem = allocate_swizzled_smem(
        cute.make_shape(tile_size, tile_size),
        cutlass.Float32
    )
    
    # Load tile (coalesced)
    tid = cute.arch.thread_idx()[0]
    for i in range(tile_size):
        smem[tid, i] = input[block_offset + tid * stride + i]
    
    cute.arch.syncthreads()
    
    # Store transposed (would conflict without swizzling!)
    for i in range(tile_size):
        output[block_offset_transposed + tid * stride + i] = smem[i, tid]
```

---

## When to Use Swizzling vs Padding

### Use Padding When:

✅ **Simple layouts:**
```
# Easy to add one column
smem = cute.make_smem_tensor(cute.make_shape(128, 33), cutlass.Float32)
```

✅ **Small memory overhead acceptable:**
```
# 32 -> 33 columns = 3% overhead (minimal)
```

✅ **Access patterns are regular:**
```
# Column accesses only, or row accesses only
```

### Use Swizzling When:

✅ **Complex access patterns:**
```
# Both row and column accesses
# Diagonal accesses
# Transpose patterns
```

✅ **Memory is tight:**
```
# Can't afford padding overhead
# Need exact tile size
```

✅ **Maximum performance needed:**
```
# Swizzling can be more efficient than padding
# No wasted memory
```

---

## Detecting Bank Conflicts

### Using NVIDIA Profiler
```bash
# Profile with nsys
nsys profile --stats=true python my_kernel.py

# Look for shared memory metrics in output
# High "shared_ld/st_bank_conflict" indicates conflicts
```

### Using Nsight Compute
```bash
# Profile with ncu
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    python my_kernel.py

# Check metric value:
# 0 = no conflicts (good!)
# >0 = conflicts detected
```

### Manual Inspection
```
@cute.kernel
def debug_bank_conflicts():
    smem = cute.make_smem_tensor(cute.make_shape(32, 32), cutlass.Float32)
    tid = cute.arch.thread_idx()[0]
    
    # Print bank for each thread
    bank = ((tid * 32 * 4) / 4) % 32  # Rough calculation
    cute.printf("Thread %d accesses bank %d\n", tid, bank)
    
    # If many threads print same bank = conflict!
```

---

## Architecture-Specific Considerations

### All Architectures (Volta+)

- 32 banks
- 4-byte bank width (Float32)
- 32-way bank conflicts possible

### Hopper (SM90) Specifics

- TMA (Tensor Memory Accelerator) can handle swizzling automatically
- May not need manual swizzling with TMA
- Check TMA documentation for built-in swizzle support

### Best Practice by Architecture

**Ampere (SM80):**
```
# Manual swizzling or padding often needed
# Especially for MMA operations
smem = allocate_with_padding(shape)
```

**Hopper (SM90+):**
```
# TMA may handle swizzling
# Check if manual swizzling still beneficial
# Profile to verify
```

---

## Common Patterns and Best Practices

### Pattern 1: GEMM Shared Memory
```
@cute.kernel
def gemm_smem_pattern():
    # Tiles for A and B matrices
    tile_m, tile_n, tile_k = 128, 128, 32
    
    # Use padding to avoid conflicts
    smem_A = cute.make_smem_tensor(
        cute.make_shape(tile_m, tile_k + 8),  # +8 padding
        cutlass.Float16
    )
    smem_B = cute.make_smem_tensor(
        cute.make_shape(tile_k, tile_n + 8),  # +8 padding
        cutlass.Float16
    )
    
    # Or use swizzled layouts (if supported)
    # smem_A = allocate_swizzled_smem(...)
```

### Pattern 2: Reduction
```
@cute.kernel
def reduction_smem_pattern():
    # Reduction buffer - add padding to avoid conflicts
    smem = cute.make_smem_tensor(
        cute.make_shape(256 + 8),  # 256 threads + 8 padding
        cutlass.Float32
    )
    
    tid = cute.arch.thread_idx()[0]
    
    # Load (no conflict - sequential)
    smem[tid] = input[tid]
    cute.arch.syncthreads()
    
    # Reduce (stride-based access - padding helps)
    for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        cute.arch.syncthreads()
```

### Pattern 3: Transpose
```
@cute.kernel
def transpose_pattern():
    tile_size = 32
    
    # Padded shared memory for transpose
    smem = cute.make_smem_tensor(
        cute.make_shape(tile_size, tile_size + 1),  # +1 padding
        cutlass.Float32
    )
    
    tid = cute.arch.thread_idx()[0]
    
    # Load (coalesced, no conflicts)
    for i in range(tile_size):
        smem[tid, i] = input[block_start + tid * N + i]
    
    cute.arch.syncthreads()
    
    # Transpose store (padding prevents conflicts)
    for i in range(tile_size):
        output[block_start_T + tid * M + i] = smem[i, tid]
```

---

## Performance Impact

### With Bank Conflicts
```
Effective bandwidth = Theoretical bandwidth / num_conflicts

Example:
- Theoretical: 10 TB/s
- 4-way conflict: 10 TB/s / 4 = 2.5 TB/s
- 32-way conflict: 10 TB/s / 32 = 312 GB/s (!)
```

### Without Bank Conflicts
```
Effective bandwidth ≈ Theoretical bandwidth

Example:
- Padding overhead: 32 -> 33 = 3% memory waste
- Performance gain: 4× or more if conflicts eliminated
- Net win: massive
```

### Typical Speedups

| Scenario | Without Swizzle/Padding | With Swizzle/Padding | Speedup |
|----------|-------------------------|----------------------|---------|
| Transpose | 100 GB/s | 800 GB/s | 8× |
| GEMM smem loads | 500 GB/s | 2 TB/s | 4× |
| Reduction | 300 GB/s | 1.2 TB/s | 4× |

---

## Debugging Swizzling Issues

### Common Issues

**Issue 1: Still seeing conflicts after padding**
```
# Check padding is actually applied
print(f"Shape: {smem.shape}")  # Should see padded dimension

# Verify stride
print(f"Stride: {smem.stride}")  # Should be padded value
```

**Issue 2: Incorrect padding amount**
```
# ✗ WRONG: Padding not coprime with 32
smem = cute.make_smem_tensor(cute.make_shape(32, 64), ...)  # 64 = 2*32, still conflicts!

# ✓ CORRECT: Padding coprime with 32
smem = cute.make_smem_tensor(cute.make_shape(32, 65), ...)  # 65 not divisible by 32
```

**Issue 3: Accessing padded elements**
```
# ✗ WRONG: Accessing padding region
for j in range(tile_n + 1):  # Includes padding!
    val = smem[i, j]

# ✓ CORRECT: Only access valid data
for j in range(tile_n):  # Excludes padding
    val = smem[i, j]
```

---

## Best Practices Summary

### ✅ DO

**Profile first:**
```
# Use nsys or ncu to detect conflicts before optimizing
nsys profile --stats=true python kernel.py
```

**Use padding for simple cases:**
```
# +1 or +8 padding often sufficient
smem = cute.make_smem_tensor(cute.make_shape(M, N + 1), dtype)
```

**Consider swizzling for complex patterns:**
```
# When both row and column accesses needed
# When memory overhead matters
```

**Verify no wasted performance:**
```
# Check achieved bandwidth vs theoretical
# Ensure conflicts are eliminated
```

### ❌ DON'T

**Don't ignore bank conflicts:**
```
# Even "small" conflicts (2-4 way) hurt performance
# Always profile shared memory kernels
```

**Don't over-pad:**
```
# ✗ WRONG: Excessive padding
smem = cute.make_smem_tensor(cute.make_shape(32, 64), ...)  # +32 padding = 100% overhead!

# ✓ CORRECT: Minimal padding
smem = cute.make_smem_tensor(cute.make_shape(32, 33), ...)  # +1 padding = 3% overhead
```

**Don't assume swizzling is always available:**
```
# Check CuTe Python documentation for current swizzle API
# May need to use padding if swizzle not exposed
```

---

## Summary

**Bank conflicts occur when:**
- Multiple threads in warp access different addresses in same bank
- Common with stride-of-32 patterns in shared memory

**Solutions:**
1. **Padding:** Add extra columns (e.g., 32 → 33)
   - Simple, effective for most cases
   - Small memory overhead (3-25%)
   
2. **Swizzling:** XOR address bits
   - No memory overhead
   - More complex, better for complex access patterns
   - May require architecture-specific code

**When it matters:**
- Shared memory intensive kernels (GEMM, transpose, reduction)
- Can improve performance 2-8×
- Critical for achieving peak shared memory bandwidth

**Best practice:**
- Profile to detect conflicts
- Start with padding (simple, effective)
- Consider swizzling if padding insufficient or memory tight
- Always verify improvement with profiler

---

## Next Steps

- [Memory Hierarchy](./memory_hierarchy.md) - Understanding shared memory
- [Copy Operations](../04_operations/copy_atoms.md) - Efficient data movement
- [GEMM Examples](../10_examples/) - Real-world swizzling usage

---

## Further Reading

- [CUDA Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Bank Conflicts Explained](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [CuTe C++ Swizzle Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/03_swizzle.md)