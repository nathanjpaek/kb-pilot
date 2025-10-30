---
topic: "copy_atoms"
difficulty: "intermediate"
related_topics: ["memory_hierarchy", "layouts", "tensors"]
---

# Copy Operations and Copy Atoms in CuTe DSL

## Overview

**Copy atoms** are the fundamental building blocks for data movement in CuTe DSL. They describe how to efficiently copy data between different memory spaces (global, shared, registers) using hardware-optimized instructions.

**Key concept:** Instead of manually writing loads/stores, use copy atoms that compile to optimal memory instructions (vectorized loads, async copy, TMA, etc.).

---

## What is a Copy Atom?

A **copy atom** encapsulates:
1. **Source memory space** (GMEM, SMEM, RMEM)
2. **Destination memory space** (GMEM, SMEM, RMEM)
3. **Data layout** (how threads cooperatively move data)
4. **Hardware instruction** (LDG, LDS, LDGSTS, TMA, etc.)

**Think of it as:** A template for how a group of threads should copy a chunk of data.

---

## Basic Copy Operation

### Simple Copy with cute.copy()
```
@cute.kernel
def basic_copy_example(src: cute.Tensor, dst: cute.Tensor):
    # Simple element-wise copy
    idx = cute.arch.thread_idx()[0]
    
    # Method 1: Direct assignment
    dst[idx] = src[idx]
    
    # Method 2: Using cute.copy (same result for simple case)
    cute.copy(src[idx], dst[idx])
```

**When to use direct assignment:**
- Simple element-wise operations
- No need for optimization
- Single element per thread

**When to use cute.copy with atoms:**
- Moving tiles of data
- Need vectorization
- Async copy (Ampere+)
- TMA (Hopper+)

---

## Copy Atoms for Different Memory Spaces

### GMEM → RMEM (Global to Registers)

**Standard load:**
```
@cute.kernel
def gmem_to_rmem(gmem_src: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Simple scalar load
    val = gmem_src[tid]  # Loads one Float32
    
    # Vectorized load (if aligned)
    # Loads 4 Float32 values at once
    vec_vals = gmem_src[tid * 4 : tid * 4 + 4]
```

**With copy atom (vectorized):**
```
@cute.kernel  
def vectorized_gmem_to_rmem(gmem_src: cute.Tensor):
    # Create copy atom for vectorized load
    # (Exact API may vary - check documentation)
    copy_atom = cute.make_copy_atom_vectorized(cutlass.Float32, vec_size=4)
    
    # Allocate register fragment
    rmem_dst = cute.make_rmem_tensor(cute.make_shape(4), cutlass.Float32)
    
    # Vectorized copy (4 elements in one instruction)
    cute.copy(copy_atom, gmem_src[tid * 4 : tid * 4 + 4], rmem_dst)
```

### GMEM → SMEM (Global to Shared)

**Without async copy:**
```
@cute.kernel
def gmem_to_smem_sync(gmem_src: cute.Tensor):
    smem_dst = cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float32)
    tid = cute.arch.thread_idx()[0]
    
    # Each thread copies its portion
    for i in range(64):
        smem_dst[tid, i] = gmem_src[tid * 64 + i]
    
    # Must sync before using shared memory
    cute.arch.syncthreads()
```

**With async copy (Ampere+):**
```
@cute.kernel
def gmem_to_smem_async(gmem_src: cute.Tensor):
    smem_dst = cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float32)
    
    # Create async copy atom
    async_copy_atom = cute.make_copy_atom_async(cutlass.Float32)
    
    # Issue async copy (doesn't block thread)
    cute.copy(async_copy_atom, gmem_src[tile_offset], smem_dst)
    
    # Do other work while copy happens...
    
    # Wait for async copy to complete
    cute.arch.cp_async_wait_all()
    cute.arch.syncthreads()
    
    # Now safe to use smem_dst
```

### SMEM → RMEM (Shared to Registers)

**Simple copy:**
```
@cute.kernel
def smem_to_rmem(smem_src: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Load from shared memory to register
    val = smem_src[tid, 0]
    
    # Or load multiple values
    rmem_fragment = cute.make_rmem_tensor(cute.make_shape(8), cutlass.Float32)
    for i in range(8):
        rmem_fragment[i] = smem_src[tid, i]
```

**With copy atom (optimized):**
```
@cute.kernel
def smem_to_rmem_optimized(smem_src: cute.Tensor):
    # Copy atom handles optimal instruction selection
    copy_atom = cute.make_copy_atom_smem(cutlass.Float32)
    
    rmem_dst = cute.make_rmem_tensor(cute.make_shape(8), cutlass.Float32)
    
    # Optimized copy (may use LDS.128, vector loads, etc.)
    cute.copy(copy_atom, smem_src[thread_portion], rmem_dst)
```

---

## Architecture-Specific Copy Instructions

### Ampere (SM80) - Async Copy

**cp.async instruction:**
```
@cute.kernel
def ampere_async_copy(gmem_src: cute.Tensor):
    smem_dst = cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float16)
    
    # Ampere async copy (cp.async)
    # Doesn't block thread - copy happens in background
    async_atom = cute.make_ampere_async_copy_atom(cutlass.Float16)
    
    cute.copy(async_atom, gmem_src[tile], smem_dst)
    
    # Thread can do other work here!
    compute_something_else()
    
    # Wait for copy
    cute.arch.cp_async_wait_all()
    cute.arch.syncthreads()
```

**Benefits:**
- Doesn't stall thread during memory transfer
- Enables software pipelining
- Better latency hiding

### Hopper (SM90) - TMA (Tensor Memory Accelerator)

**TMA bulk copy:**
```
@cute.kernel
def hopper_tma_copy(gmem_src: cute.Tensor):
    smem_dst = cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float16)
    
    # TMA descriptor (created once)
    tma_desc = cute.make_tma_descriptor(
        gmem_src.layout,
        smem_dst.layout,
        elem_type=cutlass.Float16
    )
    
    # TMA copy (hardware accelerated)
    # Copies entire tile with one instruction!
    cute.copy(tma_desc, gmem_src[tile], smem_dst)
    
    # TMA arrival/wait
    cute.arch.tma_arrive()
    cute.arch.tma_wait()
    cute.arch.syncthreads()
```

**Benefits:**
- Extremely fast (hardware accelerated)
- Copies entire tiles
- Minimal thread involvement
- Better for large tiles

### Blackwell (SM100) - Enhanced TMA

**2-CTA instructions and TMEM:**
```
@cute.kernel
def blackwell_enhanced_copy(gmem_src: cute.Tensor):
    # Blackwell can use TMEM for even faster operations
    # (Exact API TBD - check Blackwell documentation)
    
    # Enhanced TMA with 2-CTA coordination
    tma_2cta_desc = cute.make_tma_2cta_descriptor(...)
    
    # Copy to TMEM (Blackwell tensor memory)
    tmem_dst = allocate_tmem(...)
    cute.copy(tma_2cta_desc, gmem_src, tmem_dst)
```

---

## Tiled Copy: Cooperative Thread Groups

### TiledCopy Concept

**TiledCopy** organizes threads to cooperatively copy data.

**Example: 256 threads copy 128×64 tile**
```
@cute.kernel
def tiled_copy_example(gmem_src: cute.Tensor, smem_dst: cute.Tensor):
    # Create tiled copy atom
    # Describes how threads cooperate to copy a tile
    tiled_copy = cute.make_tiled_copy(
        copy_atom=cute.UniversalCopy(),  # Base copy operation
        thread_layout=cute.make_shape(256, 1),  # 256 threads
        value_layout=cute.make_shape(128, 64)   # Tile size
    )
    
    # Get this thread's portion
    thread_copy = tiled_copy.get_slice(cute.arch.thread_idx()[0])
    
    # Partition source and destination for this thread
    thread_src = thread_copy.partition_S(gmem_src)
    thread_dst = thread_copy.partition_D(smem_dst)
    
    # Each thread copies its assigned portion
    cute.copy(tiled_copy, thread_src, thread_dst)
    
    # Synchronize after cooperative copy
    cute.arch.syncthreads()
```

**What TiledCopy does:**
1. Divides tile among threads
2. Each thread copies its portion
3. Ensures efficient memory access patterns (coalescing, vectorization)

---

## Common Copy Patterns

### Pattern 1: Simple Tile Copy (GMEM → SMEM)
```
@cute.kernel
def simple_tile_copy(gmem_A: cute.Tensor, gmem_B: cute.Tensor):
    tile_size_m, tile_size_n = 128, 64
    
    # Allocate shared memory
    smem_A = cute.make_smem_tensor(cute.make_shape(tile_size_m, tile_size_n), cutlass.Float16)
    
    # Copy tile from global to shared
    tile_idx = cute.arch.block_idx()[0]
    cute.copy(gmem_A[tile_idx], smem_A)
    
    cute.arch.syncthreads()
    
    # Use smem_A...
```

### Pattern 2: Async Copy with Pipelining
```
@cute.kernel
def pipelined_copy(gmem_src: cute.Tensor, num_tiles: cutlass.Int32):
    # Allocate multiple shared memory buffers
    smem_buffers = [
        cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float16)
        for _ in range(3)  # 3 stages
    ]
    
    # Create async copy atom
    async_atom = cute.make_ampere_async_copy_atom(cutlass.Float16)
    
    # Pipelined loop
    for tile_idx in cutlass.range(num_tiles, prefetch_stages=2):
        stage = tile_idx % 3
        
        # Async copy next tile
        cute.copy(async_atom, gmem_src[tile_idx], smem_buffers[stage])
        
        # Wait for previous tile
        cute.arch.cp_async_wait_group(1)
        cute.arch.syncthreads()
        
        # Process current tile
        process(smem_buffers[(tile_idx - 1) % 3])
```

### Pattern 3: SMEM → RMEM for MMA
```
@cute.kernel
def smem_to_rmem_for_mma(smem_A: cute.Tensor, smem_B: cute.Tensor):
    # Create MMA atom
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F16F16_TN)
    
    # Create tiled MMA
    tiled_mma = cute.make_tiled_mma(mma_atom, ...)
    
    # Get this thread's MMA slice
    thread_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0])
    
    # Create register fragments
    frag_A = tiled_mma.make_fragment_A(smem_A.shape)
    frag_B = tiled_mma.make_fragment_B(smem_B.shape)
    
    # Partition shared memory for this thread
    thread_A = thread_mma.partition_A(smem_A)
    thread_B = thread_mma.partition_B(smem_B)
    
    # Copy SMEM → RMEM
    cute.copy(thread_A, frag_A)
    cute.copy(thread_B, frag_B)
    
    # Now frag_A and frag_B are in registers, ready for MMA
```

### Pattern 4: RMEM → GMEM (Store Results)
```
@cute.kernel
def rmem_to_gmem_store(gmem_dst: cute.Tensor):
    # Compute results in registers
    rmem_result = cute.make_rmem_tensor(cute.make_shape(8), cutlass.Float32)
    
    # ... compute values ...
    
    # Store back to global memory
    tid = cute.arch.thread_idx()[0]
    
    # Direct store
    for i in range(8):
        gmem_dst[tid * 8 + i] = rmem_result[i]
    
    # Or vectorized store (if aligned)
    copy_atom = cute.make_copy_atom_vectorized(cutlass.Float32, vec_size=4)
    cute.copy(copy_atom, rmem_result, gmem_dst[tid * 8 : tid * 8 + 8])
```

---

## Vectorization and Alignment

### Vectorized Loads

**Load multiple elements in one instruction:**
```
@cute.kernel
def vectorized_load_example(gmem_src: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Scalar load (4 instructions)
    a = gmem_src[tid * 4 + 0]
    b = gmem_src[tid * 4 + 1]
    c = gmem_src[tid * 4 + 2]
    d = gmem_src[tid * 4 + 3]
    
    # Vectorized load (1 instruction!)
    # Loads 128 bits (4 × Float32) at once
    vec_data = gmem_src[tid * 4 : tid * 4 + 4]
```

**Requirements for vectorization:**
- Address must be aligned (16-byte for 128-bit load)
- Contiguous elements
- Same data type

### Alignment
```
@cute.kernel
def alignment_example(gmem_src: cute.Tensor):
    # Ensure alignment for vectorization
    tid = cute.arch.thread_idx()[0]
    
    # ✓ GOOD: Aligned access (tid * 4 = multiples of 4)
    vec1 = gmem_src[tid * 4 : tid * 4 + 4]
    
    # ✗ BAD: Misaligned access (may not vectorize)
    vec2 = gmem_src[tid * 4 + 1 : tid * 4 + 5]
    
    # Use cute.assume to tell compiler about alignment
    aligned_ptr = cute.assume(gmem_src.data_ptr, align=16)
```

---

## Copy Performance Optimization

### Coalescing Global Memory

**Ensure adjacent threads access adjacent memory:**
```
@cute.kernel
def coalesced_copy(gmem_src: cute.Tensor, gmem_dst: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # ✓ GOOD: Coalesced (adjacent threads → adjacent memory)
    gmem_dst[tid] = gmem_src[tid]
    
    # ✗ BAD: Uncoalesced (strided access)
    gmem_dst[tid * 128] = gmem_src[tid * 128]
```

### Minimizing Bank Conflicts (SMEM)

**Use padding or swizzling:**
```
@cute.kernel
def smem_copy_no_conflicts():
    # Add padding to avoid bank conflicts
    smem = cute.make_smem_tensor(
        cute.make_shape(128, 64 + 1),  # +1 padding
        cutlass.Float32
    )
    
    tid = cute.arch.thread_idx()[0]
    
    # Copy without bank conflicts
    for i in range(64):
        smem[tid, i] = value
```

### Maximizing Bandwidth

**Tips:**
1. **Vectorize:** Use vector loads/stores when aligned
2. **Coalesce:** Adjacent threads → adjacent memory
3. **Async copy:** Hide latency with cp.async or TMA
4. **Pipeline:** Overlap copy with compute
5. **Avoid bank conflicts:** Pad shared memory layouts

---

## Error Handling and Debugging

### Common Copy Errors

**Error 1: Size mismatch**
```
# ✗ WRONG: Sizes don't match
src = cute.make_tensor(cute.make_shape(100), cutlass.Float32)
dst = cute.make_tensor(cute.make_shape(50), cutlass.Float32)
cute.copy(src, dst)  # ERROR: size mismatch!

# ✓ CORRECT: Sizes match
dst = cute.make_tensor(cute.make_shape(100), cutlass.Float32)
cute.copy(src, dst)
```

**Error 2: Type mismatch**
```
# ✗ WRONG: Different types
src = cute.make_tensor(cute.make_shape(100), cutlass.Float32)
dst = cute.make_tensor(cute.make_shape(100), cutlass.Float16)
cute.copy(src, dst)  # May error or produce wrong results

# ✓ CORRECT: Convert explicitly
cute.copy(src.to(cutlass.Float16), dst)
```

**Error 3: Missing synchronization**
```
# ✗ WRONG: No sync after SMEM write
cute.copy(gmem_src, smem_dst)
val = smem_dst[tid]  # Race condition!

# ✓ CORRECT: Sync after SMEM write
cute.copy(gmem_src, smem_dst)
cute.arch.syncthreads()
val = smem_dst[tid]
```

### Debug Copy Operations
```
@cute.kernel
def debug_copy(src: cute.Tensor, dst: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Print before copy
    cute.printf("Thread %d: src[0] = %f\n", tid, src[0])
    
    # Perform copy
    cute.copy(src, dst)
    
    # Sync and verify
    cute.arch.syncthreads()
    cute.printf("Thread %d: dst[0] = %f\n", tid, dst[0])
```

---

## Best Practices

### ✅ DO

**Use copy atoms for tiles:**
```
# Efficient tile copy
cute.copy(tiled_copy, gmem_tile, smem_tile)
```

**Async copy when possible:**
```
# Hide latency
cute.copy(async_atom, gmem, smem)
# do work...
cute.arch.cp_async_wait_all()
```

**Synchronize after SMEM writes:**
```
cute.copy(gmem, smem)
cute.arch.syncthreads()  # Always!
```

**Vectorize aligned accesses:**
```
# Load 4 elements at once
vec = gmem[tid * 4 : tid * 4 + 4]
```

### ❌ DON'T

**Don't use direct assignment for tiles:**
```
# ✗ SLOW: Element-by-element
for i in range(128):
    for j in range(64):
        smem[i, j] = gmem[i, j]

# ✓ FAST: Use copy atom
cute.copy(copy_atom, gmem_tile, smem_tile)
```

**Don't forget to wait for async:**
```
# ✗ WRONG: Using data before copy completes
cute.copy(async_atom, gmem, smem)
val = smem[0]  # May be stale!

# ✓ CORRECT: Wait first
cute.copy(async_atom, gmem, smem)
cute.arch.cp_async_wait_all()
cute.arch.syncthreads()
val = smem[0]
```

---

## Summary

**Copy atoms:**
- Building blocks for efficient data movement
- Abstract hardware instructions
- Enable vectorization, async copy, TMA

**Key patterns:**
- GMEM → SMEM: Use async copy or TMA
- SMEM → RMEM: Use for MMA preparation
- RMEM → GMEM: Vectorize stores when aligned

**Performance tips:**
- Coalesce global memory accesses
- Avoid shared memory bank conflicts
- Use async copy to hide latency
- Pipeline copies with computation
- Vectorize when aligned

**Architecture features:**
- Ampere: cp.async (async copy)
- Hopper: TMA (tensor memory accelerator)
- Blackwell: Enhanced TMA, TMEM

---

## Next Steps

- [MMA Atoms](./mma_atoms.md) - Tensor core operations
- [Memory Hierarchy](../03_memory_and_layouts/memory_hierarchy.md) - Memory spaces
- [Pipelining](../02_control_flow/pipelining.md) - Overlapping copy and compute

---

## Further Reading

- [CUDA Memory Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
- [Async Copy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- [TMA Documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/)