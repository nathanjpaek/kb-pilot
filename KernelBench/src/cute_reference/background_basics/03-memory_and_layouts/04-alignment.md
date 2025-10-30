---
topic: "alignment"
difficulty: "intermediate"
related_topics: ["memory_hierarchy", "tensors", "copy_atoms"]
---

# Memory Alignment in CuTe DSL

## Overview

**Memory alignment** refers to how data is positioned in memory relative to specific byte boundaries. Proper alignment is critical for:
- Vectorized memory operations
- Tensor core operations
- Maximum memory bandwidth
- Avoiding performance penalties

**Key concept:** Aligned addresses allow hardware to perform optimized memory operations. Misaligned addresses force slower, less efficient access patterns.

---

## What is Alignment?

### Address Alignment

An address is **aligned to N bytes** if it's a multiple of N.

**Examples:**
```
Address 0:    Aligned to any power of 2 (1, 2, 4, 8, 16, 32, ...)
Address 16:   Aligned to 16 bytes (good for vectorized loads)
Address 32:   Aligned to 32 bytes
Address 128:  Aligned to 128 bytes (tensor core requirements)

Address 5:    NOT aligned to 4 bytes (5 % 4 = 1)
Address 18:   NOT aligned to 16 bytes (18 % 16 = 2)
```

### Why Alignment Matters

**Modern GPUs have alignment requirements for optimal performance:**
```
Operation                     Required Alignment
------------------------------------------------------
Scalar load (Float32)         4 bytes (natural alignment)
Vectorized load (float4)      16 bytes (4 × 4 bytes)
Vectorized load (float8)      32 bytes (8 × 4 bytes)
Tensor core MMA (Ampere)      128 bytes (varies by atom)
TMA (Hopper)                  16-128 bytes (varies by tile)
```

**Performance impact:**
```
Aligned access:     Full bandwidth (e.g., 2 TB/s)
Misaligned access:  Reduced bandwidth (50-90% slower)
                    or multiple transactions required
```

---

## Alignment in CuTe DSL

### Natural Alignment from Frameworks

**PyTorch and most frameworks provide aligned tensors:**
```
import torch
from cutlass.cute.runtime import from_dlpack

# PyTorch tensor (typically aligned to 256 bytes or more)
torch_tensor = torch.randn(1024, 1024, device="cuda")

# Convert to CuTe - inherits alignment
cute_tensor = from_dlpack(torch_tensor)

# Address is aligned (can verify)
cute.printf("Address: %p\n", cute_tensor.data_ptr)
```

**Common framework alignment:**
- PyTorch: 256-byte alignment (typical)
- JAX: 128-byte alignment (typical)
- TensorFlow: 64-byte alignment (typical)

### Checking Alignment
```
@cute.jit
def check_alignment(tensor: cute.Tensor):
    addr = tensor.data_ptr
    
    # Check various alignments
    is_aligned_4 = (addr % 4) == 0
    is_aligned_16 = (addr % 16) == 0
    is_aligned_128 = (addr % 128) == 0
    
    cute.printf("4-byte aligned: %d\n", is_aligned_4)
    cute.printf("16-byte aligned: %d\n", is_aligned_16)
    cute.printf("128-byte aligned: %d\n", is_aligned_128)
```

---

## cute.assume() for Alignment

### Purpose

Tell the compiler about alignment guarantees to enable optimizations.

**Syntax:**
```
cute.assume(pointer_or_tensor, align=N)
cute.assume(value, divby=N)  # For size divisibility
```

### Basic Usage
```
@cute.kernel
def with_alignment_assumption(data: cute.Tensor):
    # Tell compiler: data pointer is 128-byte aligned
    aligned_data = cute.assume(data, align=128)
    
    # Compiler can now use 128-byte aligned loads
    # May enable better vectorization
    val = aligned_data[idx]
```

### What It Does

**Without assume:**
```
@cute.kernel
def without_assume(data: cute.Tensor):
    # Compiler doesn't know alignment
    # Must use conservative (slower) instructions
    val = data[tid]
    
    # Generated code may use:
    # - Scalar loads (safe but slow)
    # - Unaligned vector loads (slower)
```

**With assume:**
```
@cute.kernel
def with_assume(data: cute.Tensor):
    # Tell compiler about 16-byte alignment
    aligned_data = cute.assume(data, align=16)
    
    val = aligned_data[tid]
    
    # Generated code can use:
    # - Aligned vector loads (fast!)
    # - Optimal memory instructions
```

---

## Common Alignment Values

### 4-Byte Alignment (Float32, Int32)

**Natural alignment for 32-bit types:**
```
@cute.kernel
def float32_alignment(data: cute.Tensor):
    # Float32 requires 4-byte alignment minimum
    aligned_data = cute.assume(data, align=4)
    
    val = aligned_data[tid]  # Safe, aligned access
```

### 16-Byte Alignment (float4, Vectorized)

**Required for 128-bit vectorized operations:**
```
@cute.kernel
def vectorized_load(data: cute.Tensor):
    # Assume 16-byte alignment for vectorization
    aligned_data = cute.assume(data, align=16)
    
    tid = cute.arch.thread_idx()[0]
    
    # Load 4 floats at once (16 bytes)
    vec = aligned_data[tid * 4 : tid * 4 + 4]
```

**Benefits:**
- Single 128-bit load instruction
- 4× fewer memory transactions
- Better bandwidth utilization

### 32-Byte Alignment (float8, Wider Vectors)

**For larger vectorized operations:**
```
@cute.kernel
def wide_vectorized_load(data: cute.Tensor):
    # Assume 32-byte alignment
    aligned_data = cute.assume(data, align=32)
    
    tid = cute.arch.thread_idx()[0]
    
    # Load 8 floats at once (32 bytes)
    vec = aligned_data[tid * 8 : tid * 8 + 8]
```

### 128-Byte Alignment (Tensor Cores)

**Required for optimal tensor core performance:**
```
@cute.kernel
def mma_with_alignment(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    # Tensor core MMA often requires 128-byte alignment
    aligned_A = cute.assume(A, align=128)
    aligned_B = cute.assume(B, align=128)
    aligned_C = cute.assume(C, align=128)
    
    # MMA operations can use optimal paths
    cute.gemm(tiled_mma, aligned_C, aligned_A, aligned_B, aligned_C)
```

---

## Alignment for Different Operations

### Vectorized Loads/Stores
```
@cute.kernel
def vectorized_operations(input: cute.Tensor, output: cute.Tensor):
    # Assume 16-byte alignment for vectorization
    aligned_input = cute.assume(input, align=16)
    aligned_output = cute.assume(output, align=16)
    
    tid = cute.arch.thread_idx()[0]
    
    # Vectorized load (4 × Float32 = 16 bytes)
    vec = aligned_input[tid * 4 : tid * 4 + 4]
    
    # Process
    for i in range(4):
        vec[i] = vec[i] * 2.0
    
    # Vectorized store (16 bytes)
    aligned_output[tid * 4 : tid * 4 + 4] = vec
```

### Async Copy (Ampere)

**cp.async requires 4-byte alignment minimum, prefers 16-byte:**
```
@cute.kernel
def async_copy_alignment(gmem: cute.Tensor):
    smem = cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float32)
    
    # Assume 16-byte alignment for optimal async copy
    aligned_gmem = cute.assume(gmem, align=16)
    
    # Async copy with optimal alignment
    async_atom = cute.make_ampere_async_copy_atom(cutlass.Float32)
    cute.copy(async_atom, aligned_gmem, smem)
```

### TMA (Hopper)

**TMA has strict alignment requirements:**
```
@cute.kernel
def tma_alignment(gmem: cute.Tensor):
    smem = cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float16)
    
    # TMA often requires 128-byte alignment
    aligned_gmem = cute.assume(gmem, align=128)
    
    # Create TMA descriptor
    tma_desc = cute.make_tma_descriptor(
        aligned_gmem.layout,
        smem.layout,
        elem_type=cutlass.Float16
    )
    
    # TMA copy
    cute.copy(tma_desc, aligned_gmem[tile], smem)
```

---

## Divisibility Assumptions with cute.assume()

### Size Divisibility

**Tell compiler that size is divisible by N:**
```
@cute.kernel
def size_divisibility(data: cute.Tensor):
    n = cute.size(data)
    
    # Assume n is divisible by 16
    n_aligned = cute.assume(n, divby=16)
    
    # Compiler can vectorize by 16
    for i in range(n_aligned // 16):
        # Process 16 elements at once
        for j in range(16):
            process(data[i * 16 + j])
```

**Benefits:**
- Enables loop unrolling
- Better vectorization
- Eliminates remainder handling in some cases

### Combined Assumptions
```
@cute.kernel
def combined_assumptions(data: cute.Tensor):
    # Assume pointer is 128-byte aligned
    aligned_data = cute.assume(data, align=128)
    
    # Assume size is divisible by 32
    n = cute.size(aligned_data)
    n_aligned = cute.assume(n, divby=32)
    
    # Now compiler knows:
    # 1. Base address is 128-byte aligned
    # 2. Size is multiple of 32
    # Can generate very efficient code
    
    for i in range(n_aligned // 32):
        # Process 32 elements with optimal alignment
        process_chunk(aligned_data[i * 32 : i * 32 + 32])
```

---

## Framework-Specific Alignment

### PyTorch Tensors

**PyTorch typically provides 256-byte alignment:**
```
@cute.jit
def pytorch_alignment():
    import torch
    
    # PyTorch tensor (256-byte aligned)
    torch_tensor = torch.randn(1024, 1024, device="cuda")
    cute_tensor = from_dlpack(torch_tensor)
    
    # Safe to assume strong alignment
    aligned_tensor = cute.assume(cute_tensor, align=256)
    
    process_kernel.launch(...)(aligned_tensor)
```

### JAX Tensors

**JAX typically provides 128-byte alignment:**
```
@cute.jit
def jax_alignment():
    import jax.numpy as jnp
    from jax.dlpack import to_dlpack
    
    # JAX tensor (128-byte aligned)
    jax_array = jnp.ones((1024, 1024))
    cute_tensor = from_dlpack(to_dlpack(jax_array))
    
    # Safe to assume 128-byte alignment
    aligned_tensor = cute.assume(cute_tensor, align=128)
    
    process_kernel.launch(...)(aligned_tensor)
```

### Manual Allocation

**When allocating manually, ensure alignment:**
```
@cute.jit
def manual_aligned_allocation():
    # CuTe allocations are typically well-aligned
    tensor = cute.make_tensor(
        cute.make_shape(1024, 1024),
        cutlass.Float32
    )
    
    # Typically 256-byte aligned
    aligned_tensor = cute.assume(tensor, align=256)
    
    return aligned_tensor
```

---

## Alignment and Slicing

### Maintaining Alignment

**Slicing can break alignment:**
```
@cute.kernel
def slicing_alignment(data: cute.Tensor):
    # Assume original data is 16-byte aligned
    aligned_data = cute.assume(data, align=16)
    
    # ✓ GOOD: Slice maintains 16-byte alignment
    # (0 * 4 = 0, which is 16-byte aligned)
    slice1 = aligned_data[0:64]  # Still aligned
    
    # ✓ GOOD: Slice at multiple of 4 maintains alignment
    # (4 * 4 = 16, which is 16-byte aligned)
    slice2 = aligned_data[4:68]  # Still aligned
    
    # ✗ BAD: Slice breaks alignment
    # (1 * 4 = 4, which is NOT 16-byte aligned)
    slice3 = aligned_data[1:65]  # Lost 16-byte alignment!
```

### Safe Slicing
```
@cute.kernel
def safe_slicing(data: cute.Tensor):
    aligned_data = cute.assume(data, align=16)
    
    # Slice at aligned offsets only
    # For 16-byte alignment with Float32 (4 bytes):
    # Offset must be multiple of 4 elements
    
    offset = 8  # 8 * 4 = 32 bytes (aligned!)
    safe_slice = aligned_data[offset : offset + 64]
    
    # Can assume slice is still aligned
    aligned_slice = cute.assume(safe_slice, align=16)
```

---

## Alignment in Shared Memory

### Automatic Alignment

**Shared memory allocations are automatically aligned:**
```
@cute.kernel
def smem_alignment():
    # Shared memory is automatically 128-byte aligned (typically)
    smem = cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float32)
    
    # Can safely assume alignment
    aligned_smem = cute.assume(smem, align=128)
```

### Padding Preserves Alignment

**Padding doesn't break alignment:**
```
@cute.kernel
def padded_smem_alignment():
    # With padding (+1 column)
    smem = cute.make_smem_tensor(cute.make_shape(128, 65), cutlass.Float32)
    
    # First element is still aligned
    # (padding just adds to stride, not offset)
    aligned_smem = cute.assume(smem, align=128)
```

---

## Performance Impact

### Aligned vs Unaligned Performance

**Typical performance impact:**
```
Operation                        Aligned      Unaligned    Difference
-----------------------------------------------------------------------
Scalar load/store               100%         100%         None
Vectorized load (float4)        100%         50-70%       30-50% slower
Tensor core MMA                 100%         N/A          May not work!
TMA (Hopper)                    100%         N/A          May not work!
```

### Bandwidth Example
```
Aligned 128-bit loads:
- 1 transaction per 16 bytes
- Full bandwidth utilization

Unaligned 128-bit loads:
- 2 transactions per 16 bytes (crosses boundary)
- 50% bandwidth utilization
```

---

## Common Alignment Issues

### Issue 1: Assuming Wrong Alignment
```
@cute.kernel
def wrong_alignment_assumption(data: cute.Tensor):
    # ✗ WRONG: Assuming 128-byte alignment without verification
    aligned_data = cute.assume(data, align=128)
    # If data is only 16-byte aligned, this may cause issues!
    
    # ✓ CORRECT: Use conservative alignment
    aligned_data = cute.assume(data, align=16)
    # Or verify alignment first
```

### Issue 2: Breaking Alignment with Indexing
```
@cute.kernel
def breaking_alignment(data: cute.Tensor):
    aligned_data = cute.assume(data, align=16)
    
    # ✗ BAD: Odd offset breaks alignment
    subset = aligned_data[1:]  # Now misaligned!
    
    # ✓ GOOD: Even offset (multiple of 4 for Float32)
    subset = aligned_data[4:]  # Still aligned
```

### Issue 3: Mixed Alignment Requirements
```
@cute.kernel
def mixed_alignment(data_a: cute.Tensor, data_b: cute.Tensor):
    # data_a is 128-byte aligned
    # data_b is only 16-byte aligned
    
    # ✗ WRONG: Assuming same alignment for both
    aligned_a = cute.assume(data_a, align=128)
    aligned_b = cute.assume(data_b, align=128)  # May be wrong!
    
    # ✓ CORRECT: Use appropriate alignment for each
    aligned_a = cute.assume(data_a, align=128)
    aligned_b = cute.assume(data_b, align=16)
```

---

## Best Practices

### ✅ DO

**Verify framework alignment guarantees:**
```
# Check PyTorch documentation for alignment
# Typically 256 bytes, but verify for your version
aligned = cute.assume(from_dlpack(torch_tensor), align=256)
```

**Use conservative assumptions when uncertain:**
```
# If unsure, use 16-byte alignment (safe for most operations)
safe_aligned = cute.assume(tensor, align=16)
```

**Align slicing offsets:**
```
# For 16-byte alignment with Float32:
# Use offsets that are multiples of 4 elements
offset = 4 * chunk_id  # Always aligned
slice = tensor[offset : offset + chunk_size]
```

**Profile to verify alignment helps:**
```
# Measure performance with and without alignment assumptions
# Ensure assumptions are correct and beneficial
```

### ❌ DON'T

**Don't assume stronger alignment than guaranteed:**
```
# ✗ WRONG: Assuming 256-byte when only 128-byte guaranteed
aligned = cute.assume(tensor, align=256)  # May be wrong!
```

**Don't use alignment assumptions on unknown data:**
```
# ✗ WRONG: User-provided data may not be aligned
def process_user_data(user_tensor):
    aligned = cute.assume(user_tensor, align=128)  # Risky!
```

**Don't forget alignment after slicing:**
```
# ✗ WRONG: Forgetting that slicing may break alignment
subset = aligned_tensor[7:]  # No longer aligned!
vectorized_process(subset)  # May not work as expected
```

---

## Debugging Alignment Issues

### Print Alignment
```
@cute.kernel
def debug_alignment(data: cute.Tensor):
    addr = data.data_ptr
    
    cute.printf("Address: %p\n", addr)
    cute.printf("Aligned to 4: %d\n", (addr % 4) == 0)
    cute.printf("Aligned to 16: %d\n", (addr % 16) == 0)
    cute.printf("Aligned to 128: %d\n", (addr % 128) == 0)
```

### Alignment Assertion
```
@cute.jit
def assert_alignment(tensor: cute.Tensor, required_align: int):
    addr = tensor.data_ptr
    
    if (addr % required_align) != 0:
        raise ValueError(f"Tensor not aligned to {required_align} bytes!")
    
    return cute.assume(tensor, align=required_align)
```

### Profile Alignment Impact
```
@cute.jit
def profile_alignment_impact():
    import time
    
    torch_tensor = torch.randn(1024, 1024, device="cuda")
    cute_tensor = from_dlpack(torch_tensor)
    
    # Without alignment assumption
    start = time.time()
    for _ in range(1000):
        kernel.launch(...)(cute_tensor)
    cute.arch.device_synchronize()
    time_no_assume = time.time() - start
    
    # With alignment assumption
    aligned_tensor = cute.assume(cute_tensor, align=128)
    start = time.time()
    for _ in range(1000):
        kernel.launch(...)(aligned_tensor)
    cute.arch.device_synchronize()
    time_with_assume = time.time() - start
    
    speedup = time_no_assume / time_with_assume
    print(f"Alignment speedup: {speedup:.2f}×")
```

---

## Summary

**Memory alignment:**
- Critical for vectorization and tensor cores
- Framework tensors typically well-aligned (16-256 bytes)
- Use `cute.assume(tensor, align=N)` to inform compiler

**Common alignment values:**
- 4 bytes: Minimum for Float32/Int32
- 16 bytes: Vectorized operations (float4)
- 32 bytes: Wider vectors (float8)
- 128 bytes: Tensor cores, TMA

**Key principles:**
- Verify framework alignment guarantees
- Use conservative assumptions when uncertain
- Maintain alignment when slicing
- Profile to verify benefit

**Performance impact:**
- Aligned: Full bandwidth
- Misaligned: 30-50% slower
- Wrong alignment: May fail for tensor cores/TMA

---

## Next Steps

- [Memory Hierarchy](./memory_hierarchy.md) - Understanding memory spaces
- [Copy Operations](../04_operations/copy_atoms.md) - Aligned data movement
- [Tensor Cores](../04_operations/mma_atoms.md) - MMA alignment requirements

---

## Further Reading

- [CUDA Memory Alignment](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
- [Vectorized Memory Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types)
- [Tensor Core Requirements](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)