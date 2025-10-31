---
topic: "common_errors"
difficulty: "beginner"
related_topics: ["debugging_techniques", "error_messages", "validation"]
---

# Common Errors in CuTe DSL

## Overview

This guide covers the most common errors encountered when writing CuTe DSL kernels, their causes, and how to fix them. Understanding these errors will save you hours of debugging time.

**Key concepts:**
- Compilation errors
- Runtime errors
- Memory errors
- Type errors
- Launch configuration errors
- Synchronization errors

---

## Error Categories

### Quick Reference

**Error types by frequency:**

1. **Type Errors** (40%) - Missing annotations, wrong types
2. **Memory Errors** (25%) - Out of bounds, alignment issues
3. **Launch Errors** (15%) - Invalid grid/block dimensions
4. **Synchronization Errors** (10%) - Missing barriers, race conditions
5. **Compilation Errors** (10%) - Syntax, imports, JIT issues

---

## Type Errors

### Error 1: Missing Type Annotation

**Problem:**
```python
@cute.kernel
def missing_annotation(data):  # No type!
    data[0] = 1.0

# TypeError: Parameter 'data' is missing type annotation
```

**Cause:** All kernel parameters must have type annotations.

**Fix:**
```python
@cute.kernel
def with_annotation(data: cute.Tensor):
    data[0] = 1.0
```

### Error 2: Using Python int Instead of cutlass.Int32

**Problem:**
```python
@cute.kernel
def wrong_type(data: cute.Tensor, n: int):  # Python int!
    if cute.arch.thread_idx()[0] < n:
        data[cute.arch.thread_idx()[0]] = 0.0

# TypeError: Expected cutlass.Int32, got Python int
```

**Cause:** Use CuTe types, not Python types in kernels.

**Fix:**
```python
@cute.kernel
def correct_type(data: cute.Tensor, n: cutlass.Int32):
    if cute.arch.thread_idx()[0] < n:
        data[cute.arch.thread_idx()[0]] = 0.0
```

### Error 3: Type Mismatch in Operations

**Problem:**
```python
@cute.kernel
def type_mismatch(data: cute.Tensor, scale: cutlass.Int32):
    tid = cute.arch.thread_idx()[0]
    # Float tensor with int scale
    data[tid] = data[tid] * scale  # May cause issues

# Warning: Implicit conversion from Int32 to Float32
```

**Cause:** Mixing incompatible types without explicit conversion.

**Fix:**
```python
@cute.kernel
def explicit_conversion(data: cute.Tensor, scale: cutlass.Int32):
    tid = cute.arch.thread_idx()[0]
    # Explicit conversion
    scale_float = cutlass.Float32(scale)
    data[tid] = data[tid] * scale_float
```

---

## Memory Errors

### Error 4: Out of Bounds Access

**Problem:**
```python
@cute.kernel
def out_of_bounds(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    idx = bid * 256 + tid
    data[idx] = 0.0  # No bounds check!

# CUDA Error: an illegal memory access was encountered
```

**Cause:** Accessing memory outside tensor bounds.

**Fix:**
```python
@cute.kernel
def with_bounds_check(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    idx = bid * 256 + tid
    # Always check bounds!
    if idx < cute.size(data):
        data[idx] = 0.0
```

**Best practice:**
```python
def calculate_safe_grid(n: int, block_size: int) -> tuple:
    """Calculate grid size that ensures coverage"""
    num_blocks = (n + block_size - 1) // block_size
    return (num_blocks, 1, 1)

# Use safe grid
n = data.numel()
grid = calculate_safe_grid(n, 256)
block = (256, 1, 1)
kernel.launch(grid=grid, block=block)(data)
```

### Error 5: Shared Memory Out of Bounds

**Problem:**
```python
@cute.kernel
def smem_overflow():
    smem = cute.make_smem_tensor(cute.make_shape(32), cutlass.Float32)
    
    tid = cute.arch.thread_idx()[0]
    # Thread 64 accessing smem[64] but smem only has 32 elements!
    smem[tid] = 1.0  # Out of bounds for tid >= 32

# CUDA Error: illegal memory access
```

**Cause:** Shared memory allocation too small.

**Fix:**
```python
@cute.kernel
def smem_correct():
    # Allocate enough for all threads
    smem = cute.make_smem_tensor(cute.make_shape(256), cutlass.Float32)
    
    tid = cute.arch.thread_idx()[0]
    if tid < 256:  # Bounds check
        smem[tid] = 1.0
```

### Error 6: Uninitialized Memory

**Problem:**
```python
@cute.kernel
def uninitialized(output: cute.Tensor):
    acc = cute.make_rmem_tensor(cute.make_shape(8), cutlass.Float32)
    # acc contains garbage!
    
    for i in range(8):
        acc[i] = acc[i] + 1.0  # Undefined behavior!
    
    output[0] = acc[0]

# Output: Random garbage values
```

**Cause:** Using memory before initializing.

**Fix:**
```python
@cute.kernel
def initialized(output: cute.Tensor):
    acc = cute.make_rmem_tensor(cute.make_shape(8), cutlass.Float32)
    # Initialize to zero
    acc.store(0.0)
    
    for i in range(8):
        acc[i] = acc[i] + 1.0  # Now well-defined
    
    output[0] = acc[0]
```

---

## Launch Configuration Errors

### Error 7: Too Many Threads Per Block

**Problem:**
```python
grid = [10, 1, 1]
block = [2048, 1, 1]  # Too many!

kernel.launch(grid=grid, block=block)(data)

# CUDA Error: invalid configuration argument
```

**Cause:** Maximum threads per block is 1024 on most GPUs.

**Fix:**
```python
# Valid configuration
grid = [20, 1, 1]
block = [1024, 1, 1]  # Max allowed

kernel.launch(grid=grid, block=block)(data)
```

### Error 8: Non-Warp-Multiple Block Size

**Problem:**
```python
block = [250, 1, 1]  # Not multiple of 32!

kernel.launch(grid=[10,1,1], block=block)(data)

# Works, but inefficient
# Warning: Suboptimal block size
```

**Cause:** Block size should be multiple of warp size (32) for efficiency.

**Fix:**
```python
# Use multiple of 32
block = [256, 1, 1]  # 32 × 8

kernel.launch(grid=[10,1,1], block=block)(data)
```

### Error 9: 2D Block Exceeds Limit

**Problem:**
```python
# 64 × 64 = 4096 threads (> 1024 limit!)
block = [64, 64, 1]

kernel.launch(grid=[10,10,1], block=block)(data)

# CUDA Error: invalid configuration argument
```

**Cause:** Total threads (x × y × z) exceeds 1024.

**Fix:**
```python
# 32 × 32 = 1024 threads (OK)
block = [32, 32, 1]

kernel.launch(grid=[10,10,1], block=block)(data)
```

---

## Synchronization Errors

### Error 10: Missing syncthreads

**Problem:**
```python
@cute.kernel
def missing_sync():
    smem = cute.make_smem_tensor(cute.make_shape(256), cutlass.Float32)
    
    tid = cute.arch.thread_idx()[0]
    
    # Write to shared memory
    smem[tid] = tid
    
    # Immediately read (no sync!)
    value = smem[(tid + 1) % 256]  # Race condition!

# Output: Undefined/incorrect results
```

**Cause:** Reading shared memory before all threads finish writing.

**Fix:**
```python
@cute.kernel
def with_sync():
    smem = cute.make_smem_tensor(cute.make_shape(256), cutlass.Float32)
    
    tid = cute.arch.thread_idx()[0]
    
    # Write to shared memory
    smem[tid] = tid
    
    # Synchronize!
    cute.arch.syncthreads()
    
    # Now safe to read
    value = smem[(tid + 1) % 256]
```

### Error 11: Deadlock from Conditional Sync

**Problem:**
```python
@cute.kernel
def conditional_deadlock():
    tid = cute.arch.thread_idx()[0]
    
    if tid < 128:
        # Only half the threads reach sync
        cute.arch.syncthreads()  # DEADLOCK!
    
    # Other threads waiting forever

# Kernel hangs
```

**Cause:** Not all threads in block reach `syncthreads()`.

**Fix:**
```python
@cute.kernel
def unconditional_sync():
    tid = cute.arch.thread_idx()[0]
    
    # Conditional work
    if tid < 128:
        do_work()
    
    # But unconditional sync (all threads reach it)
    cute.arch.syncthreads()
```

### Error 12: Race Condition in Global Memory

**Problem:**
```python
@cute.kernel
def global_race(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Multiple threads write to same location
    data[0] = data[0] + tid  # Race condition!

# Output: Undefined (lost updates)
```

**Cause:** Multiple threads modifying same memory without synchronization.

**Fix (atomic operations):**
```python
@cute.kernel
def with_atomic(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Use atomic operation
    cute.arch.atomic_add(data, 0, tid)

# Or: Reduce in shared memory first, then atomic add once
@cute.kernel
def optimized(data: cute.Tensor):
    smem = cute.make_smem_tensor(cute.make_shape(256), cutlass.Float32)
    
    tid = cute.arch.thread_idx()[0]
    
    # Accumulate in shared memory
    smem[tid] = tid
    cute.arch.syncthreads()
    
    # Tree reduction
    stride = 128
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        cute.arch.syncthreads()
        stride //= 2
    
    # Single atomic add
    if tid == 0:
        cute.arch.atomic_add(data, 0, smem[0])
```

---

## Tensor/Shape Errors

### Error 13: Shape Mismatch

**Problem:**
```python
@cute.kernel
def shape_mismatch(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    # Assuming A: M×K, B: K×N, C: M×N
    # But user passes incompatible shapes
    pass

# Launch with A: 128×256, B: 512×128 (K doesn't match!)
A = torch.randn(128, 256, device='cuda')
B = torch.randn(512, 128, device='cuda')
C = torch.randn(128, 128, device='cuda')

kernel.launch(...)(cute.from_dlpack(A), cute.from_dlpack(B), cute.from_dlpack(C))

# Error or wrong results
```

**Cause:** Tensor shapes don't match expectations.

**Fix:**
```python
def validate_gemm_shapes(A, B, C):
    """Validate shapes before kernel launch"""
    M, K1 = A.shape
    K2, N = B.shape
    M2, N2 = C.shape
    
    assert K1 == K2, f"K dimension mismatch: A has {K1}, B has {K2}"
    assert M == M2, f"M dimension mismatch: A has {M}, C has {M2}"
    assert N == N2, f"N dimension mismatch: B has {N}, C has {N2}"
    
    print(f"✓ Shapes valid: ({M}×{K1}) @ ({K2}×{N}) → ({M2}×{N2})")

# Use before launch
validate_gemm_shapes(A, B, C)
kernel.launch(...)(cute.from_dlpack(A), cute.from_dlpack(B), cute.from_dlpack(C))
```

### Error 14: Non-Contiguous Tensor

**Problem:**
```python
# Create non-contiguous tensor (transpose)
A = torch.randn(128, 256, device='cuda')
A_t = A.t()  # Transposed (non-contiguous)

print(A_t.is_contiguous())  # False

cute_A_t = cute.from_dlpack(A_t)
kernel.launch(...)(cute_A_t)

# May work but inefficient, or cause errors
```

**Cause:** Non-contiguous memory layout.

**Fix:**
```python
# Option 1: Make contiguous
A_t_cont = A_t.contiguous()
cute_A_t = cute.from_dlpack(A_t_cont)

# Option 2: Check and warn
if not A_t.is_contiguous():
    print("⚠️  Tensor not contiguous, making copy")
    A_t = A_t.contiguous()

cute_A_t = cute.from_dlpack(A_t)
kernel.launch(...)(cute_A_t)
```

---

## Device Errors

### Error 15: CPU Tensor Passed to Kernel

**Problem:**
```python
# Create CPU tensor
cpu_tensor = torch.randn(1024)  # On CPU!

cute_tensor = cute.from_dlpack(cpu_tensor)
kernel.launch(...)(cute_tensor)

# Error: DLPack requires CUDA tensor
```

**Cause:** Tensor not on GPU.

**Fix:**
```python
# Check device
assert cpu_tensor.is_cuda, "Tensor must be on CUDA device"

# Or move to GPU
gpu_tensor = cpu_tensor.cuda()
cute_tensor = cute.from_dlpack(gpu_tensor)
kernel.launch(...)(cute_tensor)
```

### Error 16: Wrong GPU Device

**Problem:**
```python
# Tensor on GPU 0
with torch.cuda.device(0):
    tensor = torch.randn(1024, device='cuda')

# Kernel launch on GPU 1
with torch.cuda.device(1):
    kernel.launch(...)(cute.from_dlpack(tensor))

# Error: Tensor on different device
```

**Cause:** Tensor and kernel on different GPUs.

**Fix:**
```python
# Ensure same device
device = 0
with torch.cuda.device(device):
    tensor = torch.randn(1024, device=f'cuda:{device}')
    kernel.launch(...)(cute.from_dlpack(tensor))

# Or check device
def validate_device(tensor, expected_device=0):
    assert tensor.is_cuda, "Tensor must be on CUDA"
    assert tensor.device.index == expected_device, \
        f"Tensor on GPU {tensor.device.index}, expected {expected_device}"
```

---

## Compilation Errors

### Error 17: Import Errors

**Problem:**
```python
import cutlass.cute as cute

@cute.kernel
def my_kernel(data: cute.Tensor):
    # Using undefined function
    result = cute.some_function_that_doesnt_exist()

# AttributeError: module 'cutlass.cute' has no attribute 'some_function_that_doesnt_exist'
```

**Cause:** Typo or using non-existent function.

**Fix:**
```python
# Check available functions
print(dir(cute))

# Use correct function names
# Common functions:
# - cute.size()
# - cute.shape()
# - cute.make_tensor()
# - cute.make_layout()
# - cute.arch.thread_idx()
# - cute.arch.block_idx()
# - cute.arch.syncthreads()
```

### Error 18: JIT Compilation Failure

**Problem:**
```python
@cute.kernel
def jit_failure(data: cute.Tensor):
    # Complex Python code that can't be JIT compiled
    x = [1, 2, 3]  # Python list
    for i in x:
        data[i] = i

# JIT compilation error: Unsupported Python construct
```

**Cause:** Using Python features not supported in kernel.

**Fix:**
```python
@cute.kernel
def jit_compatible(data: cute.Tensor):
    # Use only supported constructs
    for i in range(3):
        data[i] = cutlass.Float32(i)
```

---

## Common Pitfalls

### Pitfall 1: Forgetting to Synchronize
```python
# Launch kernel
kernel.launch(...)(data)

# Immediately use result (may be incomplete!)
print(data.sum())  # Wrong!

# Fix: Synchronize first
kernel.launch(...)(data)
torch.cuda.synchronize()
print(data.sum())  # Correct
```

### Pitfall 2: Integer Division
```python
@cute.kernel
def integer_division():
    # Python 3 division
    result = 5 / 2  # Expects 2.5
    # But in kernel: integer division! result = 2

# Fix: Use explicit float
@cute.kernel
def float_division():
    result = 5.0 / 2.0  # 2.5
```

### Pitfall 3: Shared Memory Initialization
```python
@cute.kernel
def uninitialized_smem():
    smem = cute.make_smem_tensor(cute.make_shape(256), cutlass.Float32)
    # smem contains garbage!
    
    cute.arch.syncthreads()
    
    # Using uninitialized values
    value = smem[0]  # Undefined

# Fix: Initialize
@cute.kernel
def initialized_smem():
    smem = cute.make_smem_tensor(cute.make_shape(256), cutlass.Float32)
    
    tid = cute.arch.thread_idx()[0]
    smem[tid] = 0.0  # Initialize
    
    cute.arch.syncthreads()
    
    value = smem[0]  # Well-defined
```

### Pitfall 4: Atomic Operations on Float32
```python
@cute.kernel
def float_atomic_precision(result: cute.Tensor):
    # Atomic add on float may lose precision
    cute.arch.atomic_add(result, 0, 0.0001)
    # If many threads do this, precision lost

# Fix: Consider alternatives
# - Use double precision
# - Accumulate in shared memory first
# - Use integer atomics if possible
```

---

## Error Checking Utilities

### Comprehensive Validation
```python
def validate_kernel_inputs(
    kernel_name: str,
    tensors: dict,
    expected_shapes: dict,
    expected_dtypes: dict,
    device: int = 0
):
    """
    Comprehensive input validation
    
    Args:
        kernel_name: Name of kernel for error messages
        tensors: Dict of tensor_name → tensor
        expected_shapes: Dict of tensor_name → expected shape (can include None for any)
        expected_dtypes: Dict of tensor_name → expected dtype
        device: Expected CUDA device
    """
    
    print(f"Validating inputs for {kernel_name}...")
    
    for name, tensor in tensors.items():
        # Check is tensor
        assert isinstance(tensor, torch.Tensor), \
            f"{name} is not a tensor"
        
        # Check device
        assert tensor.is_cuda, \
            f"{name} is not on CUDA device"
        assert tensor.device.index == device, \
            f"{name} on GPU {tensor.device.index}, expected GPU {device}"
        
        # Check dtype
        if name in expected_dtypes:
            expected_dtype = expected_dtypes[name]
            assert tensor.dtype == expected_dtype, \
                f"{name} has dtype {tensor.dtype}, expected {expected_dtype}"
        
        # Check shape
        if name in expected_shapes:
            expected_shape = expected_shapes[name]
            for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                if expected is not None:
                    assert actual == expected, \
                        f"{name} dim {i}: {actual} != {expected}"
        
        # Check contiguous
        if not tensor.is_contiguous():
            print(f"⚠️  {name} is not contiguous")
        
        print(f"✓ {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    
    print(f"✓ All inputs valid for {kernel_name}")

# Usage
validate_kernel_inputs(
    kernel_name="gemm",
    tensors={'A': A, 'B': B, 'C': C},
    expected_shapes={
        'A': (M, K),
        'B': (K, N),
        'C': (M, N)
    },
    expected_dtypes={
        'A': torch.float16,
        'B': torch.float16,
        'C': torch.float16
    },
    device=0
)
```

---

## Debugging Checklist

**Before launching kernel:**
- [ ] All parameters have type annotations?
- [ ] Tensors are on CUDA device?
- [ ] Tensors are contiguous (or handled)?
- [ ] Shapes are compatible?
- [ ] Grid/block dimensions valid?
- [ ] Block size ≤ 1024?
- [ ] Block size multiple of 32?

**In kernel:**
- [ ] Bounds checking for all array accesses?
- [ ] Memory initialized before use?
- [ ] `syncthreads()` after writing to shared memory?
- [ ] All threads reach `syncthreads()` (unconditional)?
- [ ] No race conditions on shared/global memory?

**After kernel:**
- [ ] `torch.cuda.synchronize()` before using results?
- [ ] Results validated against reference?
- [ ] Check for CUDA errors?

---

## Summary

**Most common errors:**
1. Missing type annotations → Add `: cute.Tensor`, `: cutlass.Int32`
2. Out of bounds access → Add bounds checking
3. Missing syncthreads → Add barriers
4. Too many threads → Use ≤ 1024 threads/block
5. CPU tensors → Use `.cuda()` to move to GPU

**Quick fixes:**
- Type errors → Add annotations
- Memory errors → Add bounds checks
- Launch errors → Validate grid/block
- Sync errors → Add `syncthreads()`
- Device errors → Check `.is_cuda`

**Prevention:**
- Use validation functions
- Follow checklist
- Test with small inputs first
- Check each error message carefully

**Key insight:** Most errors are simple mistakes that are easy to fix once you know what to look for. Always validate inputs and check bounds!

---

## Next Steps

- [Debugging Techniques](./debugging_techniques.md) - How to debug
- [Error Messages](./error_messages.md) - Understanding errors
- [Validation](./validation.md) - Testing kernels

---

## Further Reading

- [CUDA Error Codes](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Debugging CUDA](https://docs.nvidia.com/cuda/cuda-gdb/index.html)