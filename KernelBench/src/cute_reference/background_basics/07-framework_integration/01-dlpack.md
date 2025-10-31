---
topic: "dlpack"
difficulty: "intermediate"
related_topics: ["pytorch_integration", "tensors", "memory_hierarchy"]
---

# DLPack Integration in CuTe DSL

## Overview

**DLPack** is a standard in-memory tensor exchange format that enables zero-copy sharing of tensors between frameworks. CuTe DSL uses DLPack to seamlessly interoperate with PyTorch, JAX, TensorFlow, and other frameworks.

**Key concepts:**
- What is DLPack
- `from_dlpack()` conversion
- Zero-copy tensor sharing
- Memory layout preservation
- Framework interoperability
- DLPack capsule format

---

## What is DLPack?

### The Interoperability Problem

**Without DLPack:**
```python
# Framework A has a tensor
pytorch_tensor = torch.randn(1024, device='cuda')

# Framework B needs to use it
# Must copy memory (slow + wasteful)
numpy_array = pytorch_tensor.cpu().numpy()  # Copy 1: GPU → CPU
cuda_array = cupy.array(numpy_array)        # Copy 2: CPU → GPU
# 2 expensive copies!
```

### With DLPack

**Zero-copy sharing:**
```python
# Framework A
pytorch_tensor = torch.randn(1024, device='cuda')

# Framework B (zero copy!)
cute_tensor = cute.from_dlpack(pytorch_tensor)
# No memory copy - both share the same GPU memory!
```

**Benefits:**
- ✅ Zero-copy (instant)
- ✅ Memory efficient
- ✅ Preserves GPU memory
- ✅ Framework agnostic

---

## DLPack Format

### What Gets Shared

**DLPack capsule contains:**

1. **Data pointer** - GPU memory address
2. **Shape** - Tensor dimensions
3. **Strides** - Memory layout
4. **Dtype** - Element type
5. **Device** - CPU/CUDA/etc.
```python
# PyTorch tensor
tensor = torch.randn(128, 256, device='cuda', dtype=torch.float32)

# DLPack capsule contains:
# - data_ptr: 0x7f8a4c000000 (GPU address)
# - shape: (128, 256)
# - strides: (256, 1)  # Row-major
# - dtype: float32
# - device: CUDA device 0
```

### Memory Layout Preservation

**Strides are preserved:**
```python
# Row-major tensor
row_major = torch.randn(128, 256, device='cuda')
print(row_major.stride())  # (256, 1)

cute_tensor = cute.from_dlpack(row_major)
# CuTe tensor uses same strides: (256, 1)

# Column-major tensor
col_major = row_major.t().contiguous()
print(col_major.stride())  # (1, 256)

cute_tensor = cute.from_dlpack(col_major)
# CuTe tensor uses same strides: (1, 256)
```

---

## Using from_dlpack()

### Basic Conversion

**Convert PyTorch tensor to CuTe:**
```python
import torch
import cutlass.cute as cute

# Create PyTorch tensor
pt_tensor = torch.randn(1024, device='cuda')

# Convert to CuTe (zero-copy)
cute_tensor = cute.from_dlpack(pt_tensor)

# Both reference same memory
print(pt_tensor.data_ptr())    # 0x7f8a4c000000
print(cute_tensor.data_ptr())  # 0x7f8a4c000000 (same!)
```

### Multi-Dimensional Tensors
```python
# 1D
vec = torch.randn(1024, device='cuda')
cute_vec = cute.from_dlpack(vec)

# 2D
mat = torch.randn(128, 256, device='cuda')
cute_mat = cute.from_dlpack(mat)

# 3D
tensor_3d = torch.randn(32, 64, 128, device='cuda')
cute_3d = cute.from_dlpack(tensor_3d)

# 4D
tensor_4d = torch.randn(16, 32, 64, 128, device='cuda')
cute_4d = cute.from_dlpack(tensor_4d)
```

### Different Data Types
```python
# Float types
fp32 = torch.randn(1024, device='cuda', dtype=torch.float32)
cute.from_dlpack(fp32)  # Float32

fp16 = torch.randn(1024, device='cuda', dtype=torch.float16)
cute.from_dlpack(fp16)  # Float16

bf16 = torch.randn(1024, device='cuda', dtype=torch.bfloat16)
cute.from_dlpack(bf16)  # BFloat16

# Integer types
int32 = torch.zeros(1024, device='cuda', dtype=torch.int32)
cute.from_dlpack(int32)  # Int32

int8 = torch.zeros(1024, device='cuda', dtype=torch.int8)
cute.from_dlpack(int8)  # Int8
```

---

## Zero-Copy Semantics

### Shared Memory

**Changes are bidirectional:**
```python
# PyTorch tensor
pt_tensor = torch.zeros(10, device='cuda')
print(pt_tensor)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Convert to CuTe
cute_tensor = cute.from_dlpack(pt_tensor)

# Modify via CuTe kernel
@cute.kernel
def modify_kernel(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    data[tid] = 999.0

modify_kernel.launch(grid=[1,1,1], block=[10,1,1])(cute_tensor)
torch.cuda.synchronize()

# Changes visible in PyTorch!
print(pt_tensor)  # [999, 999, 999, 999, 999, 999, 999, 999, 999, 999]
```

### Memory Lifetime

**Original tensor owns memory:**
```python
def get_cute_tensor():
    # PyTorch tensor created in function
    pt_tensor = torch.randn(1024, device='cuda')
    cute_tensor = cute.from_dlpack(pt_tensor)
    
    # pt_tensor goes out of scope here!
    return cute_tensor

# ⚠️ DANGER: PyTorch tensor deleted, memory may be freed
cute_tensor = get_cute_tensor()
# cute_tensor may point to invalid memory!
```

**Safe pattern:**
```python
def safe_pattern():
    # Keep PyTorch tensor alive
    pt_tensor = torch.randn(1024, device='cuda')
    cute_tensor = cute.from_dlpack(pt_tensor)
    
    # Use kernel
    kernel.launch(...)(cute_tensor)
    torch.cuda.synchronize()
    
    # Both tensors still valid
    return pt_tensor  # Keep alive by returning

# Or store as instance variable
class Model:
    def __init__(self):
        self.tensor = torch.randn(1024, device='cuda')
        self.cute_tensor = cute.from_dlpack(self.tensor)
```

---

## Implicit DLPack Conversion

### Automatic Conversion

**CuTe may auto-convert in some contexts:**
```python
import torch

# PyTorch tensor
pt_tensor = torch.randn(1024, device='cuda')

# Some CuTe APIs accept PyTorch tensors directly
# (internally calls from_dlpack)
@cute.jit
def may_auto_convert(tensor):  # May accept torch.Tensor
    # Internally converted if needed
    pass

# Check documentation for specific API
```

### Explicit is Better

**Recommended: Always be explicit:**
```python
# ✓ GOOD: Explicit conversion
pt_tensor = torch.randn(1024, device='cuda')
cute_tensor = cute.from_dlpack(pt_tensor)
kernel.launch(...)(cute_tensor)

# Rather than relying on implicit conversion
```

---

## DLPack Compatibility

### Supported Frameworks

**DLPack works with:**

1. **PyTorch**
```python
import torch
pt_tensor = torch.randn(1024, device='cuda')
cute_tensor = cute.from_dlpack(pt_tensor)
```

2. **CuPy**
```python
import cupy as cp
cp_array = cp.random.randn(1024)
cute_tensor = cute.from_dlpack(cp_array)
```

3. **JAX**
```python
import jax.numpy as jnp
jax_array = jnp.ones(1024)
cute_tensor = cute.from_dlpack(jax_array)
```

4. **TensorFlow** (via to_dlpack)
```python
import tensorflow as tf
tf_tensor = tf.ones((1024,), dtype=tf.float32)
dlpack_capsule = tf.experimental.dlpack.to_dlpack(tf_tensor)
cute_tensor = cute.from_dlpack(dlpack_capsule)
```

5. **NumPy** (CPU only)
```python
import numpy as np
import cupy as cp

# NumPy doesn't have GPU support, use CuPy as bridge
np_array = np.random.randn(1024)
cp_array = cp.array(np_array)  # CPU → GPU
cute_tensor = cute.from_dlpack(cp_array)
```

### Version Compatibility

**DLPack versions:**

- **DLPack 0.5+**: Most common, widely supported
- Check framework documentation for DLPack support
```python
# Check if framework supports DLPack
import torch
print(hasattr(torch.Tensor, '__dlpack__'))  # True if supported
print(hasattr(torch.Tensor, '__dlpack_device__'))  # True if supported
```

---

## Advanced DLPack Usage

### Manual DLPack Capsule

**Create capsule explicitly:**
```python
import torch

# PyTorch tensor
pt_tensor = torch.randn(1024, device='cuda')

# Get DLPack capsule
dlpack_capsule = pt_tensor.__dlpack__()

# Convert to CuTe
cute_tensor = cute.from_dlpack(dlpack_capsule)
```

### Device Information

**Check device before conversion:**
```python
import torch

pt_tensor = torch.randn(1024, device='cuda:0')

# Check device
device_type, device_id = pt_tensor.__dlpack_device__()
print(f"Device: {device_type}, ID: {device_id}")
# Output: Device: 2 (CUDA), ID: 0

# Convert only if on correct device
if device_type == 2:  # CUDA
    cute_tensor = cute.from_dlpack(pt_tensor)
```

### Strided Tensors

**Non-contiguous tensors:**
```python
# Contiguous tensor
cont = torch.randn(128, 256, device='cuda')
print(cont.stride())  # (256, 1)
cute.from_dlpack(cont)  # ✓ OK

# Transposed (non-contiguous)
trans = cont.t()
print(trans.stride())  # (1, 128)
print(trans.is_contiguous())  # False
cute.from_dlpack(trans)  # ✓ Still works! Strides preserved

# Sliced (non-contiguous)
sliced = cont[:, ::2]  # Every other column
print(sliced.stride())  # (256, 2)
cute.from_dlpack(sliced)  # ✓ Works, but be careful in kernel
```

**Handling non-contiguous:**
```python
def safe_from_dlpack(tensor: torch.Tensor):
    """Convert to CuTe, ensuring contiguous"""
    if not tensor.is_contiguous():
        print("Warning: Tensor not contiguous, making contiguous")
        tensor = tensor.contiguous()
    return cute.from_dlpack(tensor)
```

---

## Common Patterns

### Pattern 1: Framework Bridge

**Pass tensors between frameworks:**
```python
import torch
import cupy as cp

# PyTorch → CuTe
pt_tensor = torch.randn(1024, device='cuda')
cute_tensor = cute.from_dlpack(pt_tensor)

# Process with CuTe kernel
kernel.launch(...)(cute_tensor)
torch.cuda.synchronize()

# CuTe → CuPy (via PyTorch)
cp_array = cp.from_dlpack(pt_tensor)

# Now can use CuPy operations
result = cp.sum(cp_array)
```

### Pattern 2: Batch Processing

**Process multiple tensors:**
```python
def process_batch(tensors: list[torch.Tensor]):
    """Process batch of PyTorch tensors with CuTe"""
    
    cute_tensors = [cute.from_dlpack(t) for t in tensors]
    
    for cute_t in cute_tensors:
        kernel.launch(...)(cute_t)
    
    torch.cuda.synchronize()
    
    # Results in original PyTorch tensors
    return tensors
```

### Pattern 3: In-Place Operations

**Modify tensors in-place:**
```python
def inplace_operation(tensor: torch.Tensor) -> torch.Tensor:
    """Modify PyTorch tensor in-place via CuTe"""
    
    cute_tensor = cute.from_dlpack(tensor)
    
    # In-place kernel
    inplace_kernel.launch(...)(cute_tensor)
    torch.cuda.synchronize()
    
    # tensor is modified
    return tensor

# Usage
x = torch.randn(1024, device='cuda')
inplace_operation(x)  # x modified
```

---

## Error Handling

### Common Errors

**1. CPU tensor:**
```python
cpu_tensor = torch.randn(1024, device='cpu')
cute.from_dlpack(cpu_tensor)  # Error!

# Error: DLPack requires CUDA tensor
# Fix: Move to GPU first
gpu_tensor = cpu_tensor.cuda()
cute.from_dlpack(gpu_tensor)  # ✓ OK
```

**2. Unsupported dtype:**
```python
complex_tensor = torch.randn(1024, device='cuda', dtype=torch.complex64)
cute.from_dlpack(complex_tensor)  # May error

# Error: Complex dtypes not supported
# Fix: Use real dtypes (float32, int32, etc.)
```

**3. Wrong device:**
```python
# Tensor on GPU 1
tensor_gpu1 = torch.randn(1024, device='cuda:1')

# Current device GPU 0
torch.cuda.set_device(0)

cute.from_dlpack(tensor_gpu1)  # May work, but be careful!
# Kernel will run on wrong GPU
```

### Validation Helper
```python
def validate_for_dlpack(tensor: torch.Tensor):
    """Validate tensor can be safely converted via DLPack"""
    
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA")
    
    if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16,
                           torch.int32, torch.int8, torch.uint8]:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    if not tensor.is_contiguous():
        print("Warning: Tensor is not contiguous")
    
    return True

# Usage
if validate_for_dlpack(tensor):
    cute_tensor = cute.from_dlpack(tensor)
```

---

## Performance Considerations

### Zero-Copy Benefits

**Benchmark: Copy vs Zero-Copy:**
```python
import torch
import time

tensor = torch.randn(1024*1024, device='cuda')  # 4 MB

# Traditional copy
start = time.time()
copy = tensor.clone()
torch.cuda.synchronize()
copy_time = time.time() - start

# DLPack (zero-copy)
start = time.time()
cute_tensor = cute.from_dlpack(tensor)
torch.cuda.synchronize()
dlpack_time = time.time() - start

print(f"Copy: {copy_time*1000:.3f} ms")
print(f"DLPack: {dlpack_time*1000:.3f} ms")
print(f"Speedup: {copy_time/dlpack_time:.0f}×")

# Typical output:
# Copy: 0.452 ms
# DLPack: 0.003 ms
# Speedup: 150×
```

### When Zero-Copy Matters

**Large tensors:**
```python
# Small tensor (1 KB): Copy vs DLPack similar
small = torch.randn(256, device='cuda')

# Large tensor (100 MB): DLPack much faster
large = torch.randn(1024*1024*25, device='cuda')
cute.from_dlpack(large)  # Instant!
```

**Frequent conversions:**
```python
# ✗ BAD: Copy every iteration
for i in range(1000):
    copy = tensor.clone()  # 1000 copies!
    process(copy)

# ✓ GOOD: DLPack once
cute_tensor = cute.from_dlpack(tensor)
for i in range(1000):
    process(cute_tensor)  # Zero copies!
```

---

## Debugging DLPack

### Inspecting DLPack Capsule
```python
import torch

tensor = torch.randn(128, 256, device='cuda', dtype=torch.float32)

# Get capsule info
device_type, device_id = tensor.__dlpack_device__()
print(f"Device Type: {device_type}")  # 2 = CUDA
print(f"Device ID: {device_id}")      # GPU number

# Check version
print(f"DLPack version: {tensor.__dlpack__().__version__}")
```

### Comparing Tensors
```python
def compare_tensors(pt_tensor: torch.Tensor, cute_tensor):
    """Verify PyTorch and CuTe tensors share memory"""
    
    print(f"PyTorch data_ptr: {pt_tensor.data_ptr():x}")
    print(f"CuTe data_ptr: {cute_tensor.data_ptr():x}")
    
    if pt_tensor.data_ptr() == cute_tensor.data_ptr():
        print("✓ Tensors share memory (zero-copy)")
    else:
        print("✗ Tensors have different memory")
    
    print(f"PyTorch shape: {pt_tensor.shape}")
    print(f"CuTe shape: {cute_tensor.shape}")
    
    print(f"PyTorch stride: {pt_tensor.stride()}")
    print(f"CuTe stride: {cute_tensor.stride()}")
```

---

## Best Practices

### ✅ DO

**Use DLPack for zero-copy:**
```python
cute_tensor = cute.from_dlpack(pt_tensor)  # Fast!
```

**Keep original tensor alive:**
```python
class Model:
    def __init__(self):
        self.pt_tensor = torch.randn(1024, device='cuda')
        self.cute_tensor = cute.from_dlpack(self.pt_tensor)
```

**Validate before conversion:**
```python
assert tensor.is_cuda, "Must be CUDA tensor"
assert tensor.is_contiguous(), "Should be contiguous"
cute_tensor = cute.from_dlpack(tensor)
```

**Synchronize after kernel:**
```python
kernel.launch(...)(cute_tensor)
torch.cuda.synchronize()  # Ensure completion
```

### ❌ DON'T

**Don't use CPU tensors:**
```python
# ✗ BAD
cpu_tensor = torch.randn(1024)
cute.from_dlpack(cpu_tensor)  # Error!
```

**Don't let original tensor die:**
```python
# ✗ BAD
def bad():
    pt = torch.randn(1024, device='cuda')
    return cute.from_dlpack(pt)  # pt deleted!

cute_tensor = bad()  # Dangling pointer!
```

**Don't assume implicit conversion:**
```python
# ✗ BAD (unclear)
kernel.launch(...)(pt_tensor)  # Is this converted?

# ✓ GOOD (explicit)
cute_tensor = cute.from_dlpack(pt_tensor)
kernel.launch(...)(cute_tensor)
```

---

## Summary

**DLPack:**
- Standard format for tensor exchange
- Zero-copy tensor sharing
- Preserves memory layout and device
- Framework agnostic

**Key function:**
- `cute.from_dlpack(tensor)` - Convert to CuTe (zero-copy)

**What's shared:**
- Data pointer (GPU memory)
- Shape and strides
- Data type
- Device information

**Benefits:**
- Instant conversion (no copy)
- Memory efficient
- Works with PyTorch, CuPy, JAX, etc.

**Best practices:**
- Always use CUDA tensors
- Keep original tensor alive
- Validate before conversion
- Synchronize after kernels

**Key insight:** DLPack enables zero-copy interoperability, making CuTe kernels seamlessly integrate with existing framework code.

---

## Next Steps

- [PyTorch Integration](./pytorch_integration.md) - Using CuTe with PyTorch
- [Bypassing DLPack](./bypassing_dlpack.md) - Direct pointer access
- [Kernel Launch](./kernel_launch.md) - Launching CuTe kernels

---

## Further Reading

- [DLPack Specification](https://dmlc.github.io/dlpack/latest/)
- [PyTorch DLPack Support](https://pytorch.org/docs/stable/dlpack.html)
- [Zero-Copy Tensor Exchange](https://data-apis.org/array-api/latest/design_topics/data_interchange.html)