---
topic: "bypassing_dlpack"
difficulty: "advanced"
related_topics: ["dlpack", "pytorch_integration", "tensors", "memory_hierarchy"]
---

# Bypassing DLPack in CuTe DSL

## Overview

While DLPack provides convenient tensor conversion, sometimes you need direct control over memory pointers, layouts, and addressing. CuTe DSL allows you to bypass DLPack and work directly with GPU memory pointers.

**Key concepts:**
- `cute.make_ptr()` - Create pointer from address
- Manual tensor construction
- Custom layouts from raw pointers
- When to bypass DLPack
- Low-level memory control
- Performance optimization

---

## Why Bypass DLPack?

### When DLPack is Sufficient

**Most cases: Use DLPack:**
```python
# ✓ GOOD: Simple and safe
import torch
import cutlass.cute as cute

tensor = torch.randn(1024, device='cuda')
cute_tensor = cute.from_dlpack(tensor)  # Easy!

kernel.launch(...)(cute_tensor)
```

### When to Bypass DLPack

**Use direct pointers when:**

1. **Working with external libraries:**
```python
# CUDA library returns raw pointer
raw_ptr = external_cuda_library.allocate(1024)
# Need to use in CuTe kernel
```

2. **Custom memory management:**
```python
# Pre-allocated memory pool
memory_pool = MemoryPool(size=1024*1024)
ptr = memory_pool.allocate(1024)
# Manually create CuTe tensor
```

3. **Non-standard layouts:**
```python
# Complex strided or swizzled layout
# DLPack may not capture full layout info
```

4. **Performance critical paths:**
```python
# Avoid DLPack overhead (minimal, but exists)
# Direct pointer access
```

5. **Integration with C/C++ code:**
```python
# C++ function returns void* pointer
# Need to wrap in CuTe tensor
```

---

## Getting Raw Pointers

### From PyTorch Tensors

**Extract pointer from PyTorch tensor:**
```python
import torch
import cutlass.cute as cute

# PyTorch tensor
tensor = torch.randn(1024, device='cuda', dtype=torch.float32)

# Get raw pointer (integer address)
ptr = tensor.data_ptr()
print(f"Pointer address: 0x{ptr:x}")

# Get element size
element_size = tensor.element_size()  # 4 bytes for float32

# Get shape and stride
shape = tensor.shape
stride = tensor.stride()

print(f"Shape: {shape}")
print(f"Stride: {stride}")
print(f"Element size: {element_size} bytes")
```

### From CuPy Arrays
```python
import cupy as cp
import cutlass.cute as cute

# CuPy array
array = cp.random.randn(1024)

# Get pointer
ptr = array.data.ptr
print(f"Pointer: 0x{ptr:x}")

# Get dtype info
dtype = array.dtype
itemsize = array.itemsize

print(f"Dtype: {dtype}")
print(f"Item size: {itemsize} bytes")
```

---

## Creating CuTe Tensors from Pointers

### make_ptr() Function

**Create typed pointer from address:**
```python
import torch
import cutlass.cute as cute

# PyTorch tensor
tensor = torch.randn(1024, device='cuda', dtype=torch.float32)
ptr_addr = tensor.data_ptr()

# Create CuTe pointer
cute_ptr = cute.make_ptr(ptr_addr, cutlass.Float32)

# Now create tensor with this pointer
layout = cute.make_layout(cute.make_shape(1024))
cute_tensor = cute.make_tensor(cute_ptr, layout)
```

### With Explicit Layout

**Specify custom layout:**
```python
import torch
import cutlass.cute as cute

# PyTorch matrix (row-major)
matrix = torch.randn(128, 256, device='cuda', dtype=torch.float16)
ptr_addr = matrix.data_ptr()

# Create pointer
cute_ptr = cute.make_ptr(ptr_addr, cutlass.Float16)

# Explicit layout (row-major)
layout = cute.make_layout(
    cute.make_shape(128, 256),
    cute.make_stride(256, 1)  # Row-major: stride in rows is 256
)

cute_tensor = cute.make_tensor(cute_ptr, layout)
```

### Column-Major Layout

**Create column-major tensor from row-major memory:**
```python
import torch
import cutlass.cute as cute

# PyTorch tensor (row-major)
row_major = torch.randn(128, 256, device='cuda')
ptr_addr = row_major.data_ptr()

# Create as column-major view (for specific algorithms)
cute_ptr = cute.make_ptr(ptr_addr, cutlass.Float32)

# Column-major layout
col_major_layout = cute.make_layout(
    cute.make_shape(128, 256),
    cute.make_stride(1, 128)  # Column-major: stride in columns is 128
)

cute_col_major = cute.make_tensor(cute_ptr, col_major_layout)
# NOTE: Data is still row-major in memory!
# This just changes how we INDEX it
```

---

## Manual Tensor Construction

### Step-by-Step Process

**Complete manual tensor creation:**
```python
import torch
import cutlass.cute as cute

# 1. Allocate PyTorch tensor
tensor = torch.empty(1024, device='cuda', dtype=torch.float32)

# 2. Get raw pointer
raw_ptr = tensor.data_ptr()

# 3. Create typed pointer
typed_ptr = cute.make_ptr(raw_ptr, cutlass.Float32)

# 4. Define layout
layout = cute.make_layout(cute.make_shape(1024))

# 5. Create CuTe tensor
cute_tensor = cute.make_tensor(typed_ptr, layout)

# 6. Use in kernel
@cute.kernel
def fill_kernel(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    if tid < cute.size(data):
        data[tid] = 42.0

fill_kernel.launch(grid=[4,1,1], block=[256,1,1])(cute_tensor)
torch.cuda.synchronize()

print(tensor[:10])  # [42, 42, 42, ...]
```

### 2D Tensor Construction
```python
import torch
import cutlass.cute as cute

# Allocate 2D tensor
M, N = 128, 256
tensor_2d = torch.empty(M, N, device='cuda', dtype=torch.float16)

# Get pointer
ptr = tensor_2d.data_ptr()
cute_ptr = cute.make_ptr(ptr, cutlass.Float16)

# 2D layout
layout_2d = cute.make_layout(
    cute.make_shape(M, N),
    cute.make_stride(N, 1)  # Row-major
)

cute_tensor_2d = cute.make_tensor(cute_ptr, layout_2d)

# Use in 2D kernel
@cute.kernel
def fill_2d_kernel(data: cute.Tensor):
    tx = cute.arch.thread_idx()[0]
    ty = cute.arch.thread_idx()[1]
    bx = cute.arch.block_idx()[0]
    by = cute.arch.block_idx()[1]
    
    i = bx * 16 + tx
    j = by * 16 + ty
    
    if i < 128 and j < 256:
        data[i, j] = 1.0

grid = ((M + 15) // 16, (N + 15) // 16, 1)
block = (16, 16, 1)
fill_2d_kernel.launch(grid=grid, block=block)(cute_tensor_2d)
```

---

## Custom Memory Layouts

### Tiled Layout

**Create tiled memory layout:**
```python
import torch
import cutlass.cute as cute

# Allocate memory for tiled layout
# 4×4 tiles, each tile is 32×32
num_tiles_m, num_tiles_n = 4, 4
tile_m, tile_n = 32, 32
total_size = num_tiles_m * tile_m * num_tiles_n * tile_n

tensor = torch.empty(total_size, device='cuda', dtype=torch.float32)
ptr = tensor.data_ptr()
cute_ptr = cute.make_ptr(ptr, cutlass.Float32)

# Tiled layout
tiled_layout = cute.make_layout(
    cute.make_shape(
        cute.make_shape(tile_m, tile_n),      # Tile shape
        cute.make_shape(num_tiles_m, num_tiles_n)  # Num tiles
    ),
    cute.make_stride(
        cute.make_stride(1, tile_m),          # Within tile
        cute.make_stride(tile_m * tile_n, tile_m * tile_n * num_tiles_m)  # Between tiles
    )
)

tiled_tensor = cute.make_tensor(cute_ptr, tiled_layout)
```

### Padded Layout

**Add padding to avoid bank conflicts:**
```python
import torch
import cutlass.cute as cute

# Allocate with padding
M, N = 128, 64
padding = 8  # Extra columns
tensor = torch.empty(M * (N + padding), device='cuda', dtype=torch.float32)

ptr = tensor.data_ptr()
cute_ptr = cute.make_ptr(ptr, cutlass.Float32)

# Layout with padding
padded_layout = cute.make_layout(
    cute.make_shape(M, N),
    cute.make_stride(N + padding, 1)  # Stride includes padding
)

padded_tensor = cute.make_tensor(cute_ptr, padded_layout)

# Accessing [i, j] now automatically skips padding
```

### Swizzled Layout

**Complex swizzled pattern:**
```python
import torch
import cutlass.cute as cute

# Allocate memory
size = 128 * 128
tensor = torch.empty(size, device='cuda', dtype=torch.float16)

ptr = tensor.data_ptr()
cute_ptr = cute.make_ptr(ptr, cutlass.Float16)

# Create swizzled layout (for optimal shared memory access)
swizzle_bits = 3  # 8-way swizzle
swizzled_layout = cute.make_layout(
    cute.make_shape(128, 128),
    cute.make_swizzled_stride(128, 1, swizzle_bits)
)

swizzled_tensor = cute.make_tensor(cute_ptr, swizzled_layout)
```

---

## Working with External Libraries

### CUDA Library Integration

**Use memory from cudaMalloc:**
```python
import ctypes
import cutlass.cute as cute

# Allocate with cudaMalloc (via ctypes)
import cuda.cudart as cudart

size_bytes = 1024 * 4  # 1024 floats
status, ptr = cudart.cudaMalloc(size_bytes)
assert status == cudart.cudaError_t.cudaSuccess

# Create CuTe tensor
cute_ptr = cute.make_ptr(ptr, cutlass.Float32)
layout = cute.make_layout(cute.make_shape(1024))
cute_tensor = cute.make_tensor(cute_ptr, layout)

# Use in kernel
kernel.launch(...)(cute_tensor)

# Clean up
cudart.cudaFree(ptr)
```

### Unified Memory

**Use CUDA unified memory:**
```python
import torch
import cutlass.cute as cute

# Allocate unified memory via PyTorch
tensor = torch.empty(1024, device='cuda', dtype=torch.float32)
tensor = tensor.pin_memory()  # Pin for unified memory

ptr = tensor.data_ptr()
cute_ptr = cute.make_ptr(ptr, cutlass.Float32)

layout = cute.make_layout(cute.make_shape(1024))
cute_tensor = cute.make_tensor(cute_ptr, layout)

# Accessible from both CPU and GPU
```

---

## Memory Safety Considerations

### Pointer Lifetime

**Ensure memory stays valid:**
```python
# ✗ BAD: Pointer becomes invalid
def bad_example():
    tensor = torch.randn(1024, device='cuda')
    ptr = tensor.data_ptr()
    # tensor deleted here!
    return cute.make_ptr(ptr, cutlass.Float32)

cute_ptr = bad_example()  # Dangling pointer!

# ✓ GOOD: Keep tensor alive
class SafeWrapper:
    def __init__(self):
        self.tensor = torch.randn(1024, device='cuda')
        self.ptr = self.tensor.data_ptr()
        self.cute_ptr = cute.make_ptr(self.ptr, cutlass.Float32)
```

### Type Safety

**Match pointer type to actual data:**
```python
import torch
import cutlass.cute as cute

# Float32 tensor
tensor = torch.randn(1024, device='cuda', dtype=torch.float32)
ptr = tensor.data_ptr()

# ✓ GOOD: Correct type
cute_ptr = cute.make_ptr(ptr, cutlass.Float32)

# ✗ BAD: Wrong type (undefined behavior!)
# cute_ptr = cute.make_ptr(ptr, cutlass.Float16)  # Wrong!
```

### Bounds Checking

**Ensure layout matches allocation:**
```python
import torch
import cutlass.cute as cute

# Allocate 1024 elements
tensor = torch.empty(1024, device='cuda')
ptr = tensor.data_ptr()
cute_ptr = cute.make_ptr(ptr, cutlass.Float32)

# ✓ GOOD: Layout matches allocation
good_layout = cute.make_layout(cute.make_shape(1024))

# ✗ BAD: Layout larger than allocation
# bad_layout = cute.make_layout(cute.make_shape(2048))  # Out of bounds!
```

---

## Performance Considerations

### DLPack vs Direct Pointer

**Benchmark:**
```python
import torch
import time
import cutlass.cute as cute

tensor = torch.randn(1024, device='cuda')

# DLPack method
start = time.time()
for _ in range(10000):
    cute_tensor = cute.from_dlpack(tensor)
dlpack_time = time.time() - start

# Direct pointer method
ptr = tensor.data_ptr()
layout = cute.make_layout(cute.make_shape(1024))

start = time.time()
for _ in range(10000):
    cute_ptr = cute.make_ptr(ptr, cutlass.Float32)
    cute_tensor = cute.make_tensor(cute_ptr, layout)
direct_time = time.time() - start

print(f"DLPack: {dlpack_time*1000:.3f} ms")
print(f"Direct:  {direct_time*1000:.3f} ms")
print(f"Speedup: {dlpack_time/direct_time:.2f}×")

# Typical: Direct is slightly faster (5-10%)
# But difference is negligible for most use cases
```

### When Direct Pointers Help

**Cache pointer and layout:**
```python
class OptimizedKernel:
    def __init__(self, tensor: torch.Tensor):
        # Cache pointer and layout (setup once)
        self.tensor = tensor
        self.ptr = tensor.data_ptr()
        self.cute_ptr = cute.make_ptr(self.ptr, cutlass.Float32)
        self.layout = cute.make_layout(cute.make_shape(tensor.shape[0]))
        self.cute_tensor = cute.make_tensor(self.cute_ptr, self.layout)
    
    def run(self):
        # Reuse cached tensor (no conversion overhead)
        kernel.launch(...)(self.cute_tensor)

# One-time setup
opt = OptimizedKernel(tensor)

# Fast repeated calls
for _ in range(1000):
    opt.run()  # No conversion overhead
```

---

## Common Patterns

### Pattern 1: Memory Pool

**Reuse preallocated memory:**
```python
import torch
import cutlass.cute as cute

class MemoryPool:
    def __init__(self, size: int, dtype):
        self.tensor = torch.empty(size, device='cuda', dtype=dtype)
        self.ptr = self.tensor.data_ptr()
        self.size = size
        self.dtype = dtype
        self.offset = 0
    
    def allocate(self, n: int):
        """Allocate n elements from pool"""
        if self.offset + n > self.size:
            raise MemoryError("Pool exhausted")
        
        # Calculate pointer offset
        element_size = self.tensor.element_size()
        ptr_offset = self.ptr + (self.offset * element_size)
        
        # Create CuTe tensor for this slice
        cute_ptr = cute.make_ptr(ptr_offset, cutlass.Float32)
        layout = cute.make_layout(cute.make_shape(n))
        cute_tensor = cute.make_tensor(cute_ptr, layout)
        
        self.offset += n
        return cute_tensor
    
    def reset(self):
        """Reset pool"""
        self.offset = 0

# Usage
pool = MemoryPool(10000, torch.float32)

tensor1 = pool.allocate(1024)
tensor2 = pool.allocate(2048)
tensor3 = pool.allocate(512)

# Use tensors
kernel.launch(...)(tensor1)
kernel.launch(...)(tensor2)

# Reset and reuse
pool.reset()
```

### Pattern 2: Slice Views

**Create views into larger tensor:**
```python
import torch
import cutlass.cute as cute

# Large tensor
large_tensor = torch.randn(10000, device='cuda', dtype=torch.float32)
base_ptr = large_tensor.data_ptr()
element_size = large_tensor.element_size()

def get_slice_view(start: int, length: int):
    """Get CuTe view of slice [start:start+length]"""
    
    # Calculate offset pointer
    offset_ptr = base_ptr + (start * element_size)
    cute_ptr = cute.make_ptr(offset_ptr, cutlass.Float32)
    
    # Create layout for slice
    layout = cute.make_layout(cute.make_shape(length))
    return cute.make_tensor(cute_ptr, layout)

# Create views
view1 = get_slice_view(0, 1024)      # [0:1024]
view2 = get_slice_view(1024, 2048)   # [1024:3072]
view3 = get_slice_view(5000, 512)    # [5000:5512]

# Process each view
kernel.launch(...)(view1)
kernel.launch(...)(view2)
kernel.launch(...)(view3)
```

### Pattern 3: Interleaved Layouts

**Create custom interleaved memory layout:**
```python
import torch
import cutlass.cute as cute

# Interleaved RGB data: RGBRGBRGB...
height, width = 224, 224
channels = 3
total_size = height * width * channels

tensor = torch.empty(total_size, device='cuda', dtype=torch.float32)
ptr = tensor.data_ptr()
cute_ptr = cute.make_ptr(ptr, cutlass.Float32)

# Interleaved layout
interleaved_layout = cute.make_layout(
    cute.make_shape(height, width, channels),
    cute.make_stride(width * channels, channels, 1)
)

interleaved_tensor = cute.make_tensor(cute_ptr, interleaved_layout)

# Now accessing [h, w, c] correctly handles interleaving
```

---

## Debugging Direct Pointers

### Pointer Validation
```python
import torch
import cutlass.cute as cute

def validate_pointer(ptr: int, expected_size: int):
    """Validate pointer is reasonable"""
    
    # Check not null
    if ptr == 0:
        raise ValueError("Null pointer")
    
    # Check alignment (should be aligned to element size)
    if ptr % 4 != 0:  # Assuming 4-byte elements
        print(f"Warning: Pointer not aligned: 0x{ptr:x}")
    
    # Check in reasonable range (between 0 and max GPU memory)
    if ptr > 2**48:  # Modern GPUs use 48-bit addressing
        print(f"Warning: Suspicious pointer value: 0x{ptr:x}")
    
    print(f"Pointer validated: 0x{ptr:x}")
    return True
```

### Memory Dump
```python
import torch

def dump_memory(tensor: torch.Tensor, n: int = 16):
    """Dump first n elements of tensor"""
    print(f"Memory dump (first {n} elements):")
    print(f"Pointer: 0x{tensor.data_ptr():x}")
    print(f"Values: {tensor.flatten()[:n].tolist()}")
```

---

## Best Practices

### ✅ DO

**Keep memory owner alive:**
```python
class Wrapper:
    def __init__(self):
        self.tensor = torch.randn(1024, device='cuda')
        self.cute_tensor = self.create_cute_tensor()
    
    def create_cute_tensor(self):
        ptr = self.tensor.data_ptr()
        # ...
        return cute_tensor
```

**Match types correctly:**
```python
# Float32 tensor → Float32 pointer
tensor_f32 = torch.randn(1024, device='cuda', dtype=torch.float32)
cute_ptr = cute.make_ptr(tensor_f32.data_ptr(), cutlass.Float32)
```

**Validate pointer addresses:**
```python
ptr = tensor.data_ptr()
assert ptr != 0, "Null pointer"
```

**Document layout assumptions:**
```python
# Layout assumes row-major, stride = N
layout = cute.make_layout(
    cute.make_shape(M, N),
    cute.make_stride(N, 1)  # Row-major
)
```

### ❌ DON'T

**Don't use after free:**
```python
# ✗ BAD
def bad():
    tensor = torch.randn(1024, device='cuda')
    return tensor.data_ptr()  # tensor freed!

ptr = bad()  # Dangling pointer
```

**Don't mismatch types:**
```python
# ✗ BAD
fp16_tensor = torch.randn(1024, device='cuda', dtype=torch.float16)
ptr = fp16_tensor.data_ptr()
cute_ptr = cute.make_ptr(ptr, cutlass.Float32)  # Wrong type!
```

**Don't exceed bounds:**
```python
# ✗ BAD
tensor = torch.empty(1024, device='cuda')
layout = cute.make_layout(cute.make_shape(2048))  # Too large!
```

---

## Summary

**Bypassing DLPack:**
- Use `cute.make_ptr()` for direct pointer access
- Manually construct tensors with `cute.make_tensor()`
- Full control over memory layout
- Necessary for external libraries and custom memory

**Key functions:**
- `cute.make_ptr(address, dtype)` - Create typed pointer
- `cute.make_tensor(ptr, layout)` - Create tensor from pointer
- `tensor.data_ptr()` - Get PyTorch pointer

**When to use:**
- External library integration
- Custom memory management
- Non-standard layouts
- Performance optimization
- Memory pooling

**Safety considerations:**
- Keep memory owner alive
- Match pointer types to data
- Validate pointer addresses
- Ensure layouts fit allocations

**Key insight:** Direct pointer access provides maximum flexibility but requires careful memory management. Use DLPack when possible, bypass when necessary.

---

## Next Steps

- [PyTorch Integration](./pytorch_integration.md) - Higher-level integration
- [DLPack](./dlpack.md) - Standard conversion
- [Memory Hierarchy](../03_memory_and_layouts/memory_hierarchy.md) - Memory management

---

## Further Reading

- [CUDA Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-management)
- [Pointer Arithmetic](https://en.wikipedia.org/wiki/Pointer_(computer_programming))
- [Memory Alignment](https://en.wikipedia.org/wiki/Data_structure_alignment)