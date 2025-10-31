---
topic: "pytorch_integration"
difficulty: "beginner"
related_topics: ["dlpack", "jit_compilation", "kernel_launch"]
---

# PyTorch Integration with CuTe DSL

## Overview

CuTe DSL seamlessly integrates with PyTorch, allowing you to write custom CUDA kernels that work directly with PyTorch tensors. This enables you to extend PyTorch with optimized GPU operations while maintaining a familiar Python workflow.

**Key concepts:**
- Converting PyTorch tensors to CuTe tensors
- Launching kernels from PyTorch
- Memory management and ownership
- Integration with PyTorch autograd
- Mixed PyTorch/CuTe workflows

---

## Basic Workflow

### Simple PyTorch to CuTe Pipeline

**Complete example:**
```python
import torch
import cutlass.cute as cute

# 1. Define CuTe kernel
@cute.kernel
def scale_kernel(data: cute.Tensor, scale: cutlass.Float32):
    """Scale tensor elements by a constant"""
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    idx = bid * 256 + tid
    if idx < cute.size(data):
        data[idx] = data[idx] * scale

# 2. Create PyTorch tensor
tensor = torch.randn(1024, device='cuda', dtype=torch.float32)
print(f"Before: {tensor[:5]}")

# 3. Convert to CuTe tensor
cute_tensor = cute.from_dlpack(tensor)

# 4. Launch kernel
grid = ((1024 + 255) // 256, 1, 1)
block = (256, 1, 1)
scale_kernel.launch(grid=grid, block=block)(cute_tensor, scale=2.0)

# 5. Results automatically visible in PyTorch tensor
torch.cuda.synchronize()
print(f"After:  {tensor[:5]}")  # Scaled by 2.0!
```

**Key insight:** Changes to `cute_tensor` are immediately visible in the original PyTorch `tensor` (they share memory).

---

## Creating PyTorch Tensors for CuTe

### CUDA Tensors

**Always use CUDA tensors:**
```python
# ✓ GOOD: CUDA tensor
cuda_tensor = torch.randn(1024, device='cuda')
cute_tensor = cute.from_dlpack(cuda_tensor)

# ✗ BAD: CPU tensor
cpu_tensor = torch.randn(1024, device='cpu')
cute_tensor = cute.from_dlpack(cpu_tensor)  # Error!
```

### Supported Data Types

**PyTorch dtype → CuTe type mapping:**
```python
# Float types
fp32 = torch.randn(1024, device='cuda', dtype=torch.float32)
fp16 = torch.randn(1024, device='cuda', dtype=torch.float16)
bf16 = torch.randn(1024, device='cuda', dtype=torch.bfloat16)

# Integer types
int32 = torch.zeros(1024, device='cuda', dtype=torch.int32)
int8 = torch.zeros(1024, device='cuda', dtype=torch.int8)

# All can be converted to CuTe tensors
cute.from_dlpack(fp32)   # Float32
cute.from_dlpack(fp16)   # Float16
cute.from_dlpack(bf16)   # BFloat16
cute.from_dlpack(int32)  # Int32
cute.from_dlpack(int8)   # Int8
```

**Type compatibility:**

| PyTorch dtype | CuTe type | Notes |
|---------------|-----------|-------|
| `torch.float32` | `cutlass.Float32` | Most common |
| `torch.float16` | `cutlass.Float16` | Half precision |
| `torch.bfloat16` | `cutlass.BFloat16` | Brain float |
| `torch.float64` | `cutlass.Float64` | Double precision |
| `torch.int32` | `cutlass.Int32` | Signed 32-bit |
| `torch.int16` | `cutlass.Int16` | Signed 16-bit |
| `torch.int8` | `cutlass.Int8` | Signed 8-bit |
| `torch.uint8` | `cutlass.UInt8` | Unsigned 8-bit |
| `torch.bool` | `cutlass.Bool` | Boolean |

### Tensor Shapes

**Multi-dimensional tensors:**
```python
# 1D tensor
vec = torch.randn(1024, device='cuda')
cute.from_dlpack(vec)

# 2D tensor (matrix)
mat = torch.randn(128, 256, device='cuda')
cute.from_dlpack(mat)

# 3D tensor
tensor_3d = torch.randn(32, 64, 128, device='cuda')
cute.from_dlpack(tensor_3d)

# 4D tensor (batch of images)
images = torch.randn(16, 3, 224, 224, device='cuda')  # [B, C, H, W]
cute.from_dlpack(images)
```

---

## Memory Sharing

### Zero-Copy Conversion

**`from_dlpack()` creates a view (no copy):**
```python
# PyTorch tensor
pt_tensor = torch.randn(1024, device='cuda')

# CuTe tensor (shares memory)
cute_tensor = cute.from_dlpack(pt_tensor)

# Modify via CuTe kernel
@cute.kernel
def modify_kernel(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    data[tid] = 999.0

modify_kernel.launch(grid=[1,1,1], block=[1024,1,1])(cute_tensor)
torch.cuda.synchronize()

# Changes visible in PyTorch tensor!
print(pt_tensor[0])  # 999.0
```

**No copy means:**
- ✅ Very fast (microseconds)
- ✅ Memory efficient
- ⚠️ Modifications affect both tensors
- ⚠️ Lifetime management important

### Memory Ownership

**PyTorch owns the memory:**
```python
# PyTorch tensor owns memory
pt_tensor = torch.randn(1024, device='cuda')
cute_tensor = cute.from_dlpack(pt_tensor)

# Don't delete PyTorch tensor while CuTe tensor exists!
# del pt_tensor  # Dangerous if cute_tensor still in use!

# Safe pattern:
kernel.launch(...)(cute_tensor)
torch.cuda.synchronize()
# Now safe to delete pt_tensor
```

---

## Launching Kernels from PyTorch

### Helper Function Pattern

**Wrap kernel launch in Python function:**
```python
import torch
import cutlass.cute as cute

@cute.kernel
def elementwise_add_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor
):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    idx = bid * 256 + tid
    if idx < cute.size(c):
        c[idx] = a[idx] + b[idx]

def elementwise_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two PyTorch tensors using custom CuTe kernel.
    
    Args:
        a: First tensor (CUDA)
        b: Second tensor (CUDA)
    
    Returns:
        Result tensor (CUDA)
    """
    # Validate inputs
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    assert a.shape == b.shape, "Shapes must match"
    assert a.dtype == b.dtype, "Dtypes must match"
    
    # Allocate output
    c = torch.empty_like(a)
    
    # Convert to CuTe tensors
    cute_a = cute.from_dlpack(a)
    cute_b = cute.from_dlpack(b)
    cute_c = cute.from_dlpack(c)
    
    # Launch kernel
    n = a.numel()
    grid = ((n + 255) // 256, 1, 1)
    block = (256, 1, 1)
    
    elementwise_add_kernel.launch(grid=grid, block=block)(
        cute_a, cute_b, cute_c
    )
    
    return c

# Usage: Just like PyTorch
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
c = elementwise_add(a, b)
```

### In-Place Operations

**Modify tensor in-place:**
```python
@cute.kernel
def inplace_relu_kernel(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    idx = bid * 256 + tid
    if idx < cute.size(data):
        val = data[idx]
        if val < 0.0:
            data[idx] = 0.0

def relu_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place ReLU using CuTe kernel"""
    assert tensor.is_cuda, "Tensor must be on CUDA"
    
    cute_tensor = cute.from_dlpack(tensor)
    
    n = tensor.numel()
    grid = ((n + 255) // 256, 1, 1)
    block = (256, 1, 1)
    
    inplace_relu_kernel.launch(grid=grid, block=block)(cute_tensor)
    
    return tensor  # Modified in-place

# Usage
x = torch.randn(1024, device='cuda')
relu_(x)  # x modified in-place
```

---

## Working with Different Tensor Layouts

### Contiguous Tensors

**CuTe works best with contiguous tensors:**
```python
# Contiguous tensor
cont = torch.randn(128, 256, device='cuda')
print(cont.is_contiguous())  # True

cute_tensor = cute.from_dlpack(cont)  # ✓ Efficient

# Non-contiguous tensor (transpose)
non_cont = cont.t()  # Transpose
print(non_cont.is_contiguous())  # False

# Option 1: Make contiguous
cont_again = non_cont.contiguous()
cute_tensor = cute.from_dlpack(cont_again)  # ✓ OK

# Option 2: Work with non-contiguous (if supported)
cute_tensor = cute.from_dlpack(non_cont)  # May work, but less efficient
```

### Strided Tensors

**Handle strides correctly:**
```python
# Row-major (C-style, PyTorch default)
row_major = torch.randn(128, 256, device='cuda')
print(row_major.stride())  # (256, 1)

# Column-major (Fortran-style)
col_major = torch.randn(256, 128, device='cuda').t().contiguous()
print(col_major.stride())  # (1, 128)

# Both can be converted, but kernel must handle strides
cute.from_dlpack(row_major)
cute.from_dlpack(col_major)
```

---

## Integration with PyTorch Operations

### Mixed PyTorch and CuTe

**Combine PyTorch ops with custom kernels:**
```python
import torch
import torch.nn.functional as F

def custom_layer(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Custom layer mixing PyTorch and CuTe"""
    
    # PyTorch operation
    x = F.relu(x)
    
    # Custom CuTe kernel
    x = my_custom_kernel(x, weight)
    
    # PyTorch operation
    x = F.layer_norm(x, x.shape)
    
    return x
```

### Before and After PyTorch Ops
```python
@cute.kernel
def custom_activation_kernel(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    idx = bid * 256 + tid
    if idx < cute.size(data):
        val = data[idx]
        # Custom activation: swish-like
        data[idx] = val / (1.0 + cute.exp(-val))

def forward_pass(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # PyTorch linear
    x = torch.matmul(x, weight)
    
    # Custom CuTe activation
    cute_x = cute.from_dlpack(x)
    n = x.numel()
    grid = ((n + 255) // 256, 1, 1)
    block = (256, 1, 1)
    custom_activation_kernel.launch(grid=grid, block=block)(cute_x)
    
    # Continue with PyTorch
    x = F.dropout(x, p=0.1)
    
    return x
```

---

## Synchronization with PyTorch

### Explicit Synchronization

**Ensure kernel completes before PyTorch access:**
```python
# Launch CuTe kernel
my_kernel.launch(...)(cute_tensor)

# IMPORTANT: Synchronize before accessing result in PyTorch
torch.cuda.synchronize()

# Now safe to use tensor
print(tensor.sum())  # Correct result
```

### Implicit Synchronization

**PyTorch operations implicitly synchronize:**
```python
# Launch CuTe kernel
my_kernel.launch(...)(cute_tensor)

# PyTorch operation implicitly waits
result = tensor.sum()  # PyTorch synchronizes automatically

# No explicit torch.cuda.synchronize() needed
```

**But explicit is safer:**
```python
# Recommended pattern
my_kernel.launch(...)(cute_tensor)
torch.cuda.synchronize()  # Explicit and clear
result = tensor.sum()
```

---

## PyTorch Autograd Integration

### Non-Differentiable Kernels

**Simple kernels without gradients:**
```python
@cute.kernel
def scale_kernel(data: cute.Tensor, scale: cutlass.Float32):
    tid = cute.arch.thread_idx()[0]
    data[tid] = data[tid] * scale

def scale_no_grad(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale without gradient tracking"""
    with torch.no_grad():
        cute_x = cute.from_dlpack(x)
        scale_kernel.launch(...)(cute_x, scale=scale)
    return x
```

### Custom Autograd Function

**Add gradient support:**
```python
import torch
from torch.autograd import Function

class CustomReLU(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        # Forward pass with CuTe kernel
        output = input.clone()
        cute_output = cute.from_dlpack(output)
        
        relu_forward_kernel.launch(...)(cute_output)
        torch.cuda.synchronize()
        
        # Save for backward
        ctx.save_for_backward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Backward pass with CuTe kernel
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        cute_grad = cute.from_dlpack(grad_input)
        cute_input = cute.from_dlpack(input)
        
        relu_backward_kernel.launch(...)(cute_grad, cute_input)
        torch.cuda.synchronize()
        
        return grad_input

# Usage in PyTorch
custom_relu = CustomReLU.apply

x = torch.randn(1024, device='cuda', requires_grad=True)
y = custom_relu(x)
loss = y.sum()
loss.backward()  # Gradients computed via custom kernel!
```

---

## Multi-GPU Support

### Per-Device Kernels

**Launch kernels on specific GPUs:**
```python
import torch

# GPU 0
with torch.cuda.device(0):
    tensor_0 = torch.randn(1024, device='cuda:0')
    cute_tensor_0 = cute.from_dlpack(tensor_0)
    my_kernel.launch(...)(cute_tensor_0)

# GPU 1
with torch.cuda.device(1):
    tensor_1 = torch.randn(1024, device='cuda:1')
    cute_tensor_1 = cute.from_dlpack(tensor_1)
    my_kernel.launch(...)(cute_tensor_1)
```

### Data Parallel Pattern
```python
def process_on_all_gpus(data_list: list[torch.Tensor]):
    """Process tensors on multiple GPUs in parallel"""
    
    for gpu_id, tensor in enumerate(data_list):
        with torch.cuda.device(gpu_id):
            cute_tensor = cute.from_dlpack(tensor)
            my_kernel.launch(...)(cute_tensor)
    
    # Synchronize all GPUs
    for gpu_id in range(len(data_list)):
        with torch.cuda.device(gpu_id):
            torch.cuda.synchronize()
```

---

## Performance Considerations

### Memory Transfer Overhead

**Minimize CPU-GPU transfers:**
```python
# ✗ BAD: Repeated transfers
for i in range(1000):
    cpu_data = torch.randn(1024)  # CPU
    gpu_data = cpu_data.cuda()     # Transfer
    process_kernel(cute.from_dlpack(gpu_data))

# ✓ GOOD: Allocate once on GPU
gpu_data = torch.randn(1024, device='cuda')
for i in range(1000):
    # Fill with new random data on GPU
    torch.randn(1024, device='cuda', out=gpu_data)
    process_kernel(cute.from_dlpack(gpu_data))
```

### Synchronization Overhead

**Batch operations when possible:**
```python
# ✗ BAD: Sync after each kernel
for i in range(100):
    kernel.launch(...)(data)
    torch.cuda.synchronize()  # 100 syncs!

# ✓ GOOD: Single sync at end
for i in range(100):
    kernel.launch(...)(data)
torch.cuda.synchronize()  # 1 sync
```

### Tensor Contiguity

**Ensure contiguous memory:**
```python
# Check and fix contiguity
if not tensor.is_contiguous():
    tensor = tensor.contiguous()  # Create contiguous copy

cute_tensor = cute.from_dlpack(tensor)
```

---

## Common Patterns

### Pattern 1: Custom Layer in nn.Module
```python
import torch.nn as nn

class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use CuTe kernel for convolution
        output = torch.empty(
            x.shape[0], self.weight.shape[0], 
            x.shape[2], x.shape[3],
            device=x.device, dtype=x.dtype
        )
        
        cute_x = cute.from_dlpack(x)
        cute_weight = cute.from_dlpack(self.weight)
        cute_output = cute.from_dlpack(output)
        
        conv2d_kernel.launch(...)(cute_x, cute_weight, cute_output)
        torch.cuda.synchronize()
        
        return output
```

### Pattern 2: Preprocessing Pipeline
```python
def preprocess_pipeline(images: torch.Tensor) -> torch.Tensor:
    """Custom preprocessing with CuTe kernels"""
    
    # Normalize (CuTe kernel)
    cute_images = cute.from_dlpack(images)
    normalize_kernel.launch(...)(cute_images, mean=0.5, std=0.5)
    
    # Random crop (PyTorch)
    images = F.random_crop(images, size=224)
    
    # Color jitter (CuTe kernel)
    cute_images = cute.from_dlpack(images)
    color_jitter_kernel.launch(...)(cute_images)
    
    torch.cuda.synchronize()
    return images
```

### Pattern 3: Custom Loss Function
```python
class CustomLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute loss with CuTe kernel
        loss = torch.zeros(1, device=pred.device)
        
        cute_pred = cute.from_dlpack(pred)
        cute_target = cute.from_dlpack(target)
        cute_loss = cute.from_dlpack(loss)
        
        custom_loss_kernel.launch(...)(
            cute_pred, cute_target, cute_loss
        )
        
        torch.cuda.synchronize()
        return loss
```

---

## Debugging PyTorch Integration

### Checking Tensor Properties
```python
def debug_tensor(tensor: torch.Tensor, name: str = "tensor"):
    """Print debugging info about PyTorch tensor"""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Contiguous: {tensor.is_contiguous()}")
    print(f"  Stride: {tensor.stride()}")
    print(f"  Requires grad: {tensor.requires_grad}")
    print(f"  Data ptr: {tensor.data_ptr()}")
```

### Validating Kernel Results
```python
def test_kernel_vs_pytorch():
    """Compare CuTe kernel with PyTorch reference"""
    
    # Input
    x = torch.randn(1024, device='cuda')
    
    # PyTorch reference
    reference = torch.relu(x)
    
    # CuTe kernel
    result = x.clone()
    cute_result = cute.from_dlpack(result)
    relu_kernel.launch(...)(cute_result)
    torch.cuda.synchronize()
    
    # Compare
    diff = (result - reference).abs().max()
    print(f"Max difference: {diff.item()}")
    assert diff < 1e-5, "Results don't match!"
```

---

## Best Practices

### ✅ DO

**Always use CUDA tensors:**
```python
tensor = torch.randn(1024, device='cuda')
```

**Synchronize when needed:**
```python
kernel.launch(...)(cute_tensor)
torch.cuda.synchronize()
```

**Check contiguity:**
```python
if not tensor.is_contiguous():
    tensor = tensor.contiguous()
```

**Wrap kernels in functions:**
```python
def my_operation(x: torch.Tensor) -> torch.Tensor:
    # Clean PyTorch-style API
    cute_x = cute.from_dlpack(x)
    kernel.launch(...)(cute_x)
    return x
```

### ❌ DON'T

**Don't use CPU tensors:**
```python
# ✗ BAD
cpu_tensor = torch.randn(1024, device='cpu')
cute.from_dlpack(cpu_tensor)  # Error!
```

**Don't forget synchronization:**
```python
# ✗ BAD
kernel.launch(...)(cute_tensor)
print(tensor.sum())  # May be wrong!
```

**Don't mix different dtypes:**
```python
# ✗ BAD
fp32_tensor = torch.randn(1024, dtype=torch.float32, device='cuda')
fp16_kernel.launch(...)(cute.from_dlpack(fp32_tensor))  # Type mismatch!
```

---

## Summary

**PyTorch integration:**
- Use `cute.from_dlpack()` to convert tensors (zero-copy)
- CuTe and PyTorch tensors share memory
- Always use CUDA tensors
- Synchronize when needed

**Workflow:**
1. Create PyTorch tensor on CUDA
2. Convert to CuTe tensor
3. Launch kernel
4. Synchronize
5. Use result in PyTorch

**Key functions:**
- `cute.from_dlpack(torch_tensor)` - Convert to CuTe
- `torch.cuda.synchronize()` - Wait for kernels
- `tensor.is_cuda` - Check device
- `tensor.is_contiguous()` - Check layout

**Performance tips:**
- Minimize CPU-GPU transfers
- Batch synchronizations
- Use contiguous tensors
- Avoid unnecessary copies

---

## Next Steps

- [DLPack](./dlpack.md) - How from_dlpack works
- [Bypassing DLPack](./bypassing_dlpack.md) - Direct pointer access
- [Autotuning](./autotuning.md) - Optimize kernel parameters

---

## Further Reading

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [DLPack Specification](https://dmlc.github.io/dlpack/latest/)
- [PyTorch Custom Ops](https://pytorch.org/tutorials/advanced/cpp_extension.html)