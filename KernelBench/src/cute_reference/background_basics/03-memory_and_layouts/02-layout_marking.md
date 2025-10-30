---
topic: "layout_marking"
difficulty: "intermediate"
related_topics: ["static_vs_dynamic", "tensors", "layouts"]
---

# Layout Marking in CuTe DSL

## Overview

Layout marking allows you to explicitly control whether CuTe DSL treats a tensor's layout as **static** (known at compile time) or **dynamic** (known at runtime). This gives you fine-grained control over the compile-time vs runtime trade-off.

**Key functions:**
- `mark_layout_dynamic()` - Treat layout as runtime-determined
- `mark_layout_static()` - Treat layout as compile-time constant
- `mark_compact_shape_dynamic()` - Special case for compact layouts

---

## Why Mark Layouts?

### The Problem

When you create a tensor from a framework (PyTorch, JAX), CuTe DSL must decide:
- Should it specialize the kernel for this specific shape?
- Or compile a general kernel that works for any shape?

**Without explicit marking:**
- CuTe DSL makes a default choice (usually static for small tensors)
- May not match your needs
- Can lead to excessive recompilation or suboptimal performance

**With explicit marking:**
- You control the trade-off
- Compile once for many sizes (dynamic)
- Or specialize for maximum performance (static)

---

## mark_layout_dynamic()

### Purpose

Tell CuTe DSL: "Don't specialize for this specific shape. Compile a kernel that works for any shape."

### Usage
```
import torch
from cutlass.cute.runtime import from_dlpack

# PyTorch tensor
torch_tensor = torch.randn(1024, 512, device="cuda")

# Convert to CuTe tensor
cute_tensor = from_dlpack(torch_tensor)

# Mark layout as dynamic
dynamic_tensor = cute_tensor.mark_layout_dynamic()

# Now kernel won't specialize for (1024, 512)
process_kernel.launch(...)(dynamic_tensor)
```

### What Happens

**Without mark_layout_dynamic():**
```
@cute.kernel
def process(data: cute.Tensor):
    # CuTe may specialize for shape (1024, 512)
    n = cute.size(data)
    for i in range(n):
        process(data[i])

# Compiled kernel is specialized for 1024×512
process.launch(...)(tensor_1024_512)

# Different size requires recompilation!
process.launch(...)(tensor_2048_1024)  # Recompiles!
```

**With mark_layout_dynamic():**
```
@cute.kernel
def process(data: cute.Tensor):
    n = cute.size(data)
    for i in range(n):
        process(data[i])

# Compiled kernel works for ANY size
tensor_a = from_dlpack(torch_a).mark_layout_dynamic()
tensor_b = from_dlpack(torch_b).mark_layout_dynamic()

process.launch(...)(tensor_a)  # Uses same kernel
process.launch(...)(tensor_b)  # No recompilation!
```

### When to Use

✅ **Use mark_layout_dynamic() when:**

**1. Many different input sizes:**
```
@cute.jit
def flexible_processing(tensors: list):
    for tensor in tensors:
        # Each tensor has different shape
        dynamic_tensor = from_dlpack(tensor).mark_layout_dynamic()
        process(dynamic_tensor)
    
    # Single kernel handles all sizes!
```

**2. Compilation time is a bottleneck:**
```
# Without dynamic: 100 sizes = 100 kernel compilations (slow!)
# With dynamic: 1 compilation for all sizes (fast!)

for size in range(100, 200):
    tensor = torch.randn(size, size, device="cuda")
    cute_tensor = from_dlpack(tensor).mark_layout_dynamic()
    kernel.launch(...)(cute_tensor)  # Same kernel every time
```

**3. Size is user-controlled:**
```
@cute.jit
def user_input_kernel(user_tensor):
    # User provides arbitrary size
    # Don't want to compile for every possible size
    dynamic_tensor = from_dlpack(user_tensor).mark_layout_dynamic()
    process(dynamic_tensor)
```

**4. Prototyping and development:**
```
# During development, dynamic is more convenient
# Can optimize to static later if profiling shows benefit
```

---

## mark_layout_static()

### Purpose

Tell CuTe DSL: "Specialize for this exact shape. Generate the fastest possible kernel for this specific size."

### Usage
```
import torch
from cutlass.cute.runtime import from_dlpack

# PyTorch tensor
torch_tensor = torch.randn(1024, 512, device="cuda")

# Convert and mark as static
cute_tensor = from_dlpack(torch_tensor)
static_tensor = cute_tensor.mark_layout_static()

# Kernel will specialize for exactly (1024, 512)
process_kernel.launch(...)(static_tensor)
```

### What Happens

**Compiler can:**
- Constant propagate shape values
- Unroll loops with known bounds
- Optimize address calculations
- Eliminate runtime checks

**Example:**
```
@cute.kernel
def process_static(data: cute.Tensor):
    # With static marking, compiler knows exact size
    m, n = cute.size(data, mode=0), cute.size(data, mode=1)
    # Compiler sees: m = 1024, n = 512
    
    # Can optimize this loop aggressively
    for i in range(m):  # Compiler knows: 1024 iterations
        for j in range(n):  # Compiler knows: 512 iterations
            process(data[i, j])
```

### When to Use

✅ **Use mark_layout_static() when:**

**1. Fixed input sizes:**
```
# Always process 1024×1024 matrices
MATRIX_SIZE = 1024

@cute.jit
def process_fixed_size():
    tensor = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device="cuda")
    static_tensor = from_dlpack(tensor).mark_layout_static()
    
    # Highly optimized kernel for 1024×1024
    process_kernel.launch(...)(static_tensor)
```

**2. Performance-critical kernels:**
```
# Every cycle matters - specialize!
@cute.kernel
def critical_kernel(data: cute.Tensor):
    # mark_layout_static ensures best performance
    static_data = data.mark_layout_static()
    process(static_data)
```

**3. Small number of size variants:**
```
# Only 3 sizes - compile specialized kernels for each
common_sizes = [512, 1024, 2048]

for size in common_sizes:
    tensor = torch.randn(size, size, device="cuda")
    static_tensor = from_dlpack(tensor).mark_layout_static()
    
    # 3 specialized kernels total
    optimized_kernel.launch(...)(static_tensor)
```

**4. Tensor core operations:**
```
# MMA operations benefit from static shapes
@cute.kernel
def gemm_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    # Mark inputs as static for best MMA performance
    static_A = A.mark_layout_static()
    static_B = B.mark_layout_static()
    static_C = C.mark_layout_static()
    
    cute.gemm(tiled_mma, static_C, static_A, static_B, static_C)
```

---

## mark_compact_shape_dynamic()

### Purpose

Special case: Mark **only the shape** as dynamic while keeping the layout structure static.

**Use when:**
- Layout structure is fixed (e.g., always column-major)
- But dimensions vary (e.g., different M, N values)

### Usage
```
import torch
from cutlass.cute.runtime import from_dlpack

# Different sized tensors, but all column-major
tensor_a = torch.randn(512, 256, device="cuda").t()  # Column-major
tensor_b = torch.randn(1024, 512, device="cuda").t()  # Column-major

# Mark shape as dynamic, but layout structure is known
cute_a = from_dlpack(tensor_a).mark_compact_shape_dynamic()
cute_b = from_dlpack(tensor_b).mark_compact_shape_dynamic()

# Same kernel handles both (knows column-major, but not exact size)
process_kernel.launch(...)(cute_a)
process_kernel.launch(...)(cute_b)
```

### What's Different?

**mark_layout_dynamic():**
- Shape AND stride are dynamic
- Most flexible
- Slightly slower

**mark_compact_shape_dynamic():**
- Only shape is dynamic
- Stride pattern is known (compact/contiguous)
- Faster than full dynamic
- More optimizable

**Comparison:**
```
# Full dynamic layout
full_dynamic = tensor.mark_layout_dynamic()
# Compiler knows: nothing about shape or stride

# Compact shape dynamic
compact_dynamic = tensor.mark_compact_shape_dynamic()
# Compiler knows: layout is compact (e.g., column-major, row-major)
# Compiler doesn't know: exact dimensions

# Full static
full_static = tensor.mark_layout_static()
# Compiler knows: everything (exact shape and stride)
```

### When to Use

✅ **Use mark_compact_shape_dynamic() when:**

**1. Fixed layout pattern, varying dimensions:**
```
# All tensors are column-major, but different sizes
@cute.jit
def column_major_processing(tensors: list):
    for tensor in tensors:
        # Knows: column-major layout
        # Doesn't know: exact M, N
        compact = from_dlpack(tensor).mark_compact_shape_dynamic()
        process(compact)
```

**2. Better optimization than full dynamic:**
```
# Compact dynamic is faster than full dynamic
# Because compiler knows layout is contiguous
compact = tensor.mark_compact_shape_dynamic()  # Faster
full_dynamic = tensor.mark_layout_dynamic()    # Slower
```

**3. Framework tensors with known memory order:**
```
# PyTorch tensors are typically contiguous
torch_tensor = torch.randn(M, N, device="cuda")

# Mark as compact dynamic (knows contiguous, not exact M, N)
cute_tensor = from_dlpack(torch_tensor).mark_compact_shape_dynamic()
```

---

## Comparison of Marking Options

### Performance Spectrum
```
Fastest                                                 Most Flexible
    |--------------------------------------------------|
    Static          Compact Dynamic        Full Dynamic
    (specialized)   (layout known)         (fully general)
    
Compile time: Highest  →  Medium  →  Lowest
Runtime perf: Best     →  Good    →  Acceptable
Flexibility:  None     →  Some    →  Full
```

### Feature Comparison Table

| Feature | Static | Compact Dynamic | Full Dynamic |
|---------|--------|-----------------|--------------|
| Shape known at compile time | ✅ Yes | ❌ No | ❌ No |
| Stride pattern known | ✅ Yes | ✅ Yes | ❌ No |
| Loop unrolling | ✅ Full | ⚠️ Limited | ❌ No |
| Address optimization | ✅ Best | ✅ Good | ⚠️ Basic |
| Recompilation needed | ✅ Yes | ❌ No | ❌ No |
| Works with any size | ❌ No | ✅ Yes | ✅ Yes |
| Works with any layout | ❌ No | ⚠️ Compact only | ✅ Yes |

---

## Practical Examples

### Example 1: Batch Processing with Dynamic
```
@cute.jit
def batch_process(batch_tensors: list):
    """Process batch of different-sized tensors efficiently"""
    
    for tensor in batch_tensors:
        # Each tensor has different size
        # Mark as dynamic to avoid recompilation
        dynamic_tensor = from_dlpack(tensor).mark_layout_dynamic()
        
        # Same kernel for all tensors
        process_kernel.launch(...)(dynamic_tensor)
```

### Example 2: Fixed Sizes with Static
```
@cute.jit
def gemm_fixed_sizes(A_torch, B_torch):
    """GEMM with fixed 1024×1024 matrices"""
    
    # Always 1024×1024 - specialize!
    A = from_dlpack(A_torch).mark_layout_static()
    B = from_dlpack(B_torch).mark_layout_static()
    
    C_torch = torch.zeros(1024, 1024, device="cuda")
    C = from_dlpack(C_torch).mark_layout_static()
    
    # Highly optimized for 1024×1024
    gemm_kernel.launch(...)(A, B, C)
    
    return C_torch
```

### Example 3: Hybrid Approach
```
@cute.jit
def hybrid_processing(tensor):
    """Use static for common sizes, dynamic for others"""
    
    m, n = tensor.shape
    
    # Common sizes: use specialized kernels
    if (m, n) in [(512, 512), (1024, 1024), (2048, 2048)]:
        static_tensor = from_dlpack(tensor).mark_layout_static()
        fast_kernel.launch(...)(static_tensor)
    else:
        # Uncommon size: use dynamic kernel
        dynamic_tensor = from_dlpack(tensor).mark_layout_dynamic()
        general_kernel.launch(...)(dynamic_tensor)
```

### Example 4: Compact Dynamic for Framework Tensors
```
@cute.jit
def process_contiguous_tensors(tensors: list):
    """Process list of contiguous PyTorch tensors"""
    
    for tensor in tensors:
        # PyTorch tensors are contiguous
        # Use compact dynamic (faster than full dynamic)
        compact = from_dlpack(tensor).mark_compact_shape_dynamic()
        
        # Compiler knows layout is contiguous
        # Can optimize better than full dynamic
        optimized_kernel.launch(...)(compact)
```

---

## When Marking Happens

### Automatic vs Explicit

**Automatic (default behavior):**
```
# CuTe makes default choice
cute_tensor = from_dlpack(torch_tensor)
# May be treated as static or dynamic depending on heuristics
```

**Explicit (recommended for control):**
```
# You decide
dynamic = from_dlpack(torch_tensor).mark_layout_dynamic()
static = from_dlpack(torch_tensor).mark_layout_static()
compact = from_dlpack(torch_tensor).mark_compact_shape_dynamic()
```

### Multiple Calls

**Last mark wins:**
```
tensor = from_dlpack(torch_tensor)
tensor = tensor.mark_layout_static()   # Static
tensor = tensor.mark_layout_dynamic()  # Now dynamic (overwrites)
tensor = tensor.mark_compact_shape_dynamic()  # Now compact dynamic
```

---

## Performance Impact

### Typical Overhead

**Static vs Compact Dynamic:**
- Compute-bound kernels: ~0-5% slower
- Memory-bound kernels: ~5-10% slower
- Tight loops: ~5-15% slower

**Compact Dynamic vs Full Dynamic:**
- Compute-bound: ~0-5% difference
- Memory-bound: ~5-10% difference
- Complex layouts: ~10-20% difference

### Measuring Impact
```
@cute.jit
def benchmark_marking():
    import time
    
    torch_tensor = torch.randn(1024, 1024, device="cuda")
    
    # Benchmark static
    static_tensor = from_dlpack(torch_tensor).mark_layout_static()
    start = time.time()
    for _ in range(1000):
        kernel.launch(...)(static_tensor)
    cute.arch.device_synchronize()
    static_time = time.time() - start
    
    # Benchmark dynamic
    dynamic_tensor = from_dlpack(torch_tensor).mark_layout_dynamic()
    start = time.time()
    for _ in range(1000):
        kernel.launch(...)(dynamic_tensor)
    cute.arch.device_synchronize()
    dynamic_time = time.time() - start
    
    overhead = (dynamic_time / static_time - 1.0) * 100
    print(f"Dynamic overhead: {overhead:.1f}%")
```

---

## Best Practices

### ✅ DO

**Mark explicitly for clarity:**
```
# Clear intent
dynamic = tensor.mark_layout_dynamic()
static = tensor.mark_layout_static()
```

**Use dynamic during development:**
```
# Prototyping - fast iteration
dev_tensor = tensor.mark_layout_dynamic()
```

**Use static for production hot paths:**
```
# Performance-critical - fully optimized
prod_tensor = tensor.mark_layout_static()
```

**Profile before optimizing:**
```
# Measure actual overhead before choosing
# Sometimes dynamic is "fast enough"
```

### ❌ DON'T

**Don't mark unnecessarily:**
```
# If default works fine, don't add complexity
tensor = from_dlpack(torch_tensor)  # Default is often OK
```

**Don't use static for many sizes:**
```
# ✗ BAD: 1000 specialized kernels!
for size in range(1000, 2000):
    tensor = torch.randn(size, size, device="cuda")
    kernel.launch(...)(from_dlpack(tensor).mark_layout_static())
```

**Don't ignore compilation time:**
```
# Static is slower to compile
# If compilation time matters, use dynamic
```

---

## Summary

**Three marking options:**

1. **mark_layout_static()**
   - Shape and stride known at compile time
   - Best performance
   - Requires recompilation for different sizes

2. **mark_compact_shape_dynamic()**
   - Shape dynamic, layout structure known
   - Good performance (better than full dynamic)
   - Single kernel for varying sizes with same layout pattern

3. **mark_layout_dynamic()**
   - Shape and stride both dynamic
   - Most flexible
   - Slight performance overhead

**Decision guide:**
- Fixed sizes? → Static
- Many sizes, same layout? → Compact dynamic
- Arbitrary sizes and layouts? → Full dynamic
- Prototyping? → Dynamic
- Production? → Profile, then choose

---

## Next Steps

- [Static vs Dynamic](./static_vs_dynamic.md) - Understanding the trade-offs
- [Alignment](./alignment.md) - Memory alignment considerations
- [Performance Tips](../08_optimization/performance_tips.md) - When to optimize

---

## Further Reading

- [Layout Algebra](../01_fundamentals/layouts.md)
- [Code Generation](../01_fundamentals/code_generation.md)