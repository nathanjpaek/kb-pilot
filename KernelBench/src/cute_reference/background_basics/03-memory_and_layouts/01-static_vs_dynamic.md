---
topic: "static_vs_dynamic"
difficulty: "intermediate"
related_topics: ["layouts", "tensors", "code_generation"]
---

# Static vs Dynamic Layouts in CuTe DSL

## Overview

CuTe DSL supports both **static layouts** (shape/stride known at compile time) and **dynamic layouts** (shape/stride known only at runtime). Understanding when to use each is crucial for balancing performance and flexibility.

**Key trade-off:** Static layouts enable better optimization but require recompilation for different sizes. Dynamic layouts are flexible but may be slightly slower.

---

## What Are Static and Dynamic Layouts?

### Static Layouts

**Shape and stride are compile-time constants:**
```
@cute.jit
def static_layout_example():
    # Shape is compile-time constant (128, 64)
    shape = cute.make_shape(128, 64)
    stride = cute.make_stride(1, 128)  # Column-major
    
    layout = cute.make_layout(shape, stride)
    # Compiler knows exact dimensions!
```

**Benefits:**
- Compiler can optimize address calculations
- Constant propagation through arithmetic
- Loop unrolling based on known sizes
- Better register allocation
- Potentially faster code

**Drawbacks:**
- Must recompile for different sizes
- Less flexible
- More compilation overhead

### Dynamic Layouts

**Shape and/or stride known only at runtime:**
```
@cute.jit
def dynamic_layout_example(m: cutlass.Int32, n: cutlass.Int32):
    # Shape is runtime value
    shape = cute.make_shape(m, n)
    stride = cute.make_stride(1, m)  # Column-major
    
    layout = cute.make_layout(shape, stride)
    # Compiler doesn't know dimensions at compile time
```

**Benefits:**
- Single kernel works for any size
- Flexible - no recompilation needed
- Faster compilation (fewer specialized kernels)

**Drawbacks:**
- Runtime address calculation overhead
- Less aggressive compiler optimization
- May be slightly slower

---

## Creating Static Layouts

### Using Compile-Time Constants
```
@cute.jit
def create_static_layout():
    # Python literals are compile-time constants
    tile_m = 128
    tile_n = 64
    
    # This layout is fully static
    layout = cute.make_layout(
        cute.make_shape(tile_m, tile_n),
        cute.make_stride(1, tile_m)
    )
    
    return layout
```

### Using Constexpr Parameters
```
@cute.kernel
def static_kernel(
    data: cute.Tensor,
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr
):
    # tile_m and tile_n are compile-time constants
    layout = cute.make_layout(
        cute.make_shape(tile_m, tile_n),
        cute.make_stride(1, tile_m)
    )
    
    # Compiler knows exact size, can optimize aggressively

# Different sizes generate different specialized kernels
static_kernel.launch(...)(data, tile_m=128, tile_n=64)  # Kernel A
static_kernel.launch(...)(data, tile_m=256, tile_n=128)  # Kernel B (different code!)
```

### Framework Tensors with Static Shapes
```
import torch
from cutlass.cute.runtime import from_dlpack

# PyTorch tensor with known shape
torch_tensor = torch.randn(1024, 512, device="cuda")

# Convert to CuTe - shape is static in the context
cute_tensor = from_dlpack(torch_tensor)

# Shape is (1024, 512) - known at this point
# But treated as dynamic unless you tell the compiler otherwise
```

---

## Creating Dynamic Layouts

### Using Runtime Parameters
```
@cute.jit
def create_dynamic_layout(m: cutlass.Int32, n: cutlass.Int32):
    # m and n are runtime values
    layout = cute.make_layout(
        cute.make_shape(m, n),
        cute.make_stride(1, m)
    )
    
    return layout

# Same compiled function handles any size
layout1 = create_dynamic_layout(128, 64)
layout2 = create_dynamic_layout(256, 128)
layout3 = create_dynamic_layout(512, 256)
```

### Tensor with Dynamic Shape
```
@cute.kernel
def dynamic_kernel(data: cute.Tensor):
    # data.shape is dynamic (known at runtime)
    m = cute.size(data, mode=0)
    n = cute.size(data, mode=1)
    
    # Create dynamic layout based on input
    tile_layout = cute.make_layout(
        cute.make_shape(m // 8, n // 8),
        cute.make_stride(1, m // 8)
    )
```

---

## Marking Layouts as Static or Dynamic

### Explicitly Marking Static
```
@cute.jit
def mark_static(data: cute.Tensor):
    # Tell compiler to treat this shape as static
    static_tensor = data.mark_layout_static()
    
    # Now compiler can optimize assuming fixed shape
    return static_tensor
```

### Explicitly Marking Dynamic
```
@cute.jit
def mark_dynamic(data: cute.Tensor):
    # Tell compiler to treat this shape as dynamic
    dynamic_tensor = data.mark_layout_dynamic()
    
    # Compiler won't assume fixed shape
    return dynamic_tensor
```

---

## Performance Implications

### Static Layout Performance

**Address calculation example:**
```
@cute.kernel
def static_access(data: cute.Tensor, tile_size: cutlass.Constexpr):
    # tile_size is 128 (compile-time constant)
    for i in cutlass.range_constexpr(tile_size):
        # Compiler knows exact address: base + i * stride
        # Can constant-fold: base + i * 1 = base + i
        val = data[i]
```

**Generated assembly (pseudocode):**
```
# Highly optimized, addresses computed at compile time where possible
load r1, [base + 0]
load r2, [base + 1]
load r3, [base + 2]
# ... fully unrolled, optimal
```

### Dynamic Layout Performance

**Address calculation example:**
```
@cute.kernel
def dynamic_access(data: cute.Tensor, tile_size: cutlass.Int32):
    # tile_size is runtime value
    for i in range(tile_size):
        # Compiler must compute address at runtime
        val = data[i]
```

**Generated assembly (pseudocode):**
```
# Runtime address calculation
loop:
    mul temp, i, stride  # Runtime multiplication
    add addr, base, temp # Runtime addition
    load r1, [addr]
    inc i
    cmp i, tile_size
    jlt loop
```

### Typical Performance Difference

**Benchmarks (approximate):**

| Operation | Static Layout | Dynamic Layout | Overhead |
|-----------|---------------|----------------|----------|
| Simple indexing | 1.0x | 1.05-1.1x | ~5-10% |
| Complex indexing | 1.0x | 1.1-1.2x | ~10-20% |
| Tight loops | 1.0x | 1.05-1.15x | ~5-15% |
| Compute-heavy | 1.0x | 1.0-1.05x | ~0-5% |

**Key insight:** Dynamic overhead is small for compute-bound kernels, more noticeable for memory-bound kernels.

---

## When to Use Static Layouts

### ✅ Use Static Layouts When:

**1. Fixed problem sizes:**
```
@cute.kernel
def fixed_size_gemm(
    A: cute.Tensor,  # Always 1024x1024
    B: cute.Tensor,  # Always 1024x1024
    C: cute.Tensor   # Always 1024x1024
):
    # Shapes never change - use static!
    tile_m = 128
    tile_n = 128
    tile_k = 32
```

**2. Performance-critical kernels:**
```
@cute.kernel
def performance_critical(
    data: cute.Tensor,
    tile_size: cutlass.Constexpr  # Static for best performance
):
    # Every cycle counts - use static
    for i in cutlass.range_constexpr(tile_size):
        process(data[i])
```

**3. Small number of size variants:**
```
# Only need 3 sizes - generate 3 specialized kernels
for tile_size in [64, 128, 256]:
    optimized_kernel.launch(...)(data, tile_size=tile_size)
```

**4. Tensor core operations:**
```
# MMA atoms often require static shapes for optimal performance
mma_atom = cute.make_mma_atom(
    cute.SM80_16x8x16_F16F16F16F16_TN  # Fixed 16x8x16 shape
)
```

---

## When to Use Dynamic Layouts

### ✅ Use Dynamic Layouts When:

**1. Arbitrary input sizes:**
```
@cute.kernel
def flexible_kernel(data: cute.Tensor):
    # Works with any input size
    n = cute.size(data)
    
    for i in range(n):
        process(data[i])

# Same kernel handles any size
flexible_kernel.launch(...)(small_data)   # 100 elements
flexible_kernel.launch(...)(medium_data)  # 10,000 elements
flexible_kernel.launch(...)(large_data)   # 1,000,000 elements
```

**2. Many size variants:**
```
# Hundreds of possible sizes - don't want hundreds of kernels!
@cute.kernel
def general_purpose_kernel(
    data: cute.Tensor,
    tile_size: cutlass.Int32  # Dynamic
):
    # Single kernel for all sizes
    for i in range(tile_size):
        process(data[i])
```

**3. Compilation time matters:**
```
# Dynamic compiles once, static compiles multiple times
# If compilation time is bottleneck, use dynamic
```

**4. Prototyping and development:**
```
# During development, dynamic is more convenient
# Optimize to static after profiling shows it matters
```

---

## Hybrid Approach: Partially Static

### Static Outer Dimensions, Dynamic Inner
```
@cute.kernel
def hybrid_kernel(
    data: cute.Tensor,
    tile_m: cutlass.Constexpr,  # Static
    tile_n: cutlass.Constexpr,  # Static
    num_tiles: cutlass.Int32     # Dynamic
):
    # Tile size is static (optimized)
    # Number of tiles is dynamic (flexible)
    
    for tile_idx in range(num_tiles):
        # Process tile with static dimensions
        for i in cutlass.range_constexpr(tile_m):
            for j in cutlass.range_constexpr(tile_n):
                idx = tile_idx * tile_m * tile_n + i * tile_n + j
                process(data[idx])
```

**Benefits:**
- Inner loops optimized (static tile size)
- Flexibility in number of tiles (dynamic count)
- Good balance of performance and flexibility

---

## Compile-Time Size Validation

### Using Assertions
```
@cute.kernel
def validated_kernel(
    data: cute.Tensor,
    tile_size: cutlass.Constexpr
):
    # Validate at compile time
    if cutlass.const_expr(tile_size % 16 != 0):
        raise ValueError("tile_size must be multiple of 16")
    
    if cutlass.const_expr(tile_size > 512):
        raise ValueError("tile_size too large (max 512)")
    
    # If we get here, tile_size is valid
    for i in cutlass.range_constexpr(tile_size):
        process(data[i])

# This will fail at compile time:
# validated_kernel.launch(...)(data, tile_size=17)  # ERROR: not multiple of 16
```

---

## Dynamic Shapes with Assumptions

### Using cute.assume()

**Help the compiler optimize dynamic code:**
```
@cute.kernel
def optimized_dynamic(data: cute.Tensor):
    n = cute.size(data)
    
    # Tell compiler n is divisible by 16
    n = cute.assume(n, divby=16)
    
    # Compiler can now vectorize/unroll by 16
    for i in range(n):
        process(data[i])
```

**Available assumptions:**
```
# Divisibility
n = cute.assume(n, divby=16)    # n is multiple of 16
n = cute.assume(n, divby=32)    # n is multiple of 32

# Range
n = cute.assume(n, min=128)     # n >= 128
n = cute.assume(n, max=1024)    # n <= 1024
n = cute.assume(n, min=64, max=256)  # 64 <= n <= 256

# Alignment
ptr = cute.assume(ptr, align=128)  # ptr is 128-byte aligned
```

**Benefits:**
- Enables better optimization of dynamic code
- No runtime overhead (just compiler hints)
- Bridges gap between static and dynamic performance

---

## Compilation Strategies

### Strategy 1: One Static Kernel Per Size
```
# Generate specialized kernel for each common size
common_sizes = [64, 128, 256, 512]

for size in common_sizes:
    specialized_kernel.launch(...)(data, tile_size=size)

# Fast execution, but 4 compiled kernels
```

### Strategy 2: One Dynamic Kernel for All Sizes
```
# Single kernel handles all sizes
general_kernel.launch(...)(data, tile_size=user_size)

# Slower execution, but 1 compiled kernel
```

### Strategy 3: Hybrid - Static for Common, Dynamic for Others
```
# Most common sizes use static specialized kernels
if tile_size in [64, 128, 256]:
    static_kernel.launch(...)(data, tile_size=tile_size)
else:
    # Fall back to dynamic for uncommon sizes
    dynamic_kernel.launch(...)(data, tile_size)

# Good balance: fast common case, flexible uncommon case
```

---

## Real-World Example: GEMM

### Full Static GEMM
```
@cute.kernel
def static_gemm(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    m: cutlass.Constexpr,  # Static
    n: cutlass.Constexpr,  # Static
    k: cutlass.Constexpr   # Static
):
    # Everything compile-time known
    tile_m = 128
    tile_n = 128
    tile_k = 32
    
    # Fully optimized loops
    for i in cutlass.range_constexpr(m // tile_m):
        for j in cutlass.range_constexpr(n // tile_n):
            for kk in cutlass.range_constexpr(k // tile_k):
                gemm_tile(A, B, C, i, j, kk)

# Generate kernels for common sizes
static_gemm.launch(...)(A, B, C, m=1024, n=1024, k=1024)
static_gemm.launch(...)(A, B, C, m=2048, n=2048, k=2048)
```

### Hybrid GEMM (Recommended)
```
@cute.kernel
def hybrid_gemm(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    m: cutlass.Int32,          # Dynamic
    n: cutlass.Int32,          # Dynamic
    k: cutlass.Int32,          # Dynamic
    tile_m: cutlass.Constexpr, # Static
    tile_n: cutlass.Constexpr, # Static
    tile_k: cutlass.Constexpr  # Static
):
    # Problem size is dynamic (flexible)
    # Tile size is static (optimized)
    
    num_tiles_m = m // tile_m
    num_tiles_n = n // tile_n
    num_tiles_k = k // tile_k
    
    # Outer loops dynamic
    for i in range(num_tiles_m):
        for j in range(num_tiles_n):
            for kk in range(num_tiles_k):
                # Inner tile processing is optimized (static tile size)
                for ii in cutlass.range_constexpr(tile_m):
                    for jj in cutlass.range_constexpr(tile_n):
                        for kkk in cutlass.range_constexpr(tile_k):
                            compute(A, B, C, i, j, kk, ii, jj, kkk)

# Single kernel works for any problem size (with fixed tile size)
hybrid_gemm.launch(...)(A, B, C, m, n, k, tile_m=128, tile_n=128, tile_k=32)
```

---

## Measuring the Impact

### Benchmark Static vs Dynamic
```
@cute.jit
def benchmark_static_vs_dynamic():
    import time
    
    data = cute.make_tensor(cute.make_shape(1024, 1024), cutlass.Float32)
    
    # Static version
    start = time.time()
    for _ in range(1000):
        static_kernel.launch(...)(data, tile_size=128)
    cute.arch.device_synchronize()
    static_time = time.time() - start
    
    # Dynamic version
    start = time.time()
    for _ in range(1000):
        dynamic_kernel.launch(...)(data, 128)  # 128 is runtime value
    cute.arch.device_synchronize()
    dynamic_time = time.time() - start
    
    overhead = (dynamic_time / static_time - 1.0) * 100
    print(f"Dynamic overhead: {overhead:.1f}%")
```

---

## Best Practices

### ✅ DO

**Use static for performance-critical inner loops:**
```
@cute.kernel
def good_hybrid(data: cute.Tensor, outer_size: cutlass.Int32):
    inner_size = 128  # Static
    
    for i in range(outer_size):  # Dynamic
        for j in cutlass.range_constexpr(inner_size):  # Static, optimized
            process(data[i * inner_size + j])
```

**Profile before optimizing:**
```
# Measure actual overhead before switching to static
# Sometimes dynamic is "fast enough"
```

**Use assumptions to help dynamic code:**
```
n = cute.assume(n, divby=16, min=128)
# Helps compiler optimize without full static
```

### ❌ DON'T

**Don't use static unnecessarily:**
```
# ✗ BAD: Forcing static when dynamic would work fine
# Generates 100 kernels for no good reason
for size in range(100, 200):
    over_specialized_kernel.launch(...)(data, tile_size=size)
```

**Don't ignore compilation time:**
```
# ✗ BAD: Compiling thousands of static variants
# Compilation takes minutes, only saves microseconds per kernel
```

**Don't assume static is always faster:**
```
# Profile! For compute-bound kernels, difference may be negligible
```

---

## Summary

**Static layouts:**
- Shape/stride known at compile time
- Better optimization, faster execution
- Less flexible, more compilation

**Dynamic layouts:**
- Shape/stride known at runtime
- More flexible, faster compilation
- Slightly slower execution (~5-20% typical)

**Key decisions:**
- Performance critical? → Static
- Many size variants? → Dynamic
- Compute-bound? → Dynamic usually fine
- Memory-bound? → Static helps more

**Best approach:**
- **Hybrid**: Static inner dimensions, dynamic outer
- Use `cute.assume()` to help dynamic code
- Profile to measure actual impact
- Don't over-optimize prematurely

---

## Next Steps

- [Layouts](../01_fundamentals/layouts.md) - Understanding layout algebra
- [Code Generation](../01_fundamentals/code_generation.md) - How static/dynamic affects compilation
- [Performance Tips](../08_optimization/performance_tips.md) - When to optimize

---

## Further Reading

- [Compile-Time vs Runtime](https://en.wikipedia.org/wiki/Compile_time)
- [Loop Unrolling](https://en.wikipedia.org/wiki/Loop_unrolling)