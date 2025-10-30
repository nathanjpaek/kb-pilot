---
topic: "metaprogramming"
difficulty: "advanced"
related_topics: ["conditionals", "loops", "code_generation"]
---

# Compile-Time Metaprogramming in CuTe DSL

## Overview

Compile-time metaprogramming in CuTe DSL allows you to generate specialized, optimized GPU kernels based on configuration parameters known at compilation time. This enables zero-overhead abstractions and kernel specialization without runtime branching.

**Key concept:** Code that runs during JIT compilation to generate different optimized kernels for different configurations.

---

## What is Metaprogramming?

### Traditional Programming (Runtime)

All decisions happen when the program runs on the GPU:
```
@cute.kernel
def runtime_kernel(data: cute.Tensor, mode: cutlass.Int32):
    # Decision at runtime - both paths in code
    if mode == 1:
        fast_path(data)
    else:
        slow_path(data)
    
    # Runtime overhead: branch prediction, both paths compiled
```

### Metaprogramming (Compile-Time)

Decisions happen during JIT compilation - generates different code for each configuration:
```
@cute.kernel
def compile_time_kernel(data: cute.Tensor, mode: cutlass.Constexpr):
    # Decision at compile time - only one path in code
    if cutlass.const_expr(mode == 1):
        fast_path(data)
    else:
        slow_path(data)
    
    # Zero runtime overhead: no branching, dead code eliminated

# Two different kernels generated:
compile_time_kernel.launch(...)(data, mode=1)  # Only has fast_path
compile_time_kernel.launch(...)(data, mode=2)  # Only has slow_path
```

---

## Core Metaprogramming Tool: cutlass.Constexpr

### What is Constexpr?

`cutlass.Constexpr` is a type annotation that marks a parameter as a **compile-time constant**.

**Usage:**
```
@cute.jit
def my_function(
    dynamic_param: cutlass.Int32,      # Runtime value
    static_param: cutlass.Constexpr    # Compile-time constant
):
    # static_param is known at compile time
    # Can use in compile-time constructs
    pass
```

**Key properties:**
- Value must be known when calling the function
- Different values generate different compiled kernels
- Can be used in `cutlass.range_constexpr()`, `cutlass.const_expr()`, etc.

---

## Compile-Time Constructs

### 1. Compile-Time Loops: cutlass.range_constexpr()

**Fully unroll loops with known bounds:**
```
@cute.jit
def unrolled_loop(num_stages: cutlass.Constexpr):
    # Loop executes at compile time, fully unrolled
    for stage in cutlass.range_constexpr(num_stages):
        initialize_stage(stage)
    
    # If num_stages=4, generates:
    # initialize_stage(0)
    # initialize_stage(1)
    # initialize_stage(2)
    # initialize_stage(3)
```

### 2. Compile-Time Conditionals: cutlass.const_expr()

**Eliminate unused branches:**
```
@cute.kernel
def conditional_kernel(
    data: cute.Tensor,
    use_optimization: cutlass.Constexpr
):
    if cutlass.const_expr(use_optimization):
        optimized_path(data)  # Only in "optimized" kernel
    else:
        standard_path(data)   # Only in "standard" kernel
```

### 3. Compile-Time Assertions

**Validate configuration at compile time:**
```
@cute.jit
def validated_kernel(tile_size: cutlass.Constexpr):
    # Compile-time assertion
    if cutlass.const_expr(tile_size % 16 != 0):
        raise ValueError("tile_size must be multiple of 16")
    
    # If assertion fails, compilation fails (good!)
    # Runtime won't execute invalid configuration
```

---

## Common Metaprogramming Patterns

### Pattern 1: Activation Function Fusion

**Generate specialized kernels for each activation:**
```
@cute.kernel
def fused_linear_activation(
    input: cute.Tensor,
    weights: cute.Tensor,
    output: cute.Tensor,
    activation: cutlass.Constexpr  # "relu", "gelu", "sigmoid", "none"
):
    idx = cute.arch.thread_idx()[0]
    
    # Linear transformation
    result = input[idx] * weights[idx]
    
    # Compile-time dispatch - zero overhead!
    if cutlass.const_expr(activation == "relu"):
        result = cute.where(result > 0, result, 0.0)
    elif cutlass.const_expr(activation == "gelu"):
        # GELU approximation
        result = 0.5 * result * (1.0 + cute.tanh(0.797885 * (result + 0.044715 * result * result * result)))
    elif cutlass.const_expr(activation == "sigmoid"):
        result = 1.0 / (1.0 + cute.exp(-result))
    # else: activation == "none", no-op
    
    output[idx] = result

# Generate 4 specialized kernels:
fused_linear_activation.launch(...)(input, weights, output, "relu")    # ReLU kernel
fused_linear_activation.launch(...)(input, weights, output, "gelu")    # GELU kernel
fused_linear_activation.launch(...)(input, weights, output, "sigmoid") # Sigmoid kernel
fused_linear_activation.launch(...)(input, weights, output, "none")    # Identity kernel
```

**Benefits:**
- Each kernel contains only the activation it needs
- No runtime branching
- Smaller code, better instruction cache usage
- Compiler can optimize each path independently

### Pattern 2: Precision Selection

**Generate kernels for different precisions:**
```
@cute.kernel
def mixed_precision_gemm(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    compute_precision: cutlass.Constexpr  # "fp16", "fp32", "tf32"
):
    # Compile-time precision selection
    if cutlass.const_expr(compute_precision == "fp16"):
        # FP16 MMA atoms
        mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F16F16_TN)
        acc_dtype = cutlass.Float16
    elif cutlass.const_expr(compute_precision == "fp32"):
        # FP32 compute
        mma = cute.make_mma_atom(cute.SM80_16x8x8_F32F32F32F32_TN)
        acc_dtype = cutlass.Float32
    elif cutlass.const_expr(compute_precision == "tf32"):
        # TF32 (Ampere+)
        mma = cute.make_mma_atom(cute.SM80_16x8x8_F32TF32TF32F32_TN)
        acc_dtype = cutlass.Float32
    
    # Rest of GEMM logic...
    # Different MMA instruction generated for each precision!
```

### Pattern 3: Tile Size Specialization

**Generate optimal kernels for each tile size:**
```
@cute.kernel
def tiled_kernel(
    data: cute.Tensor,
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr
):
    # Tile sizes known at compile time
    # Enables aggressive loop unrolling and optimization
    
    for i in cutlass.range_constexpr(tile_m):
        for j in cutlass.range_constexpr(tile_n):
            process(data[i, j])
    
    # If tile_m=16, tile_n=16, generates 256 process() calls
    # Fully unrolled, no loop overhead

# Generate specialized kernels:
tiled_kernel.launch(...)(data, tile_m=16, tile_n=16)   # 16x16 kernel
tiled_kernel.launch(...)(data, tile_m=32, tile_n=32)   # 32x32 kernel
tiled_kernel.launch(...)(data, tile_m=64, tile_n=64)   # 64x64 kernel
```

### Pattern 4: Feature Toggles

**Enable/disable features at compile time:**
```
@cute.kernel
def configurable_kernel(
    data: cute.Tensor,
    use_shared_memory: cutlass.Constexpr,
    use_vectorization: cutlass.Constexpr,
    use_tensor_cores: cutlass.Constexpr
):
    # Each feature is compile-time toggle
    if cutlass.const_expr(use_shared_memory):
        # Shared memory code path
        smem = allocate_shared_memory()
        load_to_shared(data, smem)
        process_from_shared(smem)
    else:
        # Direct global memory access
        process_directly(data)
    
    if cutlass.const_expr(use_vectorization):
        # Vectorized loads/stores
        vectorized_process(data)
    else:
        # Scalar processing
        scalar_process(data)
    
    if cutlass.const_expr(use_tensor_cores):
        # Tensor core MMA
        tensor_core_compute(data)
    else:
        # Regular CUDA cores
        regular_compute(data)

# 2^3 = 8 possible kernel configurations
configurable_kernel.launch(...)(data, True, True, True)    # All features
configurable_kernel.launch(...)(data, True, True, False)   # SMEM + vec, no TC
configurable_kernel.launch(...)(data, False, False, False) # Minimal kernel
```

### Pattern 5: Architecture-Specific Dispatch

**Generate different code for each GPU architecture:**
```
@cute.kernel
def arch_specific_kernel(
    data: cute.Tensor,
    architecture: cutlass.Constexpr  # "Ampere", "Hopper", "Blackwell"
):
    if cutlass.const_expr(architecture == "Ampere"):
        # Use Ampere-specific features
        use_ampere_mma(data)
        max_smem = 164 * 1024
    elif cutlass.const_expr(architecture == "Hopper"):
        # Use Hopper TMA
        use_hopper_tma(data)
        max_smem = 256 * 1024
    elif cutlass.const_expr(architecture == "Blackwell"):
        # Use Blackwell 2-CTA instructions
        use_blackwell_2cta(data)
        max_smem = 256 * 1024
    
    # Compile-time computed shared memory size
    smem_needed = compute_smem_size()
    if cutlass.const_expr(smem_needed > max_smem):
        raise ValueError(f"Shared memory {smem_needed} exceeds {max_smem}")

# Generate architecture-specific kernels:
arch_specific_kernel.launch(...)(data, "Ampere")    # Ampere-optimized
arch_specific_kernel.launch(...)(data, "Hopper")    # Hopper-optimized
arch_specific_kernel.launch(...)(data, "Blackwell") # Blackwell-optimized
```

---

## Advanced: Type-Based Metaprogramming

### Generic Functions with Constexpr Types
```
@cute.jit
def type_dispatched_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    output_dtype: cutlass.Constexpr  # cutlass.Float16, cutlass.Float32, etc.
):
    idx = cute.arch.thread_idx()[0]
    
    # Load input
    val = input[idx]
    
    # Process based on output type
    if cutlass.const_expr(output_dtype == cutlass.Float16):
        # FP16 output - may need clamping
        processed = clamp_to_fp16_range(val)
        output[idx] = processed.to(cutlass.Float16)
    elif cutlass.const_expr(output_dtype == cutlass.Float32):
        # FP32 output - full precision
        output[idx] = val.to(cutlass.Float32)
    elif cutlass.const_expr(output_dtype == cutlass.Int8):
        # INT8 output - quantize
        quantized = quantize_to_int8(val)
        output[idx] = quantized.to(cutlass.Int8)
```

---

## Metaprogramming with Python Functions

### Python Functions as Compile-Time Helpers

Pure Python functions can be called during compilation:
```
# Pure Python helper (runs at compile time)
def compute_optimal_tile_size(problem_size):
    """Calculate best tile size based on problem size"""
    if problem_size <= 1024:
        return 64
    elif problem_size <= 4096:
        return 128
    else:
        return 256

@cute.kernel
def adaptive_tiling_kernel(
    data: cute.Tensor,
    problem_size: cutlass.Constexpr
):
    # Call Python function at compile time
    tile_size = compute_optimal_tile_size(problem_size)
    
    # Use compile-time computed tile size
    for i in cutlass.range_constexpr(tile_size):
        process(data[i])

# Different tile sizes for different problem sizes:
adaptive_tiling_kernel.launch(...)(data, problem_size=512)   # tile_size=64
adaptive_tiling_kernel.launch(...)(data, problem_size=2048)  # tile_size=128
adaptive_tiling_kernel.launch(...)(data, problem_size=8192)  # tile_size=256
```

### Complex Compile-Time Logic
```
def select_mma_atom(dtype, architecture):
    """Select optimal MMA atom based on dtype and architecture"""
    if architecture == "Hopper":
        if dtype == "fp8":
            return "MMA_16x8x32_E4M3E5M2F16_SS"
        elif dtype == "fp16":
            return "MMA_16x8x16_F16F16F16F16_TN"
    elif architecture == "Ampere":
        if dtype == "fp16":
            return "MMA_16x8x16_F16F16F16F16_TN"
        elif dtype == "bf16":
            return "MMA_16x8x16_BF16BF16F32F32_TN"
    
    raise ValueError(f"Unsupported combination: {dtype} on {architecture}")

@cute.kernel
def auto_mma_kernel(
    data: cute.Tensor,
    dtype: cutlass.Constexpr,
    arch: cutlass.Constexpr
):
    # Compute MMA atom at compile time
    mma_atom_name = select_mma_atom(dtype, arch)
    
    # Different MMA instructions for each configuration
    if cutlass.const_expr(mma_atom_name == "MMA_16x8x32_E4M3E5M2F16_SS"):
        # Hopper FP8
        mma = cute.make_mma_atom(...)
    # ... other cases
```

---

## Benefits of Metaprogramming

### 1. Zero Runtime Overhead

**Without metaprogramming (runtime branching):**
```
# All paths compiled, runtime decision
if mode == 1:
    path_a()  # Branch prediction
elif mode == 2:
    path_b()  # Both exist in code
else:
    path_c()  # Instruction cache pollution
```

**With metaprogramming (compile-time):**
```
# Only one path compiled per kernel
if cutlass.const_expr(mode == 1):
    path_a()  # Only this in kernel
# Other paths don't exist!
```

### 2. Better Compiler Optimization

**Compiler can:**
- Constant propagate through computation
- Dead code elimination
- Loop unrolling with known bounds
- Instruction scheduling
- Register allocation optimization

### 3. Reduced Code Size

**Each kernel contains only what it needs:**
- Smaller instruction cache footprint
- Better I-cache hit rate
- Faster kernel launch

### 4. Type Safety

**Catch configuration errors at compile time:**
```
@cute.jit
def validated(tile_size: cutlass.Constexpr):
    if cutlass.const_expr(tile_size % 16 != 0):
        raise ValueError("Invalid tile_size")
    # Compilation fails before runtime!
```

---

## Trade-offs and Considerations

### Compilation Time

**More configurations = more compilation time:**
```
# 4 activations × 3 precisions × 8 tile sizes = 96 kernels!
for activation in ["relu", "gelu", "sigmoid", "none"]:
    for precision in ["fp16", "fp32", "tf32"]:
        for tile in [64, 96, 128, 160, 192, 224, 256, 288]:
            kernel.launch(...)(data, activation, precision, tile)
```

**Mitigation strategies:**
- Cache compiled kernels (CuTe DSL does this automatically)
- Pre-compile common configurations
- Use runtime parameters for less critical choices

### Code Size

**Each configuration is a separate kernel:**
- Binary size grows with configurations
- May exhaust GPU code cache
- Consider limiting to most important configurations

### When NOT to Use Metaprogramming

❌ **Don't use for:**
- Values that change frequently
- User-controlled parameters
- Data-dependent decisions
- When small number of configurations covers most cases

✅ **Do use for:**
- Fixed configurations (precision, architecture, features)
- Performance-critical decisions
- Values known at kernel launch time
- Avoiding runtime overhead in hot paths

---

## Debugging Metaprogramming

### Print Compile-Time Values
```
@cute.jit
def debug_metaprog(tile_size: cutlass.Constexpr):
    # Compile-time print (Python print)
    print(f"Compiling with tile_size={tile_size}")
    
    # Shows during compilation
    for i in cutlass.range_constexpr(tile_size):
        print(f"  Unrolling iteration {i}")
```

### Dump Generated IR
```
export CUTE_DSL_PRINT_IR=1

# Then run code - see generated IR for each configuration
# Verify dead code elimination, unrolling, etc.
```

### Verify Different Kernels Generated
```
@cute.jit
def test_variants():
    kernel_a = cute.compile(my_kernel, data, config=1)
    kernel_b = cute.compile(my_kernel, data, config=2)
    
    # These should be different objects (different compiled kernels)
    assert kernel_a is not kernel_b
    print("✓ Different kernels generated for different configs")
```

---

## Best Practices

### ✅ DO

**Use Constexpr for configuration:**
```
@cute.kernel
def good_kernel(
    data: cute.Tensor,
    feature_enabled: cutlass.Constexpr  # ✓
):
    if cutlass.const_expr(feature_enabled):
        use_feature(data)
```

**Validate early:**
```
@cute.jit
def good_validation(tile_size: cutlass.Constexpr):
    if cutlass.const_expr(tile_size <= 0 or tile_size > 1024):
        raise ValueError("Invalid tile_size")
```

**Document configurations:**
```
@cute.kernel
def documented_kernel(
    data: cute.Tensor,
    mode: cutlass.Constexpr  # "fast", "balanced", "accurate"
):
    """
    Kernel with three compilation modes:
    - "fast": Maximum performance, lower precision
    - "balanced": Good performance, good precision
    - "accurate": Maximum precision, lower performance
    """
    pass
```

### ❌ DON'T

**Don't use Constexpr for data-dependent values:**
```
@cute.kernel
def bad_kernel(
    data: cute.Tensor,
    threshold: cutlass.Constexpr  # ✗ Should be runtime parameter!
):
    # This forces recompilation for every threshold value!
    if cutlass.const_expr(data[0] > threshold):
        pass
```

**Don't over-specialize:**
```
# ✗ TOO MANY: 10×10×10 = 1000 kernel variants!
def bad_specialization(
    tile_m: cutlass.Constexpr,  # 10 values
    tile_n: cutlass.Constexpr,  # 10 values
    tile_k: cutlass.Constexpr   # 10 values
):
    pass

# ✓ BETTER: Fewer, meaningful configurations
def good_specialization(
    tile_config: cutlass.Constexpr  # "small", "medium", "large"
):
    pass
```

---

## Summary

**Metaprogramming in CuTe DSL:**
- Use `cutlass.Constexpr` for compile-time parameters
- Use `cutlass.const_expr()` for compile-time conditionals
- Use `cutlass.range_constexpr()` for compile-time loops
- Generates specialized kernels for each configuration

**Benefits:**
- Zero runtime overhead (no branching)
- Better compiler optimization
- Smaller code size per kernel
- Compile-time error checking

**Best for:**
- Activation function selection
- Precision modes
- Architecture-specific features
- Feature enable/disable
- Tile size specialization

**Watch out for:**
- Compilation time (many configs = many kernels)
- Code size growth
- Overspecialization

**Remember:** Use metaprogramming for configuration, use runtime parameters for data!

---

## Next Steps

- [Conditionals](./conditionals.md) - Compile-time vs runtime branching
- [Loops](./loops.md) - Compile-time vs runtime loops
- [Code Generation](../01_fundamentals/code_generation.md) - How CuTe generates code

---

## Further Reading

- [C++ Template Metaprogramming](https://en.cppreference.com/w/cpp/language/templates) - Similar concepts in C++
- [Kernel Specialization](../08_optimization/performance_tips.md) - When to specialize