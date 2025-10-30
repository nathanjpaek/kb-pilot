---
topic: "conditionals"
difficulty: "intermediate"
related_topics: ["loops", "metaprogramming", "code_generation"]
---

# Conditionals in CuTe DSL

## Overview

CuTe DSL supports Python's if/elif/else statements and converts them into GPU-executable code. Like loops, conditionals can be either **compile-time** (evaluated during JIT compilation) or **runtime** (evaluated during GPU execution).

---

## Two Types of Conditionals

### 1. Runtime Conditionals (Default)

**Predicate without annotation** → lowered to IR, evaluated at runtime on GPU.

**Basic usage:**
```
@cute.jit
def runtime_conditional(x: cutlass.Int32):
    # Runtime conditional - evaluated on GPU
    if x > 10:
        cute.printf("x is large\n")
    else:
        cute.printf("x is small\n")
```

**What happens:**
- Both branches exist in generated GPU code
- GPU evaluates the condition at runtime
- Executes appropriate branch

### 2. Compile-Time Conditionals

**Predicate annotated with cutlass.const_expr** → evaluated at compile time.

**Basic usage:**
```
@cute.jit
def compile_time_conditional(flag: cutlass.Constexpr):
    # Compile-time conditional
    if cutlass.const_expr(flag):
        cute.printf("Flag is True\n")
    else:
        cute.printf("Flag is False\n")
```

**What happens:**
- Condition evaluated during JIT compilation
- Only the taken branch exists in generated code
- Unused branch is completely eliminated

---

## Runtime If/Else Statements

### Basic If Statement
```
@cute.jit
def basic_if(value: cutlass.Float32):
    if value > 0:
        cute.printf("Positive\n")
```

**Generated code has conditional branch.**

### If/Else
```
@cute.jit
def if_else(x: cutlass.Int32):
    if x > 0:
        cute.printf("Positive\n")
    else:
        cute.printf("Non-positive\n")
```

**Both branches in generated code.**

### If/Elif/Else Chain
```
@cute.jit
def if_elif_else(grade: cutlass.Int32):
    if grade >= 90:
        cute.printf("A\n")
    elif grade >= 80:
        cute.printf("B\n")
    elif grade >= 70:
        cute.printf("C\n")
    else:
        cute.printf("F\n")
```

**All branches in generated code, evaluated sequentially.**

### Nested Conditionals
```
@cute.jit
def nested_if(x: cutlass.Int32, y: cutlass.Int32):
    if x > 0:
        if y > 0:
            cute.printf("Both positive\n")
        else:
            cute.printf("x positive, y non-positive\n")
    else:
        cute.printf("x non-positive\n")
```

---

## Compile-Time If/Else Statements

### Using cutlass.const_expr()

**Purpose:** Evaluate conditions at compile time to eliminate unused code paths.

**Basic usage:**
```
@cute.jit
def compile_time_if(use_optimization: cutlass.Constexpr):
    # Condition evaluated at compile time
    if cutlass.const_expr(use_optimization):
        optimized_path()
    else:
        standard_path()
```

**Result:**
- If called with use_optimization=True, only optimized_path() in generated code
- If called with use_optimization=False, only standard_path() in generated code

### Feature Selection Example
```
@cute.kernel
def kernel_with_features(
    data: cute.Tensor,
    use_vectorization: cutlass.Constexpr,
    use_shared_memory: cutlass.Constexpr
):
    # Both conditions evaluated at compile time
    if cutlass.const_expr(use_vectorization):
        # Vectorized loads
        vectorized_load(data)
    else:
        # Scalar loads
        scalar_load(data)
    
    if cutlass.const_expr(use_shared_memory):
        # Use shared memory staging
        smem_process(data)
    else:
        # Direct from global memory
        direct_process(data)

# Generate different kernel variants
kernel_with_features.launch(...)(data, True, True)   # Vectorized + SMEM
kernel_with_features.launch(...)(data, True, False)  # Vectorized, no SMEM
kernel_with_features.launch(...)(data, False, True)  # Scalar + SMEM
kernel_with_features.launch(...)(data, False, False) # Scalar, no SMEM
```

**Four different compiled kernels**, each optimized for its configuration.

---

## Comparing Runtime vs Compile-Time

### Visual Comparison

**Runtime conditional:**
```
@cute.jit
def runtime_example(x: cutlass.Int32):
    if x > 5:
        path_a()
    else:
        path_b()
```

**Generated GPU code:**
```
if (x > 5) {
    path_a();
} else {
    path_b();
}
```

**Compile-time conditional:**
```
@cute.jit
def compile_time_example(flag: cutlass.Constexpr):
    if cutlass.const_expr(flag):
        path_a()
    else:
        path_b()
```

**If flag=True, generated GPU code (illustrative):**
```
path_a()  # path_b() doesn't exist
```

**If flag=False, generated GPU code (illustrative):**
```
path_b()  # path_a() doesn't exist
```

### Feature Comparison

| Aspect | Runtime If | Compile-Time If |
|--------|-----------|-----------------|
| Evaluation time | GPU runtime | JIT compile time |
| Code size | Both branches | Only taken branch |
| Performance | Branch prediction overhead | No branching overhead |
| Flexibility | Any runtime value | Only compile-time constants |
| Use case | Data-dependent logic | Configuration-dependent logic |

---

## When to Use Each Type

### Use Runtime Conditionals When:

✅ **Condition depends on runtime data**
```
@cute.jit
def runtime_logic(data: cute.Tensor):
    # Condition depends on actual data values
    if data[0] > threshold:
        process_high(data)
    else:
        process_low(data)
```

✅ **Condition is different per thread**
```
@cute.kernel
def thread_conditional(data: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    
    # Different threads take different paths
    if idx % 2 == 0:
        even_thread_work(data[idx])
    else:
        odd_thread_work(data[idx])
```

✅ **Need to handle bounds checking**
```
@cute.kernel
def bounds_check(data: cute.Tensor, n: cutlass.Int32):
    idx = cute.arch.thread_idx()[0]
    
    # Runtime bounds check
    if idx < n:
        process(data[idx])
```

### Use Compile-Time Conditionals When:

✅ **Condition is a configuration parameter**
```
@cute.kernel
def configured_kernel(
    data: cute.Tensor,
    precision_mode: cutlass.Constexpr  # "high" or "low"
):
    if cutlass.const_expr(precision_mode == "high"):
        high_precision_compute(data)
    else:
        low_precision_compute(data)
```

✅ **Want to enable/disable features**
```
@cute.kernel
def optional_features(
    data: cute.Tensor,
    enable_fusion: cutlass.Constexpr,
    enable_tiling: cutlass.Constexpr
):
    # Dead code elimination for disabled features
    if cutlass.const_expr(enable_fusion):
        fused_operation(data)
    
    if cutlass.const_expr(enable_tiling):
        tiled_operation(data)
```

✅ **Architecture-specific code paths**
```
@cute.jit
def arch_specific(data: cute.Tensor, arch: cutlass.Constexpr):
    if cutlass.const_expr(arch == "Ampere"):
        ampere_optimized(data)
    elif cutlass.const_expr(arch == "Hopper"):
        hopper_optimized(data)
    elif cutlass.const_expr(arch == "Blackwell"):
        blackwell_optimized(data)
```

---

## Common Patterns

### Pattern 1: Activation Functions
```
@cute.kernel
def fused_activation(
    data: cute.Tensor,
    activation: cutlass.Constexpr  # "relu", "gelu", "none"
):
    idx = cute.arch.thread_idx()[0]
    val = data[idx]
    
    # Compile-time dispatch - no runtime overhead
    if cutlass.const_expr(activation == "relu"):
        val = cute.where(val > 0, val, 0.0)
    elif cutlass.const_expr(activation == "gelu"):
        # GELU approximation
        val = 0.5 * val * (1.0 + cute.tanh(0.797885 * (val + 0.044715 * val * val * val)))
    # else: no activation (identity)
    
    data[idx] = val
```

### Pattern 2: Precision Selection
```
@cute.kernel
def mixed_precision_compute(
    input: cute.Tensor,
    output: cute.Tensor,
    use_fp16: cutlass.Constexpr
):
    idx = cute.arch.thread_idx()[0]
    
    if cutlass.const_expr(use_fp16):
        # FP16 computation
        val = input[idx].to(cutlass.Float16)
        result = compute_fp16(val)
        output[idx] = result.to(output.element_type)
    else:
        # FP32 computation
        val = input[idx].to(cutlass.Float32)
        result = compute_fp32(val)
        output[idx] = result
```

### Pattern 3: Bounds Checking (Runtime)
```
@cute.kernel
def safe_access(data: cute.Tensor, n: cutlass.Int32):
    idx = cute.arch.block_idx()[0] * cute.arch.block_dim_x() + cute.arch.thread_idx()[0]
    
    # Runtime bounds check
    if idx < n:
        process(data[idx])
```

### Pattern 4: Warp Divergence Mitigation
```
@cute.kernel
def minimize_divergence(data: cute.Tensor, flags: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    
    # All threads evaluate, minimize divergence
    flag = flags[idx]
    
    # Use ternary to reduce divergence
    result = cute.where(flag, compute_a(data[idx]), compute_b(data[idx]))
    data[idx] = result
```

---

## Limitations

### Cannot Use in Dynamic Conditionals:

❌ **return statements**
```
@cute.jit
def bad_return(predicate: cutlass.Boolean):
    if predicate:
        return 42  # ❌ NOT ALLOWED in dynamic if!
```

❌ **raise statements**
```
@cute.jit
def bad_raise(predicate: cutlass.Boolean):
    if predicate:
        raise ValueError("error")  # ❌ NOT ALLOWED!
```

❌ **pass statements**
```
@cute.jit
def bad_pass(predicate: cutlass.Boolean):
    if predicate:
        pass  # ❌ NOT ALLOWED!
```

### Variable Scope Issues:

❌ **Variables defined in if block not available outside**
```
@cute.jit
def bad_scope(predicate: cutlass.Boolean):
    if predicate:
        val = 10
    
    # ❌ ERROR: val not available here!
    cute.printf("%d\n", val)
```

✅ **Workaround: Define before conditional**
```
@cute.jit
def good_scope(predicate: cutlass.Boolean):
    # Define before
    val = 0
    
    if predicate:
        val = 10
    
    # ✅ OK: val was defined before if
    cute.printf("%d\n", val)
```

### Cannot Change Variable Type:

❌ **Type changes not allowed**
```
@cute.jit
def bad_type_change(predicate: cutlass.Boolean):
    val = 10  # Int
    
    if predicate:
        val = 10.0  # ❌ ERROR: changing from int to float!
```

---

## Errors and Common Mistakes

### Mistake 1: Using Runtime Value in const_expr
```
@cute.jit
def wrong_const_expr(dynamic_val: cutlass.Int32):
    # ❌ WRONG: dynamic_val is runtime value!
    if cutlass.const_expr(dynamic_val > 10):
        pass
    
    # ✅ CORRECT: don't use const_expr for runtime values
    if dynamic_val > 10:
        pass
```

### Mistake 2: Expecting Compile-Time Without const_expr
```
@cute.jit
def no_const_expr(flag: cutlass.Constexpr):
    # ⚠️ This is runtime conditional, even though flag is Constexpr!
    if flag:
        path_a()
    else:
        path_b()
    
    # Both branches exist in generated code!
```

**To get compile-time evaluation, must use const_expr:**
```
@cute.jit
def with_const_expr(flag: cutlass.Constexpr):
    # ✅ Compile-time conditional
    if cutlass.const_expr(flag):
        path_a()
    else:
        path_b()
    
    # Only one branch in generated code!
```

### Mistake 3: Forgetting Both Branches May Execute
```
@cute.kernel
def divergent_kernel(data: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    
    # Runtime conditional - may cause warp divergence!
    if idx < 16:
        expensive_operation_a(data)  # Threads 0-15
    else:
        expensive_operation_b(data)  # Threads 16-31
    
    # Threads in same warp executing different paths = slower
```

---

## Performance Considerations

### Warp Divergence

**What is it?**
When threads in the same warp (32 threads) take different branches, execution is serialized.

**Example of bad divergence:**
```
@cute.kernel
def bad_divergence(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Every other thread takes different path
    if tid % 2 == 0:
        expensive_operation_a(data[tid])
    else:
        expensive_operation_b(data[tid])
    
    # Warp executes both branches, half-idle each time!
```

**Better approach:**
```
@cute.kernel
def better_approach(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Use predicated execution instead
    result = cute.where(
        tid % 2 == 0,
        expensive_operation_a(data[tid]),
        expensive_operation_b(data[tid])
    )
    data[tid] = result
    
    # Or structure work so whole warps take same path
```

### Compile-Time Specialization Benefits

**Single kernel with runtime conditional:**
- Code size: both branches
- Performance: branch prediction overhead
- Register usage: max of both paths

**Two kernels with compile-time conditional:**
- Code size: one branch each (2 kernels total)
- Performance: no branching overhead
- Register usage: only what each path needs
- Better occupancy potential

**Trade-off:** More compilation time vs better runtime performance.

---

## Debugging Conditionals

### Print Which Branch Taken
```
@cute.jit
def debug_conditional(x: cutlass.Int32):
    # Compile-time info
    print(f"Compiling conditional for x parameter")
    
    if x > 10:
        # Runtime info
        cute.printf("Taking large branch\n")
        process_large(x)
    else:
        cute.printf("Taking small branch\n")
        process_small(x)
```

### Verify Compile-Time Elimination
```
export CUTE_DSL_PRINT_IR=1

# Then run your code - check generated IR
# const_expr conditionals should show only one branch
```

---

## Summary

**Two conditional types:**
1. **Runtime** - Default, both branches in code
2. **Compile-time** - Use cutlass.const_expr(), only taken branch in code

**Key decisions:**
- Data-dependent? → Runtime conditional
- Configuration-dependent? → Compile-time conditional (const_expr)
- Want dead code elimination? → Compile-time conditional
- Different per thread? → Runtime conditional (watch divergence!)

**Remember:**
- No return/raise/pass in dynamic conditionals
- Define variables before conditionals if needed outside
- Can't change variable types in conditionals
- Runtime value ≠ const_expr compatible

---

## Next Steps

- [Compile-Time Metaprogramming](./metaprogramming.md) - Advanced const_expr patterns
- [Loops](./loops.md) - Loop control flow
- [Performance Tips](../08_optimization/performance_tips.md) - Minimizing divergence

---

## Further Reading

- [CUDA Warp Divergence](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture)
- [Branch Prediction on GPUs](https://developer.nvidia.com/blog/cuda-pro-tip-optimize-for-warp-divergence/)