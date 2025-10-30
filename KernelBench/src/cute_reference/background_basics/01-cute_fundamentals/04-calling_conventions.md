---
topic: "calling_conventions"  
difficulty: "intermediate"
related_topics: ["decorators", "jit_arguments", "code_generation"]
---

# CuTe DSL Calling Conventions

## Overview

Understanding how functions call each other in CuTe DSL is crucial for building complex GPU programs. This document explains the rules and patterns for function calls between Python, @cute.jit functions, and @cute.kernel functions.

**Key principle:** The calling convention determines whether a call happens at **compile time** (inlined) or **runtime** (dynamic GPU launch).

---

## The Three Execution Contexts

### 1. Python Context
- Regular Python code
- Runs on CPU
- Interprets code dynamically
- Can call @cute.jit functions

### 2. Host JIT Context (@cute.jit)
- JIT-compiled functions
- Runs on CPU (host)
- Can orchestrate GPU work
- Can call other @cute.jit functions and launch @cute.kernel functions

### 3. Device Context (@cute.kernel)
- GPU kernel functions
- Runs on GPU (device)
- Massively parallel execution
- Cannot dynamically call other @cute.kernel functions

---

## Calling Convention Rules

### Rule Table

| Caller | Callee | Allowed | How It Works |
|--------|--------|---------|--------------|
| Python | @cute.jit | ✅ Yes | DSL runtime invocation |
| Python | @cute.kernel | ❌ No | Must use .launch() from @cute.jit |
| @cute.jit | @cute.jit | ✅ Yes | Compile-time call, inlined |
| @cute.jit | Python function | ✅ Yes | Compile-time call, inlined |
| @cute.jit | @cute.kernel | ✅ Yes | Runtime GPU launch via driver |
| @cute.kernel | @cute.jit | ✅ Yes | Compile-time call, inlined |
| @cute.kernel | Python function | ✅ Yes | Compile-time call, inlined |
| @cute.kernel | @cute.kernel | ❌ No | Dynamic parallelism not supported |

---

## Detailed Calling Patterns

### Pattern 1: Python → @cute.jit

**Most common entry point** - Python code calls JIT-compiled host function.
```
@cute.jit
def host_function(x: cute.Tensor, y: cute.Tensor):
    # Process on CPU
    result = x + y
    return result

# Call from Python
import torch
x = torch.randn(100, device="cuda")
y = torch.randn(100, device="cuda")

result = host_function(
    from_dlpack(x), 
    from_dlpack(y)
)  # ✅ Works!
```

**What happens:**
1. CuTe DSL runtime receives Python call
2. Converts Python arguments to C ABI types
3. JIT compiles function (or uses cached version)
4. Executes compiled function
5. Returns result to Python

---

### Pattern 2: Python → @cute.kernel (FORBIDDEN)

**You cannot call kernels directly from Python.**
```
@cute.kernel
def my_kernel(data: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    data[idx] = data[idx] * 2

# ❌ WRONG - This will error!
my_kernel(cute_tensor)

# ✅ CORRECT - Launch from @cute.jit
@cute.jit
def wrapper(data: cute.Tensor):
    my_kernel.launch(
        grid=[1, 1, 1],
        block=[256, 1, 1]
    )(data)

wrapper(cute_tensor)  # ✅ Works!
```

**Why?** Kernels require grid/block configuration which must be specified via .launch().

---

### Pattern 3: @cute.jit → @cute.jit (Inlined)

**JIT functions can call other JIT functions** - these calls are inlined at compile time.
```
@cute.jit
def helper_function(x, y):
    return x * y + 1

@cute.jit
def main_function(a, b, c):
    # Call helper function - inlined at compile time
    temp = helper_function(a, b)
    result = temp + c
    return result

# Call from Python
result = main_function(x, y, z)
```

**What happens:**
- `helper_function` is inlined into `main_function`
- No runtime function call overhead
- Single compiled function in the end

---

### Pattern 4: @cute.jit → Python Function (Inlined)

**JIT functions can call pure Python functions** - executed at compile time.
```
def pure_python_helper(shape):
    """Regular Python function"""
    return shape[0] * shape[1]

@cute.jit
def jit_function(tensor: cute.Tensor):
    # Call Python function at compile time
    total_size = pure_python_helper(tensor.shape)
    
    # Use the result in JIT code
    cute.printf("Total size: %d\n", total_size)

jit_function(my_tensor)
```

**What happens:**
- Python function executes during JIT compilation
- Result is baked into the compiled code
- No runtime Python call

**Limitation:** The Python function cannot access runtime values (only compile-time constants).

---

### Pattern 5: @cute.jit → @cute.kernel (GPU Launch)

**The primary way to launch GPU kernels** - runtime dispatch.
```
@cute.kernel
def gpu_kernel(input: cute.Tensor, output: cute.Tensor):
    idx = cute.arch.thread_idx()[0] + cute.arch.block_idx()[0] * cute.arch.block_dim_x()
    if idx < cute.size(output):
        output[idx] = input[idx] * 2

@cute.jit
def orchestrator(data: cute.Tensor):
    # Allocate output
    output = cute.make_tensor(data.shape, data.element_type)
    
    # Calculate launch configuration
    threads_per_block = 256
    num_blocks = (cute.size(data) + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    gpu_kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[threads_per_block, 1, 1]
    )(data, output)
    
    return output

# Call from Python
result = orchestrator(my_data)
```

**What happens:**
1. JIT function executes on CPU
2. Prepares kernel launch parameters
3. Issues GPU kernel launch via CUDA driver
4. Kernel executes on GPU
5. Control returns to JIT function

---

### Pattern 6: @cute.kernel → @cute.jit (Inlined)

**Kernels can call JIT helper functions** - inlined at compile time.
```
@cute.jit
def compute_offset(row, col, width):
    """Helper function callable from kernel"""
    return row * width + col

@cute.kernel
def matrix_kernel(matrix: cute.Tensor):
    row = cute.arch.block_idx()[0]
    col = cute.arch.thread_idx()[0]
    
    # Call JIT function - inlined
    offset = compute_offset(row, col, cute.size(matrix, mode=1))
    
    # Use the result
    matrix[offset] = matrix[offset] + 1
```

**What happens:**
- `compute_offset` is inlined into the kernel
- No function call overhead
- Single monolithic kernel in the end

---

### Pattern 7: @cute.kernel → Python Function (Compile-Time)

**Kernels can call pure Python functions** for compile-time computation.
```
def calculate_tile_size(shape):
    """Pure Python - runs at compile time"""
    if shape[0] >= 128:
        return 128
    else:
        return 64

@cute.kernel
def tiled_kernel(data: cute.Tensor, shape: cutlass.Constexpr):
    # Call Python function at compile time
    tile_size = calculate_tile_size(shape)
    
    # Use compile-time constant
    for i in cutlass.range_constexpr(tile_size):
        # Process tile
        pass
```

**Key:** The Python function executes during compilation, not at runtime.

---

### Pattern 8: @cute.kernel → @cute.kernel (FORBIDDEN)

**Kernels cannot call other kernels dynamically.**
```
@cute.kernel
def kernel_a(data):
    pass

@cute.kernel
def kernel_b(data):
    # ❌ WRONG - Cannot call another kernel!
    kernel_a.launch(...)(data)  # Error!

# ✅ CORRECT - Launch both from JIT function
@cute.jit
def wrapper(data):
    kernel_a.launch(...)(data)
    kernel_b.launch(...)(data)
```

**Why?** CuTe DSL doesn't support CUDA Dynamic Parallelism. All kernel launches must come from the host.

---

## Argument Passing Conventions

### Static vs Dynamic Arguments

#### Static Arguments (cutlass.Constexpr)

**Known at compile time**, baked into the kernel.
```
@cute.kernel
def kernel_with_static_arg(
    data: cute.Tensor,
    tile_size: cutlass.Constexpr  # Static!
):
    # tile_size is known at compile time
    for i in cutlass.range_constexpr(tile_size):
        # Compiler can fully unroll this loop
        process(data[i])

# Launch with different static values
kernel_with_static_arg.launch(
    grid=[...], block=[...]
)(data, tile_size=128)  # Compiles version with tile_size=128

kernel_with_static_arg.launch(
    grid=[...], block=[...]
)(data, tile_size=64)   # Compiles different version with tile_size=64
```

**Benefits:**
- Enables compile-time optimization
- Loop unrolling
- Dead code elimination
- Constant propagation

#### Dynamic Arguments (Default)

**Known only at runtime**, passed to kernel at launch.
```
@cute.kernel
def kernel_with_dynamic_arg(
    data: cute.Tensor,
    scale: cutlass.Float32  # Dynamic!
):
    idx = cute.arch.thread_idx()[0]
    # scale is a runtime value
    data[idx] = data[idx] * scale

# Same compiled kernel handles any scale value
kernel.launch(...)(data, 2.0)
kernel.launch(...)(data, 3.5)
```

**Benefits:**
- Single kernel handles any value
- Less compilation overhead
- Smaller code size

---

## Passing Tensors Between Functions

### By Reference (Zero-Copy)

Tensors are **always passed by reference** - no copying occurs.
```
@cute.jit
def modify_tensor(tensor: cute.Tensor):
    # Modifies original tensor
    tensor[0] = 42

tensor = cute.make_tensor(cute.make_shape(10), cutlass.Float32)
modify_tensor(tensor)
# tensor[0] is now 42!
```

### Sharing Tensors with Kernels
```
@cute.kernel
def kernel_modifies(data: cute.Tensor):
    idx = cute.arch.thread_idx()[0]
    data[idx] = idx

@cute.jit
def host_function(data: cute.Tensor):
    # Pass tensor to kernel - same memory
    kernel_modifies.launch(
        grid=[1, 1, 1],
        block=[cute.size(data), 1, 1]
    )(data)
    
    # data has been modified by kernel
    return data
```

---

## Synchronization Between Calls

### Implicit Synchronization

Kernel launches from the **same stream** are implicitly ordered.
```
@cute.jit
def sequential_kernels(data):
    kernel_a.launch(...)(data)
    kernel_b.launch(...)(data)  # Waits for kernel_a to finish
```

### Explicit Synchronization

Use CUDA streams for explicit control.
```
import cuda.bindings.driver as cuda

@cute.jit
def async_kernels(data, stream1: cuda.CUstream, stream2: cuda.CUstream):
    # Launch in different streams - may overlap!
    kernel_a.launch(..., stream=stream1)(data)
    kernel_b.launch(..., stream=stream2)(data)
    
    # Explicit sync if needed
    cuda.cuStreamSynchronize(stream1)
    cuda.cuStreamSynchronize(stream2)
```

---

## Common Patterns

### Pattern A: Multi-Kernel Pipeline
```
@cute.kernel
def preprocess(input, temp):
    # Preprocessing kernel
    pass

@cute.kernel  
def compute(temp, output):
    # Main computation kernel
    pass

@cute.kernel
def postprocess(output):
    # Postprocessing kernel
    pass

@cute.jit
def pipeline(input):
    temp = cute.make_tensor(...)
    output = cute.make_tensor(...)
    
    # Sequential pipeline
    preprocess.launch(...)(input, temp)
    compute.launch(...)(temp, output)
    postprocess.launch(...)(output)
    
    return output
```

### Pattern B: Conditional Kernel Selection
```
@cute.kernel
def small_kernel(data):
    # Optimized for small inputs
    pass

@cute.kernel
def large_kernel(data):
    # Optimized for large inputs
    pass

@cute.jit
def adaptive_dispatch(data):
    size = cute.size(data)
    
    if size < 1000:
        small_kernel.launch(...)(data)
    else:
        large_kernel.launch(...)(data)
```

### Pattern C: Iterative Kernel Calls
```
@cute.kernel
def iterative_step(data, iteration):
    # One iteration of algorithm
    pass

@cute.jit
def iterative_solver(data, num_iterations):
    for i in range(num_iterations):
        iterative_step.launch(...)(data, i)
```

---

## Performance Considerations

### Inlining vs Function Calls

**Inlined calls (JIT ↔ JIT, Kernel ↔ JIT):**
- ✅ Zero overhead
- ✅ Enables cross-function optimization
- ⚠️ Larger code size

**Separate kernel launches:**
- ⚠️ Launch overhead (~1-5 microseconds)
- ✅ Smaller code size
- ✅ Can run on different streams

### Minimizing Launch Overhead
```
# ❌ BAD - Many small kernel launches
@cute.jit
def inefficient(data):
    for i in range(1000):
        tiny_kernel.launch(...)(data, i)  # 1000 launches!

# ✅ GOOD - One kernel does all iterations
@cute.kernel
def efficient_kernel(data, iterations):
    for i in range(iterations):
        # Process inside kernel
        pass

@cute.jit
def efficient(data):
    efficient_kernel.launch(...)(data, 1000)  # 1 launch!
```

---

## Debugging Call Chains

### Enable Call Tracing
```
export CUTE_DSL_LOG_TO_CONSOLE=1
export CUTE_DSL_LOG_LEVEL=10  # Debug level
```

### Print Call Stack
```
@cute.jit
def debug_caller():
    print("In debug_caller")  # Compile-time print
    helper()

@cute.jit
def helper():
    print("In helper")  # Compile-time print
    cute.printf("Helper executing\n")  # Runtime print

debug_caller()
```

---

## Summary

**Key takeaways:**
- Python calls @cute.jit functions directly
- Python CANNOT call @cute.kernel directly (must use .launch())
- JIT functions can launch kernels with .launch()
- JIT ↔ JIT and Kernel ↔ JIT calls are inlined (zero overhead)
- Kernel ↔ Kernel calls are NOT supported
- Tensors pass by reference (no copying)
- Static arguments enable compile-time optimization
- Dynamic arguments provide runtime flexibility

**Decision tree:**
```
Need to call from Python?
  → Use @cute.jit wrapper

Need parallel GPU execution?
  → Use @cute.kernel with .launch()

Need helper logic?
  → Use @cute.jit (can be called from both contexts)

Need multiple kernels?
  → Orchestrate from @cute.jit wrapper
```

---

## Next Steps

- [JIT Caching](../06_compilation/jit_caching.md) - How compiled functions are cached
- [Control Flow](../02_control_flow/loops.md) - Flow control in different contexts
- [Examples](../10_examples/) - Real-world calling patterns

---

## Further Reading

- [CUDA Programming Guide - Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels)
- [CuTe DSL Source Code](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass/cute)