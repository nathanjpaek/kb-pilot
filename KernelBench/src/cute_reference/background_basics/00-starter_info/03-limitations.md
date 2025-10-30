
---
topic: "limitations"
difficulty: "beginner"
related_topics: ["control_flow", "debugging"]
---

# CuTe DSL Limitations

## Overview

While CuTe DSL is powerful, it has intentional limitations to maintain safety, predictability, and performance. Understanding these constraints will help you write correct code and avoid frustrating debugging sessions.

---

## Dynamic Control Flow Limitations

### 1. No Early Exit from Loops

❌ **NOT ALLOWED**: break, continue, return, pass inside dynamic loops

**Why**: Dynamic loops are lowered to structured IR that doesn't support arbitrary control flow exits.

**Workaround**: Use conditional logic instead of early exits.

### 2. Variables Defined in Control Flow Are Not Available Outside

❌ **NOT ALLOWED**: Using variables first defined inside dynamic if/loop outside that block

**Why**: Dynamic control flow regions are isolated - variables defined inside them don't escape.

**Workaround**: Define variables before the control flow.

### 3. Cannot Change Variable Types in Control Flow

❌ **NOT ALLOWED**: Reassigning a variable to a different type

**Why**: Type stability is required for IR generation.

**Workaround**: Use different variables or ensure type consistency.

### 4. No Exceptions in Dynamic Control Flow

❌ **NOT ALLOWED**: Raising exceptions in dynamic control flow

**Workaround**: Use assertions or handle errors differently.

---

## Compile-Time vs Runtime Restrictions

### Cannot Mix Compile-Time and Runtime Values

❌ **NOT ALLOWED**: Using runtime values where compile-time constants expected

**Why**: range_constexpr requires compile-time known bounds.

**Solution**: Use range() for runtime values, range_constexpr() for compile-time values.

---

## Memory and Pointer Limitations

### 1. Limited Python Object Support

CuTe DSL doesn't support arbitrary Python objects in kernels.

❌ **NOT ALLOWED**: Python lists, dictionaries, or complex objects

✅ **ALLOWED**: CuTe tensors, numeric types, booleans

### 2. No Dynamic Memory Allocation in Kernels

❌ **NOT ALLOWED**: Creating new tensors inside kernels

**Why**: GPU kernels cannot perform dynamic memory allocation.

**Solution**: Allocate in host code, pass to kernel.

---

## Type System Limitations

### 1. Limited Type Inference in Some Cases

While CuTe DSL infers many types automatically, explicit annotation may be needed for intermediate values.

### 2. No Automatic Broadcasting (Like NumPy)

CuTe DSL does not automatically broadcast operations. Shapes must match exactly.

**Solution**: Manually handle broadcasting in your kernel logic.

---

## Preprocessor Mode Limitations

### When preprocessor=False (Tracing Mode)

**Severe limitations** - avoid unless you know what you're doing:

1. **Branches collapse**: Only the executed branch is captured
2. **Loops unroll**: Loops execute only once during trace
3. **Data-dependent control flow breaks**: Cannot handle runtime decisions

**Recommendation**: Use default preprocessor=True unless you have a specific reason and understand the implications.

---

## Architecture-Specific Limitations

### Tensor Core Requirements

Certain MMA operations require specific:
- Data types (FP16, BF16, FP8, INT8)
- Tile sizes (multiples of MMA atom dimensions)
- Memory layouts (specific strides)

### Shared Memory Limits

Each architecture has shared memory limits per block:
- Ampere: 164KB max
- Hopper: 232KB max  
- Blackwell: Check specs

---

## Threading and Synchronization Limitations

### 1. Block Size Limits

Maximum threads per block is architecture dependent (typically 1024).

### 2. Grid Size Limits

CUDA has maximum grid dimensions:
- X, Y: 2^31 - 1
- Z: 65535

Usually not a problem, but be aware for very large problems.

---

## Debugging Limitations

### 1. Limited Python Debugging Tools

Standard Python debuggers (pdb, ipdb) don't work inside kernels.

**Solution**: Use cute.printf() for debugging.

### 2. Error Messages Can Be Cryptic

Compilation errors sometimes reference MLIR internals.

**Solution**: 
- Enable logging: export CUTE_DSL_LOG_LEVEL=10
- Dump IR: export CUTE_DSL_PRINT_IR=1
- Simplify code to isolate issue

---

## Performance Limitations

### 1. JIT Compilation Overhead

First call to a function incurs compilation cost (can be ~100ms). Subsequent calls are fast (~1ms) due to caching.

**Mitigation**: Use cute.compile() for explicit control.

### 2. Dynamic Shapes May Be Slower

Dynamic layouts add runtime overhead compared to fully static layouts.

**Trade-off**: Flexibility vs performance

---

## Language Feature Limitations

### Not Supported in Kernels

- ❌ Classes and OOP features
- ❌ Generators and yield
- ❌ async/await
- ❌ Context managers (with statements)
- ❌ Decorators inside kernels
- ❌ Lambda functions
- ❌ List/dict comprehensions

### Supported

- ✅ Basic arithmetic and math operations
- ✅ Loops (for, while)
- ✅ Conditionals (if/else)
- ✅ Function calls to other @jit or @kernel functions
- ✅ CuTe operations (copy, gemm, etc.)
- ✅ Tensor indexing and slicing

---

## Working Within Limitations

### Best Practices

1. **Start simple**: Get basic version working before optimizing
2. **Use default settings**: preprocessor=True unless you have a reason
3. **Validate early**: Check shapes, types, and bounds at host level
4. **Profile first**: Don't optimize prematurely
5. **Read error messages carefully**: They often point to the exact limitation

### When You Hit a Limitation

1. **Check documentation**: Is there a supported pattern for what you need?
2. **Simplify**: Can you restructure your code to avoid the limitation?
3. **Ask for help**: GitHub issues, NVIDIA forums
4. **Consider alternatives**: Maybe raw CUDA or Triton is better for your use case

---

## Summary Table

| Feature | Supported | Notes |
|---------|-----------|-------|
| Basic loops (for, while) | ✅ Yes | With preprocessor=True |
| Early exit (break, continue) | ❌ No | Use conditionals instead |
| Dynamic memory allocation | ❌ No | Allocate in host code |
| cute.printf() | ✅ Yes | Performance impact |
| Python print() | ✅ Yes | Compile-time only |
| Exceptions in kernels | ❌ No | Use assertions |
| Type changes | ❌ No | Maintain type stability |
| Tensor cores | ✅ Yes | Architecture-specific |
| Multi-GPU | ⚠️ Partial | Manual management |
| Debugging tools | ⚠️ Limited | Use printf, logging |

---

## Next Steps

- [Control Flow](../02_control_flow/loops.md) - Learn what IS supported
- [Debugging Guide](../09_debugging/debugging_overview.md) - How to debug within limitations
- [Examples](../10_examples/) - See patterns that work

---

## Getting Help
If you're unsure whether something is supported:
1. Look at [official examples](https://github.com/NVIDIA/cutlass/tree/main/examples/cute_dsl) -- found in examples folder
2. Try it! Compilation errors will tell you quickly most probably!
3. Check [NVIDIA CUTLASS GitHub](https://github.com/NVIDIA/cutlass/issues)