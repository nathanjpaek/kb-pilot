---
topic: "type_safety"
difficulty: "intermediate"
related_topics: ["jit_compilation", "data_types", "code_generation"]
---

# Type Safety in CuTe DSL

## Overview

CuTe DSL uses type annotations to ensure correctness and enable optimization. Strong typing helps catch errors at compile time rather than runtime, and allows the compiler to generate optimal code.

**Key concepts:**
- Type annotations for parameters
- Type inference
- Type checking at compile time
- Runtime type validation
- Common type errors and fixes
- Benefits of strong typing

---

## Why Type Safety Matters

### The Problem Without Types

**Weakly typed (Python-like):**
```python
# Hypothetical weak typing (NOT CuTe DSL)
@cute.kernel
def bad_kernel(data, n, threshold):  # No types!
    # What are the types?
    # - data: Tensor? List? Float?
    # - n: Int? Float? 
    # - threshold: Float? Int?
    
    if data[n] > threshold:  # May fail at runtime
        data[n] = threshold
```

**Problems:**
- Errors only caught at runtime
- Poor performance (type checks at runtime)
- Hard to optimize
- Unclear API

### With Type Annotations

**Strongly typed (CuTe DSL):**
```python
@cute.kernel
def good_kernel(
    data: cute.Tensor,
    n: cutlass.Int32,
    threshold: cutlass.Float32
):
    # Types explicit and checked
    # Compiler knows exact types
    # Errors caught at compile time
    
    if data[n] > threshold:
        data[n] = threshold
```

**Benefits:**
- Compile-time error checking
- Optimal code generation
- Self-documenting code
- Better IDE support

---

## Type Annotation Syntax

### Basic Types

**CuTe tensor types:**
```python
@cute.kernel
def tensor_types(
    tensor: cute.Tensor,           # Generic tensor
    tensor_2d: cute.Tensor,        # Can be any rank
    tensor_3d: cute.Tensor,        # Shape not specified in type
):
    pass
```

**Scalar types:**
```python
@cute.kernel
def scalar_types(
    i8: cutlass.Int8,              # 8-bit signed integer
    i16: cutlass.Int16,            # 16-bit signed integer
    i32: cutlass.Int32,            # 32-bit signed integer
    i64: cutlass.Int64,            # 64-bit signed integer
    
    u8: cutlass.UInt8,             # 8-bit unsigned integer
    u16: cutlass.UInt16,           # 16-bit unsigned integer
    u32: cutlass.UInt32,           # 32-bit unsigned integer
    u64: cutlass.UInt64,           # 64-bit unsigned integer
    
    f16: cutlass.Float16,          # 16-bit float (half precision)
    f32: cutlass.Float32,          # 32-bit float (single precision)
    f64: cutlass.Float64,          # 64-bit float (double precision)
    
    bf16: cutlass.BFloat16,        # 16-bit bfloat16
    
    flag: cutlass.Bool,            # Boolean
):
    pass
```

**Compile-time constants:**
```python
@cute.kernel
def constexpr_types(
    size: cutlass.Constexpr,       # Compile-time integer constant
    flag: cutlass.Constexpr,       # Compile-time boolean constant
):
    pass
```

### Type Annotations Are Required

**✗ Missing annotations cause errors:**
```python
@cute.kernel
def missing_types(data, n):  # ERROR: No type annotations!
    data[n] = 0.0

# CompileError: Parameter 'data' missing type annotation
# CompileError: Parameter 'n' missing type annotation
```

**✓ Always annotate:**
```python
@cute.kernel
def with_types(data: cute.Tensor, n: cutlass.Int32):
    data[n] = 0.0
```

---

## Type Inference

### Return Type Inference

**Return types are inferred:**
```python
@cute.kernel
def inferred_return(data: cute.Tensor) -> None:  # Optional annotation
    data[0] = 1.0
    # Return type inferred as None

@cute.kernel  
def auto_infer(data: cute.Tensor):  # No return annotation
    data[0] = 1.0
    # Still inferred as None
```

### Local Variable Inference

**Local variable types inferred from usage:**
```python
@cute.kernel
def local_inference(data: cute.Tensor):
    # Type inferred from literal
    x = 1.0  # cutlass.Float32
    y = 42   # cutlass.Int32
    
    # Type inferred from expression
    z = data[0]  # Same type as data elements
    
    # Type inferred from operation
    result = x + y  # cutlass.Float32 (promotion)
```

---

## Type Checking

### Compile-Time Checking

**Type errors caught at compilation:**
```python
@cute.kernel
def type_mismatch(data: cute.Tensor, n: cutlass.Float32):  # n is float
    # ERROR: Can't use float as array index
    value = data[n]  # Compile error!

# TypeError: Expected Int32 for array index, got Float32
```

**Fix:**
```python
@cute.kernel
def type_correct(data: cute.Tensor, n: cutlass.Int32):  # n is int
    value = data[n]  # OK
```

### Assignment Type Checking

**Type compatibility checked:**
```python
@cute.kernel
def assignment_check(
    data_f32: cute.Tensor,  # Float32 tensor
    data_f16: cute.Tensor,  # Float16 tensor
):
    # OK: Same type
    data_f32[0] = 1.0
    
    # OK: Implicit conversion
    data_f16[0] = 1.0  # Float32 literal converts to Float16
    
    # ERROR: Can't assign tensor to scalar
    x: cutlass.Float32 = data_f32  # Compile error!
```

### Function Call Type Checking

**Argument types must match:**
```python
@cute.kernel
def callee(x: cutlass.Float32):
    return x * 2.0

@cute.kernel
def caller(data: cute.Tensor):
    # OK: Correct type
    result = callee(3.14)
    
    # ERROR: Wrong type
    result = callee(42)  # Int passed to Float32 parameter
```

---

## Type Conversions

### Explicit Casts

**Convert between types:**
```python
@cute.kernel
def explicit_cast(data: cute.Tensor):
    i: cutlass.Int32 = 42
    
    # Explicit cast to float
    f: cutlass.Float32 = cutlass.Float32(i)
    
    # Explicit cast to other int sizes
    i8: cutlass.Int8 = cutlass.Int8(i)
    i64: cutlass.Int64 = cutlass.Int64(i)
```

### Implicit Conversions

**Some conversions happen automatically:**
```python
@cute.kernel
def implicit_conversion(data: cute.Tensor):
    # Int to Float (widening)
    i: cutlass.Int32 = 42
    f: cutlass.Float32 = i  # OK: Implicit conversion
    
    # Float literal to typed variable
    x: cutlass.Float16 = 3.14  # OK: Converts to Float16
```

### Unsafe Conversions

**Some conversions may lose precision:**
```python
@cute.kernel
def precision_loss(data: cute.Tensor):
    # Float to Int (truncation)
    f: cutlass.Float32 = 3.14
    i: cutlass.Int32 = cutlass.Int32(f)  # i = 3 (truncated)
    
    # Float32 to Float16 (precision loss)
    f32: cutlass.Float32 = 1.23456789
    f16: cutlass.Float16 = cutlass.Float16(f32)  # Precision lost
    
    # Int64 to Int32 (overflow possible)
    big: cutlass.Int64 = 2147483648
    small: cutlass.Int32 = cutlass.Int32(big)  # Overflow!
```

---

## Tensor Types

### Generic Tensor Type

**`cute.Tensor` is generic:**
```python
@cute.kernel
def generic_tensor(data: cute.Tensor):
    # Tensor element type inferred from actual argument
    # Can be Float16, Float32, Int32, etc.
    value = data[0]  # Type matches tensor element type
```

### Runtime Type Information

**Tensor carries type at runtime:**
```python
@cute.jit
def check_tensor_type(tensor: torch.Tensor):
    # PyTorch tensor has dtype
    print(f"Tensor dtype: {tensor.dtype}")
    
    # Convert to CuTe tensor
    cute_tensor = cute.from_dlpack(tensor)
    
    # Launch kernel
    my_kernel.launch(...)(cute_tensor)
    # Kernel compiled for specific dtype
```

### Type Specialization

**Different dtypes trigger different kernels:**
```python
@cute.kernel
def specialized(data: cute.Tensor):
    data[0] = data[0] * 2.0

# Kernel compiled separately for each dtype
fp32_tensor = torch.randn(1024, dtype=torch.float32, device="cuda")
specialized.launch(...)(cute.from_dlpack(fp32_tensor))  # Kernel A

fp16_tensor = torch.randn(1024, dtype=torch.float16, device="cuda")
specialized.launch(...)(cute.from_dlpack(fp16_tensor))  # Kernel B (different!)
```

---

## Constexpr Type Safety

### Compile-Time vs Runtime

**Type determines when value is known:**
```python
@cute.kernel
def constexpr_vs_runtime(
    runtime_n: cutlass.Int32,      # Value known at runtime
    compile_n: cutlass.Constexpr   # Value known at compile time
):
    # OK: Runtime loop
    for i in range(runtime_n):
        process(i)
    
    # OK: Compile-time loop
    for i in cutlass.range_constexpr(compile_n):
        process(i)
    
    # ERROR: Can't use runtime value in compile-time context
    for i in cutlass.range_constexpr(runtime_n):  # Type error!
        process(i)
```

### Constexpr Validation

**Constexpr values validated at call site:**
```python
@cute.kernel
def requires_constexpr(n: cutlass.Constexpr):
    for i in cutlass.range_constexpr(n):
        process(i)

# OK: Literal is Constexpr
requires_constexpr.launch(...)(n=128)

# ERROR: Runtime variable is not Constexpr
size = 128
requires_constexpr.launch(...)(n=size)  # Type error!
```

---

## Common Type Errors

### Error 1: Missing Type Annotation

**Problem:**
```python
@cute.kernel
def missing_type(data):  # ERROR: No annotation
    data[0] = 1.0
```

**Error message:**
```
TypeError: Parameter 'data' is missing type annotation
```

**Fix:**
```python
@cute.kernel
def with_type(data: cute.Tensor):
    data[0] = 1.0
```

### Error 2: Type Mismatch

**Problem:**
```python
@cute.kernel
def mismatch(data: cute.Tensor, n: cutlass.Float32):
    # ERROR: Float used as index
    value = data[n]
```

**Error message:**
```
TypeError: Expected Int32 for array index, got Float32
```

**Fix:**
```python
@cute.kernel
def correct(data: cute.Tensor, n: cutlass.Int32):
    value = data[n]
```

### Error 3: Constexpr Violation

**Problem:**
```python
@cute.kernel
def constexpr_error(n: cutlass.Constexpr):
    # ERROR: Can't modify Constexpr
    n = n + 1
```

**Error message:**
```
TypeError: Cannot assign to compile-time constant
```

**Fix:**
```python
@cute.kernel
def constexpr_correct(n: cutlass.Constexpr):
    # Create new variable
    n_plus_1 = n + 1  # OK: New compile-time constant
```

### Error 4: Wrong Tensor Type

**Problem:**
```python
@cute.kernel
def wrong_tensor(data: torch.Tensor):  # ERROR: Use cute.Tensor
    data[0] = 1.0
```

**Error message:**
```
TypeError: Expected cute.Tensor, got torch.Tensor
```

**Fix:**
```python
@cute.kernel
def correct_tensor(data: cute.Tensor):
    data[0] = 1.0
```

### Error 5: Return Type Error

**Problem:**
```python
@cute.kernel
def bad_return(data: cute.Tensor) -> cutlass.Float32:
    data[0] = 1.0
    # ERROR: No return statement, but declared Float32 return
```

**Error message:**
```
TypeError: Function declared to return Float32 but returns None
```

**Fix:**
```python
@cute.kernel
def good_return(data: cute.Tensor) -> None:
    data[0] = 1.0
    # OK: Matches declared return type
```

---

## Type Safety Best Practices

### ✅ DO

**Always annotate kernel parameters:**
```python
@cute.kernel
def kernel(
    data: cute.Tensor,
    n: cutlass.Int32,
    threshold: cutlass.Float32
):
    pass
```

**Use specific scalar types:**
```python
# ✓ GOOD: Explicit type
size: cutlass.Int32 = 1024

# ✗ BAD: Python int (may be inferred incorrectly)
size = 1024
```

**Use Constexpr for compile-time values:**
```python
@cute.kernel
def kernel(tile_size: cutlass.Constexpr):  # Compile-time
    for i in cutlass.range_constexpr(tile_size):
        process(i)
```

**Document expected tensor dtypes:**
```python
@cute.kernel
def fp16_kernel(data: cute.Tensor):
    """
    Kernel expecting Float16 tensors.
    
    Args:
        data: cute.Tensor with dtype Float16
    """
    pass
```

### ❌ DON'T

**Don't skip type annotations:**
```python
# ✗ BAD
@cute.kernel
def kernel(data, n):  # Missing types!
    pass
```

**Don't mix Python and CuTe types in kernel:**
```python
# ✗ BAD
@cute.kernel
def kernel(data: cute.Tensor, n: int):  # Use cutlass.Int32!
    pass
```

**Don't ignore type errors:**
```python
# ✗ BAD: Suppressing type error
@cute.kernel
def kernel(data: cute.Tensor, n: cutlass.Float32):
    value = data[int(n)]  # Workaround for type error - FIX THE TYPE!
```

**Don't use implicit conversions for precision:**
```python
# ✗ BAD: Precision loss
f32: cutlass.Float32 = 3.141592653589793
f16: cutlass.Float16 = f32  # Loses precision silently

# ✓ GOOD: Explicit cast shows intent
f16: cutlass.Float16 = cutlass.Float16(f32)  # Clear precision loss
```

---

## Type Checking Tools

### Static Type Checking

**Check types before running:**
```python
# mypy or similar (if supported)
# $ mypy my_kernel.py

@cute.kernel
def kernel(data: cute.Tensor, n: cutlass.Int32):
    pass

# Type checker validates:
# - All parameters annotated
# - Types used correctly
# - Return types match
```

### Runtime Type Validation

**Validate tensor types at runtime:**
```python
@cute.jit
def validate_and_launch(tensor: torch.Tensor):
    # Check tensor dtype
    if tensor.dtype != torch.float16:
        raise TypeError(f"Expected float16, got {tensor.dtype}")
    
    # Check tensor device
    if not tensor.is_cuda:
        raise TypeError("Expected CUDA tensor")
    
    # Now safe to launch
    cute_tensor = cute.from_dlpack(tensor)
    my_kernel.launch(...)(cute_tensor)
```

---

## Advanced Type Features

### Type Aliases

**Create readable type aliases:**
```python
# Type aliases for clarity
TileSize = cutlass.Constexpr
Index = cutlass.Int32
Accumulator = cutlass.Float32

@cute.kernel
def kernel(
    data: cute.Tensor,
    tile: TileSize,
    idx: Index,
    acc: Accumulator
):
    pass
```

### Generic Functions (if supported)

**Parameterized types:**
```python
# Hypothetical generic syntax
@cute.kernel
def generic_kernel[T](data: cute.Tensor[T], value: T):
    data[0] = value

# Specialized at call site
generic_kernel[cutlass.Float32].launch(...)(float32_tensor, 1.0)
generic_kernel[cutlass.Float16].launch(...)(float16_tensor, 1.0)
```

---

## Debugging Type Issues

### Verbose Type Checking

**Enable type checking output:**
```python
import os
os.environ['CUTLASS_VERBOSE_TYPES'] = '1'

import cutlass.cute as cute

@cute.kernel
def kernel(data: cute.Tensor, n: cutlass.Int32):
    pass

# Prints:
# Type checking kernel 'kernel':
#   - Parameter 'data': cute.Tensor
#   - Parameter 'n': cutlass.Int32
#   - Return type: None
# Type checking passed ✓
```

### Type Error Messages

**Understanding error messages:**
```python
@cute.kernel
def problematic(data: cute.Tensor, n: cutlass.Float32):
    value = data[n]  # Type error

# Error message breakdown:
# TypeError: Expected Int32 for array index, got Float32
#   File "my_kernel.py", line 3, in problematic
#     value = data[n]
#                  ^
#   Parameter 'n' has type Float32
#   Array index requires Int32
#   
# Suggestion: Change n to cutlass.Int32 or cast: data[cutlass.Int32(n)]
```

---

## Summary

**Type safety in CuTe DSL:**
- All kernel parameters must have type annotations
- Types checked at compile time
- Errors caught before execution
- Enables optimization

**Key types:**
- `cute.Tensor`: Generic tensor
- `cutlass.Int32`, `cutlass.Float32`: Scalar types
- `cutlass.Constexpr`: Compile-time constants

**Type checking:**
- Parameters: Required annotations
- Locals: Inferred from usage
- Returns: Inferred from function
- Conversions: Explicit casts preferred

**Benefits:**
- Earlier error detection
- Better performance
- Self-documenting code
- IDE support

**Best practices:**
- Always annotate parameters
- Use specific types (not Python int/float)
- Use Constexpr for compile-time values
- Prefer explicit casts for conversions

---

## Next Steps

- [JIT Compilation](./jit_compilation.md) - How types are compiled
- [Data Types](../01_fundamentals/data_types.md) - Available types
- [Static vs Dynamic](./static_vs_dynamic.md) - Constexpr details

---

## Further Reading

- [Type Systems](https://en.wikipedia.org/wiki/Type_system)
- [Static Type Checking](https://en.wikipedia.org/wiki/Type_system#Static_type_checking)
- [CUDA Type System](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-types)