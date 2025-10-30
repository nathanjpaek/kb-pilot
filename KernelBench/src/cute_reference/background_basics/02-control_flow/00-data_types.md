---
topic: "data_types"
difficulty: "beginner"
related_topics: ["tensors", "layouts", "mma_atoms"]
---

# CuTe DSL Data Types

## Overview

CuTe DSL provides native GPU data types that map directly to hardware-supported formats. Understanding these types is crucial for:
- Tensor core operations (MMA)
- Memory bandwidth optimization
- Numerical precision trade-offs
- Architecture compatibility

---

## Native CuTe Types

All CuTe types are in the `cutlass` namespace:
```
import cutlass

# Integer types
cutlass.Int8
cutlass.Int16
cutlass.Int32
cutlass.Int64
cutlass.UInt8
cutlass.UInt16
cutlass.UInt32
cutlass.UInt64

# Floating-point types
cutlass.Float16     # IEEE FP16
cutlass.BFloat16    # Brain Float 16
cutlass.Float32     # IEEE FP32
cutlass.Float64     # IEEE FP64

# Low-precision types (Hopper+)
cutlass.Float8_E4M3  # FP8 with 4 exponent bits, 3 mantissa bits
cutlass.Float8_E5M2  # FP8 with 5 exponent bits, 2 mantissa bits

# Boolean
cutlass.Boolean

# Complex types
cutlass.Complex64   # Complex<Float32>
cutlass.Complex128  # Complex<Float64>
```

---

## Floating-Point Types

### Float32 (FP32)

**Standard IEEE 754 single precision**

- **Bit width:** 32 bits (1 sign, 8 exponent, 23 mantissa)
- **Range:** ~1.4e-45 to ~3.4e38
- **Precision:** ~7 decimal digits

**Use cases:**
- General-purpose computation
- Accumulation in mixed-precision
- When precision is critical

**Example:**
```
@cute.kernel
def fp32_example(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    # All tensors use Float32
    idx = cute.arch.thread_idx()[0]
    c[idx] = a[idx] + b[idx]  # FP32 arithmetic
```

### Float16 (FP16)

**IEEE 754 half precision**

- **Bit width:** 16 bits (1 sign, 5 exponent, 10 mantissa)
- **Range:** ~6.1e-5 to 65504
- **Precision:** ~3 decimal digits

**Advantages:**
- 2× memory bandwidth vs FP32
- Native tensor core support
- Good for deep learning

**Disadvantages:**
- Limited range (can overflow/underflow)
- Lower precision

**Use cases:**
- Neural network inference
- Training with mixed precision
- Tensor core matrix operations

**Example:**
```
# Typical mixed-precision pattern
@cute.kernel
def mixed_precision_gemm(
    A: cute.Tensor,  # Float16
    B: cute.Tensor,  # Float16
    C: cute.Tensor   # Float32 accumulator
):
    # Compute in FP16, accumulate in FP32
    # MMA operations: FP16 × FP16 → FP32
    pass
```

### BFloat16 (BF16)

**Brain Float 16 - Google's format**

- **Bit width:** 16 bits (1 sign, 8 exponent, 7 mantissa)
- **Range:** Same as FP32 (~1.4e-45 to ~3.4e38)
- **Precision:** ~2 decimal digits (less than FP16)

**Key insight:** Same exponent range as FP32, just truncated mantissa.

**Advantages:**
- Same range as FP32 (no overflow issues)
- Easy conversion to/from FP32 (just truncate)
- 2× memory bandwidth vs FP32
- Tensor core support (Ampere+)

**Disadvantages:**
- Lower precision than FP16

**Use cases:**
- Deep learning training (popular for large models)
- When range matters more than precision
- Replacing FP32 without overflow concerns

**BF16 vs FP16 comparison:**

| Aspect | FP16 | BF16 |
|--------|------|------|
| Range | Limited | Same as FP32 |
| Precision | Better | Worse |
| Overflow risk | Higher | Lower |
| Training stability | Needs care | More stable |
| Hardware support | Volta+ | Ampere+ |

### Float8 (FP8) - Hopper and Beyond

**Two variants:**

**E4M3 (4 exponent, 3 mantissa bits)**
- Better precision, smaller range
- Good for forward pass

**E5M2 (5 exponent, 2 mantissa bits)**
- Better range, less precision  
- Good for backward pass gradients

**Advantages:**
- 4× memory bandwidth vs FP32
- 2× bandwidth vs FP16/BF16
- Native support on Hopper+
- Transformer-specific optimizations

**Use cases:**
- Large language model inference
- Extreme performance requirements
- Hopper H100 / Blackwell GPUs

**Example:**
```
# FP8 GEMM (Hopper+)
@cute.kernel
def fp8_gemm(
    A: cute.Tensor,  # Float8_E4M3
    B: cute.Tensor,  # Float8_E4M3
    C: cute.Tensor   # Float32 accumulator
):
    # Ultra-fast FP8 tensor core operations
    pass
```

### Float64 (FP64)

**IEEE 754 double precision**

- **Bit width:** 64 bits
- **Range:** ~5e-324 to ~1.8e308
- **Precision:** ~15 decimal digits

**Use cases:**
- Scientific computing requiring high precision
- Iterative solvers (conjugate gradient, etc.)
- Financial calculations

**Note:** Much slower on consumer GPUs (1/32 of FP32 throughput on gaming GPUs, 1/2 on datacenter GPUs).

---

## Integer Types

### Standard Integers
```
cutlass.Int8     # -128 to 127
cutlass.Int16    # -32768 to 32767
cutlass.Int32    # -2^31 to 2^31-1
cutlass.Int64    # -2^63 to 2^63-1

cutlass.UInt8    # 0 to 255
cutlass.UInt16   # 0 to 65535
cutlass.UInt32   # 0 to 2^32-1
cutlass.UInt64   # 0 to 2^64-1
```

### Integer Tensor Core Support

Tensor cores support integer operations on certain architectures:

**Int8 (Turing+):**
- INT8 × INT8 → INT32 matrix operations
- 4× throughput vs INT32
- Used in quantized neural networks

**Int4 (Ampere+):**
- INT4 × INT4 → INT32 operations
- Even higher throughput
- Extreme quantization

**Example:**
```
# INT8 quantized GEMM
@cute.kernel
def int8_gemm(
    A: cute.Tensor,  # Int8
    B: cute.Tensor,  # Int8
    C: cute.Tensor   # Int32 accumulator
):
    # Fast INT8 tensor core MMA
    pass
```

---

## Type Conversions

### Explicit Casting
```
# Convert tensor element type
float_val = some_tensor[idx].to(cutlass.Float32)
int_val = some_tensor[idx].to(cutlass.Int32)

# Convert entire tensors (creates a view with different type)
fp32_tensor = fp16_tensor.to(cutlass.Float32)
```

### Automatic Promotion

CuTe DSL follows standard type promotion rules:
```
# FP16 + FP32 → FP32 (promotes to wider type)
result = fp16_val + fp32_val  # Result is FP32

# Int32 + Float32 → Float32 (promotes to floating-point)
result = int32_val + float32_val  # Result is Float32
```

### Mixed-Precision Patterns

**Pattern 1: Compute in lower precision, accumulate in higher**
```
# Common in deep learning
# Compute: FP16 × FP16
# Accumulate: FP32
@cute.kernel
def mixed_precision_kernel(
    input_fp16: cute.Tensor,
    weights_fp16: cute.Tensor,
    output_fp32: cute.Tensor
):
    # MMA does FP16 compute, FP32 accumulation automatically
    cute.gemm(tiled_mma, output_fp32, input_fp16, weights_fp16, output_fp32)
```

**Pattern 2: Load in low precision, compute in high precision**
```
@cute.kernel
def load_convert_compute(
    input_fp16: cute.Tensor,
    output_fp32: cute.Tensor
):
    idx = cute.arch.thread_idx()[0]
    
    # Load FP16
    val = input_fp16[idx]
    
    # Convert to FP32 for precise computation
    val_fp32 = val.to(cutlass.Float32)
    
    # Compute in FP32
    result = expensive_operation(val_fp32)
    
    # Store as FP32
    output_fp32[idx] = result
```

---

## Architecture-Specific Type Support

### Ampere (SM80) - A100

**Native tensor core support:**
- FP16 (fast)
- BF16 (fast)
- TF32 (automatic for FP32 operations)
- INT8, INT4 (fast)
- FP64 (1/2 speed of FP32 on datacenter)

### Hopper (SM90) - H100

**All Ampere types, plus:**
- FP8 E4M3 (very fast)
- FP8 E5M2 (very fast)
- INT8 with better throughput
- Enhanced FP16/BF16 performance

### Blackwell (SM100) - B100/B200

**All Hopper types, plus:**
- Further FP8 optimizations
- 2-CTA instructions for higher throughput
- Enhanced mixed-precision support

---

## Choosing the Right Data Type

### Decision Tree
```
Need high precision? 
  → Use Float32 or Float64

Doing deep learning inference?
  → Modern model (2023+): Float8 (if Hopper+)
  → Older model: Float16 or BFloat16

Doing deep learning training?
  → Large models: BFloat16 (better range)
  → Small models: Float16 (better precision)
  → Accumulation: always Float32

Need integer operations?
  → Quantized inference: Int8
  → Extreme quantization: Int4
  → Counting/indexing: Int32

Memory bandwidth limited?
  → Use lowest precision that maintains accuracy
  → Float8 > Float16/BF16 > Float32 > Float64
```

### Performance vs Precision Trade-offs

| Type | Memory BW | Compute Speed | Precision | Range |
|------|-----------|---------------|-----------|-------|
| FP8 | 4× | 4× | Lowest | Limited |
| FP16 | 2× | 2× | Medium | Limited |
| BF16 | 2× | 2× | Lower | Good |
| FP32 | 1× | 1× | High | Good |
| FP64 | 0.5× | 0.5× | Highest | Excellent |

---

## Type Annotations in Function Signatures

### Static Types (Known at Compile Time)
```
@cute.jit
def typed_function(
    x: cutlass.Float32,      # Static: must be Float32
    y: cutlass.Int32,        # Static: must be Int32
    scale: cutlass.Float16   # Static: must be Float16
):
    result = x * scale + y
    return result
```

### Dynamic Types (Inferred from Arguments)
```
@cute.jit
def flexible_function(x, y):
    # Types inferred from call site
    return x + y

# Call with different types
result1 = flexible_function(1.0, 2.0)      # Float32 + Float32
result2 = flexible_function(fp16_a, fp16_b) # Float16 + Float16
```

### Tensor Element Types
```
@cute.jit
def tensor_function(tensor: cute.Tensor):
    # Access element type
    dtype = tensor.element_type
    
    # Check type
    if dtype == cutlass.Float16:
        # FP16-specific code
        pass
    elif dtype == cutlass.Float32:
        # FP32-specific code
        pass
```

---

## Common Patterns and Best Practices

### Pattern 1: Mixed-Precision GEMM
```
# Standard pattern: FP16 compute, FP32 accumulate
@cute.kernel
def gemm_mixed_precision(
    A: cute.Tensor,  # Float16
    B: cute.Tensor,  # Float16
    C: cute.Tensor   # Float32
):
    # MMA operation: Float16 × Float16 → Float32
    # Hardware automatically handles the precision
    cute.gemm(tiled_mma, C, A, B, C)
```

### Pattern 2: Quantization-Aware Operations
```
# INT8 quantized inference
@cute.kernel
def quantized_inference(
    input_int8: cute.Tensor,
    weights_int8: cute.Tensor,
    output_int32: cute.Tensor,
    scale: cutlass.Float32
):
    # Compute in INT8
    cute.gemm(tiled_mma, output_int32, input_int8, weights_int8, output_int32)
    
    # Dequantize output
    idx = cute.arch.thread_idx()[0]
    output_int32[idx] = (output_int32[idx].to(cutlass.Float32) * scale).to(cutlass.Int32)
```

### Pattern 3: Dynamic Precision Selection
```
@cute.jit
def adaptive_precision(
    data: cute.Tensor,
    use_fp16: cutlass.Constexpr  # Compile-time flag
):
    if cutlass.const_expr(use_fp16):
        # FP16 path
        result = process_fp16(data)
    else:
        # FP32 path
        result = process_fp32(data)
    return result

# Compiles different kernels for each precision
result_fp16 = adaptive_precision(data, use_fp16=True)
result_fp32 = adaptive_precision(data, use_fp16=False)
```

---

## Debugging Type Issues

### Common Type Errors

**Error: Type mismatch in assignment**
```
# ❌ Wrong
fp16_tensor[idx] = fp32_value  # Type mismatch!

# ✅ Correct
fp16_tensor[idx] = fp32_value.to(cutlass.Float16)
```

**Error: Incompatible types in operation**
```
# ❌ Wrong  
result = int8_val + float32_val  # Implicit promotion may not work

# ✅ Correct - explicit conversion
result = int8_val.to(cutlass.Float32) + float32_val
```

**Error: Wrong accumulator type for MMA**
```
# ❌ Wrong - FP16 accumulator too small
cute.gemm(tiled_mma, fp16_accum, fp16_a, fp16_b, fp16_accum)

# ✅ Correct - FP32 accumulator
cute.gemm(tiled_mma, fp32_accum, fp16_a, fp16_b, fp32_accum)
```

### Type Debugging
```
@cute.jit
def debug_types(tensor: cute.Tensor):
    # Print type information
    print(f"Element type: {tensor.element_type}")
    print(f"Element size: {cute.size_in_bytes(tensor.element_type, 1)} bytes")
    
    # Runtime type checking
    if tensor.element_type == cutlass.Float16:
        cute.printf("Processing FP16 tensor\n")
    elif tensor.element_type == cutlass.Float32:
        cute.printf("Processing FP32 tensor\n")
```

---

## Framework Interoperability

### PyTorch Type Mapping
```
import torch
import cutlass

# PyTorch → CuTe type mapping
torch_to_cute = {
    torch.float32: cutlass.Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float64: cutlass.Float64,
    torch.int8: cutlass.Int8,
    torch.int16: cutlass.Int16,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.uint8: cutlass.UInt8,
}

# Automatic conversion via DLPack
torch_tensor = torch.randn(100, dtype=torch.float16, device="cuda")
cute_tensor = from_dlpack(torch_tensor)  # Preserves Float16 type
```

---

## Performance Considerations

### Memory Bandwidth

**Bandwidth hierarchy (from fastest to slowest):**
1. FP8: 4× baseline
2. FP16/BF16: 2× baseline
3. FP32: 1× baseline (reference)
4. FP64: 0.5× baseline

**Rule of thumb:** Use lowest precision that maintains acceptable accuracy.

### Compute Throughput

**Tensor core throughput (relative to FP32):**
- FP8: ~4× faster (Hopper+)
- FP16/BF16: ~2× faster (Ampere+)
- TF32: ~1× (automatic for FP32 on Ampere+)
- FP64: ~0.5× slower (datacenter GPUs)

### Register Pressure

**Register usage per element:**
- FP8: 8 bits
- FP16/BF16: 16 bits
- FP32/Int32: 32 bits
- FP64/Int64: 64 bits

Lower precision → more values fit in registers → higher occupancy possible.

---

## Summary

**Key points:**
- Use **Float16/BF16** for most deep learning workloads
- Use **Float32** for accumulation and when precision matters
- Use **Float8** on Hopper+ for extreme performance
- Use **Int8** for quantized inference
- Always match types to hardware capabilities
- Consider memory bandwidth vs precision trade-offs

**Type selection priorities:**
1. Correctness (sufficient precision for task)
2. Hardware support (architecture compatibility)
3. Performance (bandwidth and compute throughput)
4. Memory footprint (especially for large models)