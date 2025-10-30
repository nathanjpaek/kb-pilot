---
topic: "mma_atoms"
difficulty: "advanced"
related_topics: ["copy_atoms", "tiled_operations", "architecture_specific"]
---

# MMA Atoms and Tensor Core Operations

## Overview

**MMA (Matrix Multiply-Accumulate) atoms** are the fundamental building blocks for tensor core operations in CuTe DSL. They describe how to perform matrix multiplication using specialized hardware (tensor cores) for maximum performance.

**Key concept:** Instead of manually computing matrix products with loops, use MMA atoms that compile directly to tensor core instructions - achieving 10-100× speedup over CUDA cores.

---

## What is an MMA Atom?

An **MMA atom** encapsulates:
1. **Matrix dimensions** (M×N×K tile size)
2. **Data types** (input and accumulator types)
3. **Tensor core instruction** (architecture-specific)
4. **Thread layout** (how threads cooperate)

**Operation:** `D = A × B + C`
- **A:** M×K matrix (input)
- **B:** K×N matrix (input)
- **C:** M×N matrix (accumulator)
- **D:** M×N matrix (output)

---

## Basic MMA Operation

### Conceptual Example
```
@cute.kernel
def simple_mma_example():
    # Create MMA atom (16x8x16 tile, FP16 inputs, FP16 output)
    mma_atom = cute.make_mma_atom(
        cute.SM80_16x8x16_F16F16F16F16_TN
    )
    
    # Allocate register fragments
    frag_A = cute.make_rmem_tensor(cute.make_shape(16, 16), cutlass.Float16)
    frag_B = cute.make_rmem_tensor(cute.make_shape(16, 8), cutlass.Float16)
    frag_C = cute.make_rmem_tensor(cute.make_shape(16, 8), cutlass.Float16)
    
    # Initialize accumulator
    frag_C.store(0.0)
    
    # Load data into fragments (from shared memory, etc.)
    # ... load frag_A and frag_B ...
    
    # Perform MMA: frag_C = frag_A × frag_B + frag_C
    cute.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
    
    # frag_C now contains result
```

---

## MMA Atom Naming Convention

### Format: `SM{ARCH}_{M}x{N}x{K}_{TYPES}_{LAYOUT}`

**Example:** `SM80_16x8x16_F16F16F16F16_TN`

**Breakdown:**
- `SM80`: Architecture (SM80 = Ampere)
- `16x8x16`: Tile dimensions (M=16, N=8, K=16)
- `F16F16F16F16`: Types (A_type, B_type, C_type, D_type)
- `TN`: Layout (Transposed/Non-transposed)

**Type codes:**
- `F16`: Float16 (FP16)
- `BF16`: BFloat16
- `F32`: Float32
- `TF32`: TensorFloat32 (Ampere+)
- `F8`: Float8 (Hopper+)
- `E4M3`: FP8 E4M3 format
- `E5M2`: FP8 E5M2 format
- `I8`: Int8
- `I4`: Int4

**Layout codes:**
- `TN`: A is Transposed, B is Non-transposed
- `NT`: A is Non-transposed, B is Transposed
- `NN`: Both Non-transposed
- `TT`: Both Transposed

---

## Architecture-Specific MMA Atoms

### Ampere (SM80) - A100

**FP16 MMA:**
```
@cute.kernel
def ampere_fp16_mma():
    # 16x8x16 FP16 MMA
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F16F16_TN)
    
    # Inputs: FP16, Output: FP16
    # Fast, good precision
```

**FP16 with FP32 accumulation (most common):**
```
@cute.kernel
def ampere_mixed_precision_mma():
    # 16x8x16 FP16 inputs, FP32 accumulator
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Compute in FP16, accumulate in FP32
    # Best balance of speed and accuracy
```

**BF16 MMA:**
```
@cute.kernel
def ampere_bf16_mma():
    # 16x8x16 BF16 inputs, FP32 accumulator
    mma = cute.make_mma_atom(cute.SM80_16x8x16_BF16BF16F32F32_TN)
    
    # BF16 has same range as FP32, less precision than FP16
    # Good for large models
```

**TF32 (automatic for FP32 on Ampere):**
```
@cute.kernel
def ampere_tf32_mma():
    # 16x8x8 TF32 (reduced precision FP32)
    mma = cute.make_mma_atom(cute.SM80_16x8x8_F32TF32TF32F32_TN)
    
    # Automatically used for FP32 operations on Ampere
    # ~8× faster than FP32 on CUDA cores
```

**INT8 MMA:**
```
@cute.kernel
def ampere_int8_mma():
    # 16x8x32 INT8 inputs, INT32 accumulator
    mma = cute.make_mma_atom(cute.SM80_16x8x32_I8I8I32I32_TN)
    
    # Quantized inference
    # Very fast, need dequantization afterward
```

### Hopper (SM90) - H100

**FP16 MMA:**
```
@cute.kernel
def hopper_fp16_mma():
    # Hopper has same FP16 atoms as Ampere, but faster
    mma = cute.make_mma_atom(cute.SM90_16x8x16_F16F16F32F32_TN)
    
    # 2× faster than Ampere
```

**FP8 MMA (Hopper exclusive):**
```
@cute.kernel
def hopper_fp8_mma():
    # 16x8x32 FP8 E4M3 inputs, FP16 accumulator
    mma = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)
    
    # E4M3 for forward pass (better precision)
    # E5M2 for backward pass (better range)
    # 4× faster than FP16!
```

**Enhanced INT8:**
```
@cute.kernel
def hopper_int8_mma():
    # 16x8x64 INT8 (larger K dimension than Ampere)
    mma = cute.make_mma_atom(cute.SM90_16x8x64_I8I8I32I32_TN)
    
    # Better throughput than Ampere
```

### Blackwell (SM100) - B100/B200

**Note:** Exact atom names may vary - check documentation for Blackwell specifics.
```
@cute.kernel
def blackwell_mma():
    # Blackwell enhancements (example)
    # Enhanced FP8, potentially larger tiles
    # Check official documentation for exact atoms
    pass
```

---

## MMA Operation: cute.gemm()

### Basic Syntax
```
cute.gemm(mma_atom, D, A, B, C)
```

**Parameters:**
- `mma_atom`: MMA atom descriptor
- `D`: Output accumulator (destination)
- `A`: Left matrix (M×K)
- `B`: Right matrix (K×N)
- `C`: Input accumulator (same as D for in-place)

**Operation:** `D = A × B + C`

### Simple Example
```
@cute.kernel
def basic_gemm():
    # Create MMA atom
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Allocate fragments (in registers)
    frag_A = cute.make_rmem_tensor(cute.make_shape(16, 16), cutlass.Float16)
    frag_B = cute.make_rmem_tensor(cute.make_shape(16, 8), cutlass.Float16)
    frag_C = cute.make_rmem_tensor(cute.make_shape(16, 8), cutlass.Float32)
    
    # Initialize accumulator
    for i in range(16):
        for j in range(8):
            frag_C[i, j] = 0.0
    
    # Load A and B from shared memory (example)
    # cute.copy(smem_A, frag_A)
    # cute.copy(smem_B, frag_B)
    
    # Perform matrix multiply-accumulate
    cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)
    
    # frag_C contains result: C = A × B + C
```

---

## Fragment Shapes and Layouts

### Understanding Fragment Dimensions

**MMA atom specifies the compute tile:**
- M × N × K: 16 × 8 × 16

**But fragments may have different layouts for different threads.**

### Creating Fragments
```
@cute.kernel
def create_fragments():
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Create tiled MMA (covers multiple threads)
    tiled_mma = cute.make_tiled_mma(mma_atom)
    
    # Get fragments for this thread
    thread_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0])
    
    # Make fragments with correct shapes for this thread
    frag_A = tiled_mma.make_fragment_A(cute.make_shape(128, 64))
    frag_B = tiled_mma.make_fragment_B(cute.make_shape(64, 128))
    frag_C = tiled_mma.make_fragment_C(cute.make_shape(128, 128))
    
    # Fragments are automatically sized correctly for thread
```

---

## Data Type Combinations

### FP16 Compute, FP32 Accumulate (Recommended)
```
@cute.kernel
def fp16_fp32_mma():
    # Most common: FP16 inputs, FP32 accumulator
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # A and B are FP16 (fast compute, saves memory)
    frag_A = cute.make_rmem_tensor(..., cutlass.Float16)
    frag_B = cute.make_rmem_tensor(..., cutlass.Float16)
    
    # C is FP32 (accurate accumulation, prevents overflow)
    frag_C = cute.make_rmem_tensor(..., cutlass.Float32)
    
    cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)
```

**Why this combination?**
- Fast computation (FP16 tensor cores)
- Accurate accumulation (FP32 prevents precision loss)
- Standard for deep learning training

### BF16 Compute, FP32 Accumulate
```
@cute.kernel
def bf16_fp32_mma():
    # BF16 inputs, FP32 accumulator
    mma = cute.make_mma_atom(cute.SM80_16x8x16_BF16BF16F32F32_TN)
    
    frag_A = cute.make_rmem_tensor(..., cutlass.BFloat16)
    frag_B = cute.make_rmem_tensor(..., cutlass.BFloat16)
    frag_C = cute.make_rmem_tensor(..., cutlass.Float32)
    
    cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)
```

**When to use BF16:**
- Large language models
- When range matters more than precision
- Same exponent range as FP32 (no overflow issues)

### FP8 Compute (Hopper)
```
@cute.kernel
def fp8_mma():
    # FP8 E4M3 inputs, FP16 accumulator
    mma = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)
    
    frag_A = cute.make_rmem_tensor(..., cutlass.Float8_E4M3)
    frag_B = cute.make_rmem_tensor(..., cutlass.Float8_E5M2)
    frag_C = cute.make_rmem_tensor(..., cutlass.Float16)
    
    cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)
```

**When to use FP8:**
- Hopper H100 GPU
- Maximum throughput (4× FP16 speed)
- Large model inference
- Need careful scaling/quantization

### INT8 Quantized
```
@cute.kernel
def int8_mma():
    # INT8 inputs, INT32 accumulator
    mma = cute.make_mma_atom(cute.SM80_16x8x32_I8I8I32I32_TN)
    
    frag_A = cute.make_rmem_tensor(..., cutlass.Int8)
    frag_B = cute.make_rmem_tensor(..., cutlass.Int8)
    frag_C = cute.make_rmem_tensor(..., cutlass.Int32)
    
    cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)
    
    # Need to dequantize output:
    # frag_C_float = frag_C * scale + bias
```

---

## Complete MMA Example: Tiled GEMM

### Full GEMM Kernel
```
@cute.kernel
def tiled_gemm_kernel(
    gA: cute.Tensor,  # Global A (M×K)
    gB: cute.Tensor,  # Global B (K×N)
    gC: cute.Tensor,  # Global C (M×N)
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
    tile_k: cutlass.Constexpr
):
    # 1. Create MMA atom
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # 2. Create tiled MMA
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        cute.make_shape(tile_m, tile_n, tile_k)
    )
    
    # 3. Get thread's MMA slice
    thread_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0])
    
    # 4. Allocate shared memory
    smem_A = cute.make_smem_tensor(
        cute.make_shape(tile_m, tile_k),
        cutlass.Float16
    )
    smem_B = cute.make_smem_tensor(
        cute.make_shape(tile_k, tile_n),
        cutlass.Float16
    )
    
    # 5. Allocate register fragments
    frag_A = tiled_mma.make_fragment_A(smem_A.shape)
    frag_B = tiled_mma.make_fragment_B(smem_B.shape)
    frag_C = tiled_mma.make_fragment_C(gC.shape)
    
    # 6. Initialize accumulator
    frag_C.store(0.0)
    
    # 7. Get block's tile coordinates
    block_m = cute.arch.block_idx()[0]
    block_n = cute.arch.block_idx()[1]
    
    # 8. Loop over K dimension
    num_k_tiles = cute.size(gA, mode=1) // tile_k
    
    for k_tile in range(num_k_tiles):
        # Load tiles from global to shared memory
        cute.copy(gA[block_m, k_tile], smem_A)
        cute.copy(gB[k_tile, block_n], smem_B)
        
        cute.arch.syncthreads()
        
        # Load from shared to registers
        cute.copy(thread_mma.partition_A(smem_A), frag_A)
        cute.copy(thread_mma.partition_B(smem_B), frag_B)
        
        # MMA operation
        cute.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
        
        cute.arch.syncthreads()
    
    # 9. Store result back to global memory
    cute.copy(frag_C, gC[block_m, block_n])
```

---

## MMA Performance Characteristics

### Throughput by Data Type (Relative to FP32 CUDA cores)

**Ampere A100:**

| Data Type | Throughput | Notes |
|-----------|------------|-------|
| FP32 (CUDA) | 1× | Baseline (19.5 TFLOPS) |
| TF32 | 8× | Auto for FP32 on tensor cores |
| FP16 | 16× | 312 TFLOPS |
| BF16 | 16× | 312 TFLOPS |
| INT8 | 32× | 624 TOPS |
| INT4 | 64× | 1248 TOPS |

**Hopper H100:**

| Data Type | Throughput | Notes |
|-----------|------------|-------|
| FP32 (CUDA) | 1× | Baseline (60 TFLOPS) |
| TF32 | 8× | 480 TFLOPS |
| FP16 | 16× | 960 TFLOPS |
| BF16 | 16× | 960 TFLOPS |
| FP8 | 64× | 3840 TFLOPS! |
| INT8 | 64× | 3840 TOPS |

### Tile Size Impact

**Larger tiles = better efficiency:**
```
16×8×16:   ~50-60% peak tensor core utilization
32×16×16:  ~70-80% peak utilization
64×32×16:  ~85-90% peak utilization
128×64×16: ~90-95% peak utilization
```

**Trade-off:** Larger tiles use more registers, may reduce occupancy.

---

## Common MMA Patterns

### Pattern 1: Accumulation Loop
```
@cute.kernel
def accumulation_loop(num_k_tiles: cutlass.Int32):
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Accumulator in FP32 (important!)
    frag_C = cute.make_rmem_tensor(cute.make_shape(16, 8), cutlass.Float32)
    frag_C.store(0.0)
    
    # Loop over K tiles
    for k in range(num_k_tiles):
        # Load A and B tiles
        load_tiles(k, frag_A, frag_B)
        
        # Accumulate: C += A × B
        cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)
    
    # frag_C contains sum of all K tiles
```

### Pattern 2: Mixed Precision Pipeline
```
@cute.kernel
def mixed_precision_pipeline():
    # FP16 MMA atom
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Load in FP16 (bandwidth efficient)
    frag_A_fp16 = cute.make_rmem_tensor(..., cutlass.Float16)
    frag_B_fp16 = cute.make_rmem_tensor(..., cutlass.Float16)
    
    # Accumulate in FP32 (precision)
    frag_C_fp32 = cute.make_rmem_tensor(..., cutlass.Float32)
    frag_C_fp32.store(0.0)
    
    # Compute
    cute.gemm(mma, frag_C_fp32, frag_A_fp16, frag_B_fp16, frag_C_fp32)
    
    # Convert back to FP16 for output (if needed)
    frag_C_fp16 = frag_C_fp32.to(cutlass.Float16)
```

### Pattern 3: Split-K (Parallel K Reduction)
```
@cute.kernel
def split_k_gemm(k_slice_start: cutlass.Int32, k_slice_end: cutlass.Int32):
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    frag_C = cute.make_rmem_tensor(..., cutlass.Float32)
    frag_C.store(0.0)
    
    # Each block computes partial K range
    for k in range(k_slice_start, k_slice_end):
        load_tiles(k, frag_A, frag_B)
        cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)
    
    # Store partial result
    # Later, reduce across blocks
    store_partial(frag_C, block_id)
```

---

## Alignment Requirements

### Memory Alignment for MMA

**Tensor cores require specific alignment:**
```
@cute.kernel
def aligned_mma(A: cute.Tensor, B: cute.Tensor):
    # Ensure 128-byte alignment for optimal performance
    aligned_A = cute.assume(A, align=128)
    aligned_B = cute.assume(B, align=128)
    
    # MMA operations benefit significantly from alignment
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    # ... use aligned_A and aligned_B ...
```

### Layout Requirements

**MMA atoms expect specific layouts:**
```
# A matrix: Usually column-major or transposed
# B matrix: Usually row-major or non-transposed
# C matrix: Usually row-major

# TN layout means:
# - A is Transposed (column-major)
# - B is Non-transposed (row-major)
```

---

## Debugging MMA Operations

### Check Fragment Shapes
```
@cute.kernel
def debug_mma_shapes():
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    frag_A = cute.make_rmem_tensor(cute.make_shape(16, 16), cutlass.Float16)
    frag_B = cute.make_rmem_tensor(cute.make_shape(16, 8), cutlass.Float16)
    frag_C = cute.make_rmem_tensor(cute.make_shape(16, 8), cutlass.Float32)
    
    # Print shapes at compile time
    print(f"Fragment A shape: {frag_A.shape}")
    print(f"Fragment B shape: {frag_B.shape}")
    print(f"Fragment C shape: {frag_C.shape}")
```

### Verify MMA Results
```
@cute.jit
def verify_mma_result():
    import torch
    
    # CPU reference
    A_torch = torch.randn(128, 64, dtype=torch.float16, device="cpu")
    B_torch = torch.randn(64, 128, dtype=torch.float16, device="cpu")
    C_reference = torch.matmul(A_torch.float(), B_torch.float())
    
    # GPU MMA result
    A_cuda = A_torch.cuda()
    B_cuda = B_torch.cuda()
    C_cuda = torch.zeros(128, 128, dtype=torch.float32, device="cuda")
    
    mma_kernel.launch(...)(
        from_dlpack(A_cuda),
        from_dlpack(B_cuda),
        from_dlpack(C_cuda)
    )
    
    # Compare
    torch.testing.assert_close(C_cuda.cpu(), C_reference, rtol=1e-2, atol=1e-2)
    print("✓ MMA result matches reference")
```

---

## Best Practices

### ✅ DO

**Use FP32 accumulation:**
```
# ✓ GOOD: FP32 accumulator prevents precision loss
mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
frag_C = cute.make_rmem_tensor(..., cutlass.Float32)
```

**Initialize accumulators:**
```
# ✓ GOOD: Always initialize to zero
frag_C.store(0.0)

# Or add bias
frag_C = load_bias()
```

**Use largest tile that fits:**
```
# ✓ GOOD: Larger tiles = better efficiency
# But watch register usage!
tile_m, tile_n = 128, 128  # Good balance
```

**Match MMA atom to data types:**
```
# ✓ GOOD: Atom matches actual data types
if input_dtype == cutlass.Float16:
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
elif input_dtype == cutlass.BFloat16:
    mma = cute.make_mma_atom(cute.SM80_16x8x16_BF16BF16F32F32_TN)
```

### ❌ DON'T

**Don't use FP16 accumulation for many iterations:**
```
# ✗ BAD: FP16 accumulator loses precision
mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F16F16_TN)
frag_C = cute.make_rmem_tensor(..., cutlass.Float16)

for k in range(1000):  # Many accumulations = precision loss!
    cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)
```

**Don't forget to initialize:**
```
# ✗ BAD: Uninitialized accumulator
frag_C = cute.make_rmem_tensor(..., cutlass.Float32)
cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)  # Random values!
```

**Don't mismatch atom and data types:**
```
# ✗ BAD: FP32 atom, but FP16 fragments
mma = cute.make_mma_atom(cute.SM80_16x8x8_F32F32F32F32_TN)
frag_A = cute.make_rmem_tensor(..., cutlass.Float16)  # Type mismatch!
```

---

## Summary

**MMA atoms:**
- Fundamental building blocks for tensor core operations
- Architecture-specific (SM80, SM90, SM100)
- Define tile size, data types, and layout

**Key MMA atoms:**
- Ampere: `SM80_16x8x16_F16F16F32F32_TN` (most common)
- Hopper: `SM90_16x8x32_E4M3E5M2F16_SS` (FP8 for max speed)
- Use FP32 accumulation for precision

**Performance:**
- Tensor cores: 10-100× faster than CUDA cores
- FP16: 16× speedup on Ampere
- FP8: 64× speedup on Hopper
- Larger tiles = better efficiency (up to register limits)

**Best practices:**
- Always initialize accumulators
- Use FP32 accumulation
- Match atom to data types
- Verify results against reference

---

## Next Steps

- [Tiled Operations](./tiled_operations.md) - Creating tiled MMA for thread cooperation
- [Copy Atoms](./copy_atoms.md) - Loading data for MMA
- [Architecture-Specific Features](../05_architecture_specific/) - Per-GPU optimizations

---

## Further Reading

- [CUDA Tensor Cores](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Ampere Tensor Cores](https://www.nvidia.com/en-us/data-center/a100/)
- [Hopper Tensor Cores](https://www.nvidia.com/en-us/data-center/h100/)