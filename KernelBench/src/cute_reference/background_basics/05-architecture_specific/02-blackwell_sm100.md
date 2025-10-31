---
topic: "blackwell_sm100"
difficulty: "advanced"
related_topics: ["hopper_sm90", "mma_atoms", "choosing_atoms"]
---

# Blackwell (SM100) Architecture Features

## Overview

**Blackwell** (compute capability SM100) is NVIDIA's latest GPU architecture, introducing significant improvements for AI workloads including enhanced tensor cores, tensor memory (TMEM), 2-CTA instructions, and improved FP4/FP6 support. The flagship Blackwell GPUs are the **B100** and **B200**.

**Key innovations:**
- 5th generation Tensor Cores with FP4, FP6, and enhanced FP8
- Tensor Memory (TMEM) - a new memory hierarchy level
- 2-CTA instructions for cross-block cooperation
- Block-scaled formats (MXFP4, MXFP6, MXFP8)
- Enhanced TMA with direct TMEM support

**Note:** Blackwell is very new. Some features may not be fully documented yet. Check latest NVIDIA documentation for updates.

---

## Blackwell Key Specifications

### B100/B200 GPU Specifications (Approximate)

**B200 Compute:**
- 208 SMs (Streaming Multiprocessors) (estimated)
- ~30,000+ CUDA cores
- 832+ 5th generation Tensor Cores

**B200 Memory:**
- 192GB HBM3e
- Memory bandwidth: 8 TB/s (2.4× H100!)
- Enhanced L2 cache
- 256KB+ shared memory per SM
- New: Tensor Memory (TMEM)

**B200 Performance (estimated):**
- FP16: ~2000 TFLOPS (2× H100)
- FP8: ~8000 TFLOPS (2× H100)
- FP6: ~10,000+ TFLOPS
- FP4: ~16,000+ TFLOPS (!)
- TF32: ~1000 TFLOPS

**Note:** Exact specifications may vary. These are based on available information at time of writing.

---

## Tensor Memory (TMEM)

### What is TMEM?

**Tensor Memory (TMEM)** is a new memory hierarchy level in Blackwell, sitting between registers and shared memory.

**Memory hierarchy (Blackwell):**
```
Registers (RMEM)
    ↕
Tensor Memory (TMEM) ← NEW!
    ↕
Shared Memory (SMEM)
    ↕
L2 Cache
    ↕
Global Memory (GMEM)
```

**TMEM characteristics:**
- Larger than registers, smaller than SMEM
- Faster access than SMEM
- Optimized for tensor core accumulator storage
- Reduces register pressure

### Allocating TMEM

**Conceptual API (exact syntax may vary):**
```
@cute.kernel
def blackwell_with_tmem():
    # Allocate tensor memory for MMA accumulator
    # (Exact API TBD - check Blackwell documentation)
    tmem_accumulator = cute.make_tmem_tensor(
        cute.make_shape(128, 128),
        cutlass.Float32
    )
    
    # Use TMEM for large accumulators
    # Offloads register pressure
```

### TMEM Use Cases

**1. Large MMA accumulators:**
```
# Instead of:
frag_C = cute.make_rmem_tensor(...)  # Uses many registers

# Use TMEM:
frag_C = cute.make_tmem_tensor(...)  # Frees up registers
```

**2. Intermediate results:**
```
# Store intermediate MMA results in TMEM
# Process in chunks without exhausting registers
```

---

## Enhanced FP8 Support

### Blackwell FP8 Improvements

**Faster FP8 operations:**
- 2× FP8 throughput compared to Hopper
- Enhanced E4M3 and E5M2 support
- Better scaling/conversion support

### FP8 MMA Atoms (Blackwell)
```
@cute.kernel
def blackwell_fp8_mma():
    # Enhanced FP8 MMA (same dimensions as Hopper, but faster)
    mma = cute.make_mma_atom(cute.SM100_16x8x32_E4M3E5M2F16_SS)
    
    # ~8000 TFLOPS on B200 (vs 3958 on H100)
```

---

## FP6 Support

### What is FP6?

**FP6** (6-bit floating-point) is a new ultra-low precision format:
- 3-bit exponent, 2-bit mantissa (E3M2)
- Even lower precision than FP8
- Massive throughput increase

### FP6 MMA Atoms
```
@cute.kernel
def blackwell_fp6_mma():
    # FP6 E3M2 MMA atom
    # (Exact atom name TBD - check documentation)
    mma = cute.make_mma_atom(cute.SM100_FP6_MMA_ATOM)
    
    # Estimated: ~10,000+ TFLOPS on B200
```

**Use cases:**
- Ultra-low precision inference
- Heavily quantized models
- When accuracy can be sacrificed for speed

---

## FP4 Support

### What is FP4?

**FP4** (4-bit floating-point):
- 2-bit exponent, 1-bit mantissa (E2M1)
- Extremely low precision
- Highest throughput

### FP4 MMA Atoms
```
@cute.kernel
def blackwell_fp4_mma():
    # FP4 E2M1 MMA atom
    # (Exact atom name TBD - check documentation)
    mma = cute.make_mma_atom(cute.SM100_FP4_MMA_ATOM)
    
    # Estimated: ~16,000+ TFLOPS on B200
    # 2× FP8, 8× FP16!
```

---

## Block-Scaled Formats

### What are Block-Scaled Formats?

**Block scaling:** Apply a different scale factor to each block of values, improving quantization quality.

**Supported formats:**
- **NVFP4** (NVIDIA FP4 with block scaling)
- **MXFP4** (Microscaling FP4)
- **MXFP6** (Microscaling FP6)
- **MXFP8** (Microscaling FP8)

### Block-Scaled Example
```
# Regular FP8: Single scale for entire tensor
tensor_fp8 = tensor / global_scale

# Block-scaled MXFP8: Different scale per block
for block in blocks:
    block_fp8 = block / block_scale[block_id]
```

**Benefits:**
- Better accuracy than uniform quantization
- Handles outliers better
- Minimal overhead with hardware support

### Using Block-Scaled Formats
```
@cute.kernel
def blackwell_block_scaled_mma():
    # MXFP8 MMA atom with block scaling
    # (API example - check documentation)
    mma = cute.make_mma_atom(cute.SM100_MXFP8_MMA_ATOM)
    
    # Requires scale factors per block
    # Hardware handles scaling automatically
```

---

## 2-CTA Instructions

### What are 2-CTA Instructions?

**2-CTA instructions** allow two thread blocks (CTAs) to cooperate on a single tensor core operation.

**Traditional (Hopper):**
```
# One CTA computes 128×128 tile
CTA 0: 128×128 output tile
```

**Blackwell 2-CTA:**
```
# Two CTAs cooperate on 256×128 tile (or 128×256)
CTA 0 + CTA 1: 256×128 output tile
```

### Using 2-CTA Instructions
```
@cute.kernel
def blackwell_2cta_mma():
    # 2-CTA MMA atom
    # (Exact API TBD)
    mma_2cta = cute.make_mma_atom_2cta(
        cute.SM100_2CTA_MMA_ATOM,
        cta_layout=cute.make_shape(2, 1)  # 2 CTAs cooperate
    )
    
    # Two CTAs share the MMA workload
    # Requires cluster of at least 2 CTAs
```

**Benefits:**
- Process larger tiles efficiently
- Better load balancing
- Reduced epilogue overhead

---

## Enhanced TMA

### TMA Improvements in Blackwell

**Blackwell TMA enhancements:**
- Direct TMEM support
- Faster transfer rates
- Better handling of complex layouts
- Improved swizzling

### TMA with TMEM
```
@cute.kernel
def blackwell_tma_to_tmem(
    gmem: cute.Tensor,
    tma_desc: cute.TMADescriptor
):
    # Allocate TMEM
    tmem = cute.make_tmem_tensor(cute.make_shape(128, 64), cutlass.Float16)
    
    # TMA can load directly to TMEM (bypassing SMEM)
    if cute.arch.thread_idx()[0] == 0:
        cute.tma_load_to_tmem(tma_desc, gmem[tile], tmem)
    
    # Wait for TMA
    cute.arch.tma_wait()
    
    # Use TMEM data
```

---

## Blackwell MMA Atoms Summary

### Available Precision Formats

**High precision:**
```
SM100_16x8x16_F16F16F32F32_TN     # FP16 (2× Hopper speed)
SM100_16x8x16_BF16BF16F32F32_TN   # BF16
SM100_16x8x8_F32TF32TF32F32_TN    # TF32
```

**Medium precision:**
```
SM100_16x8x32_E4M3E5M2F16_SS      # FP8 (2× Hopper speed)
SM100_MXFP8_MMA_ATOM              # Block-scaled FP8
```

**Low precision (new):**
```
SM100_FP6_MMA_ATOM                # FP6 E3M2
SM100_MXFP6_MMA_ATOM              # Block-scaled FP6
```

**Ultra-low precision (new):**
```
SM100_FP4_MMA_ATOM                # FP4 E2M1
SM100_MXFP4_MMA_ATOM              # Block-scaled FP4
```

**Note:** Exact atom names TBD. Check official Blackwell CuTe documentation.

---

## Blackwell-Specific Best Practices

### Practice 1: Use TMEM for Large Accumulators
```
# ✓ GOOD: TMEM for accumulators (reduces register pressure)
frag_C = cute.make_tmem_tensor(cute.make_shape(256, 256), cutlass.Float32)

# ⚠️ OK: Registers (if TMEM not needed)
frag_C = cute.make_rmem_tensor(cute.make_shape(128, 128), cutlass.Float32)
```

### Practice 2: Use FP8/FP6/FP4 Aggressively
```
# ✓ GOOD: Use lowest precision that maintains accuracy
# FP4 for extreme quantization
mma_fp4 = cute.make_mma_atom(cute.SM100_FP4_MMA_ATOM)

# FP6 for better accuracy
mma_fp6 = cute.make_mma_atom(cute.SM100_FP6_MMA_ATOM)

# FP8 when FP4/FP6 insufficient
mma_fp8 = cute.make_mma_atom(cute.SM100_16x8x32_E4M3E5M2F16_SS)
```

### Practice 3: Leverage 2-CTA Instructions
```
# ✓ GOOD: Use 2-CTA for large tiles
mma_2cta = cute.make_mma_atom_2cta(...)
kernel.launch(..., cluster=[2, 2, 1])

# ⚠️ OK: Single CTA for smaller tiles
```

### Practice 4: Use Block-Scaled Formats
```
# ✓ GOOD: Block scaling improves accuracy
mma = cute.make_mma_atom(cute.SM100_MXFP8_MMA_ATOM)

# ⚠️ OK: Uniform scaling if block scaling overhead too high
mma = cute.make_mma_atom(cute.SM100_16x8x32_E4M3E5M2F16_SS)
```

---

## Blackwell Example: GEMM with TMEM

### Conceptual Example (API subject to change)
```
@cute.kernel
def blackwell_gemm_with_tmem(
    gmem_A: cute.Tensor,
    gmem_B: cute.Tensor,
    gmem_C: cute.Tensor,
    tma_desc_A: cute.TMADescriptor,
    tma_desc_B: cute.TMADescriptor
):
    # ========================================
    # Configuration
    # ========================================
    tile_m, tile_n, tile_k = 256, 256, 64
    
    # ========================================
    # Allocate Shared Memory
    # ========================================
    smem_A = cute.make_smem_tensor(cute.make_shape(tile_m, tile_k), cutlass.Float16)
    smem_B = cute.make_smem_tensor(cute.make_shape(tile_k, tile_n), cutlass.Float16)
    
    # ========================================
    # Allocate Tensor Memory (TMEM)
    # ========================================
    # Use TMEM for accumulator (reduces register pressure!)
    tmem_accumulator = cute.make_tmem_tensor(
        cute.make_shape(tile_m, tile_n),
        cutlass.Float32
    )
    
    # Initialize accumulator in TMEM
    tmem_accumulator.store(0.0)
    
    # ========================================
    # Create MMA
    # ========================================
    mma_atom = cute.make_mma_atom(cute.SM100_16x8x16_F16F16F32F32_TN)
    tiled_mma = cute.make_tiled_mma(mma_atom, ...)
    
    tid = cute.arch.thread_idx()[0]
    thread_mma = tiled_mma.get_slice(tid)
    
    # ========================================
    # Main Loop
    # ========================================
    block_m = cute.arch.block_idx()[0]
    block_n = cute.arch.block_idx()[1]
    
    num_k_tiles = cute.size(gmem_A, mode=1) // tile_k
    
    for k in range(num_k_tiles):
        # TMA load to SMEM
        if tid == 0:
            cute.tma_load(tma_desc_A, gmem_A[block_m, k], smem_A)
            cute.tma_load(tma_desc_B, gmem_B[k, block_n], smem_B)
        
        cute.arch.tma_wait()
        cute.arch.syncthreads()
        
        # Load SMEM → Registers
        frag_A = cute.make_rmem_tensor(...)
        frag_B = cute.make_rmem_tensor(...)
        
        thread_A = thread_mma.partition_A(smem_A)
        thread_B = thread_mma.partition_B(smem_B)
        
        cute.copy(thread_A, frag_A)
        cute.copy(thread_B, frag_B)
        
        # MMA: accumulate into TMEM
        cute.gemm_tmem(tiled_mma, tmem_accumulator, frag_A, frag_B, tmem_accumulator)
        
        cute.arch.syncthreads()
    
    # ========================================
    # Epilogue: TMEM → GMEM
    # ========================================
    # Move accumulator from TMEM to registers
    frag_C = cute.make_rmem_tensor(...)
    cute.copy(tmem_accumulator, frag_C)
    
    # Store to global memory
    block_C = gmem_C[block_m, block_n]
    thread_C = thread_mma.partition_C(block_C)
    cute.copy(frag_C, thread_C)
```

---

## FP4 Quantization Example

### Quantizing to FP4
```
@cute.jit
def quantize_to_fp4(tensor_fp16: torch.Tensor):
    # Find per-block maximum
    block_size = 32
    num_blocks = tensor_fp16.numel() // block_size
    
    tensor_blocked = tensor_fp16.view(num_blocks, block_size)
    
    # Compute scales per block
    block_max = torch.max(torch.abs(tensor_blocked), dim=1)[0]
    fp4_max = 7.0  # E2M1 max value (approximate)
    scales = block_max / fp4_max
    
    # Quantize each block
    tensor_fp4 = tensor_blocked / scales.unsqueeze(1)
    
    # Convert to FP4 (conceptual - exact API TBD)
    tensor_fp4 = tensor_fp4.to(cutlass.Float4_E2M1)
    
    return tensor_fp4, scales

@cute.jit
def dequantize_from_fp4(tensor_fp4, scales):
    # Dequantize
    tensor_fp16 = tensor_fp4.to(torch.float16)
    
    num_blocks = scales.shape[0]
    block_size = tensor_fp4.numel() // num_blocks
    
    tensor_blocked = tensor_fp16.view(num_blocks, block_size)
    tensor_dequantized = tensor_blocked * scales.unsqueeze(1)
    
    return tensor_dequantized.flatten()
```

---

## Performance Expectations

### Blackwell Performance Targets

**Compared to Hopper H100:**

| Precision | Hopper H100 | Blackwell B200 | Speedup |
|-----------|-------------|----------------|---------|
| FP16      | 989 TFLOPS  | ~2000 TFLOPS   | 2× |
| FP8       | 3958 TFLOPS | ~8000 TFLOPS   | 2× |
| FP6       | N/A         | ~10,000+ TFLOPS| N/A |
| FP4       | N/A         | ~16,000+ TFLOPS| N/A |
| TF32      | 495 TFLOPS  | ~1000 TFLOPS   | 2× |

**Memory bandwidth:**
- H100: 3.35 TB/s
- B200: 8 TB/s (2.4×)

### Expected GEMM Performance

**B200 estimates:**
- FP16: 1700-1900 TFLOPS (85-95% efficiency)
- FP8: 7000-7800 TFLOPS (87-97% efficiency)
- FP6: 9000-10000 TFLOPS (90-100% efficiency)
- FP4: 14000-16000 TFLOPS (87-100% efficiency)

---

## Migration Guide: Hopper → Blackwell

### What Stays the Same

**✓ Most Hopper code works on Blackwell:**
- TMA descriptors and operations
- Cluster operations
- Pipeline patterns
- Basic MMA atoms (FP16, BF16, FP8)

### What Changes

**New features to adopt:**

1. **Use TMEM for accumulators:**
```
# Hopper
frag_C = cute.make_rmem_tensor(...)

# Blackwell (better)
tmem_C = cute.make_tmem_tensor(...)
```

2. **Adopt FP6/FP4 for inference:**
```
# Hopper (FP8 is lowest)
mma = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)

# Blackwell (FP4 even faster)
mma = cute.make_mma_atom(cute.SM100_FP4_MMA_ATOM)
```

3. **Use 2-CTA instructions for large tiles:**
```
# Hopper (single CTA)
tile_m, tile_n = 128, 128

# Blackwell (2-CTA for larger tiles)
mma_2cta = cute.make_mma_atom_2cta(...)
tile_m, tile_n = 256, 256
```

---

## Current Limitations

**As of early 2025, Blackwell support is evolving:**

1. **Documentation may be incomplete**
   - Check latest NVIDIA documentation
   - APIs may change

2. **CuTe Python DSL support**
   - Full Blackwell support being added
   - Some features may be C++ only initially

3. **Availability**
   - B100/B200 availability limited
   - Cloud availability pending

4. **Tooling**
   - Profiling tools being updated
   - Best practices still emerging

---

## Summary

**Blackwell (SM100) key features:**
- 5th generation Tensor Cores (2× Hopper throughput)
- Tensor Memory (TMEM) for reduced register pressure
- FP4 and FP6 support for extreme quantization
- Block-scaled formats (MXFP4/6/8) for better accuracy
- 2-CTA instructions for larger cooperative tiles
- Enhanced TMA with TMEM support

**Performance tips:**
- Use TMEM for large accumulators
- Adopt FP6/FP4 for inference when possible
- Use block-scaled formats for better quantization
- Leverage 2-CTA instructions for large tiles
- Profile different precision formats

**Migration from Hopper:**
- Most code works unchanged
- Add TMEM usage for better performance
- Explore FP4/FP6 for new workloads
- Tune for 2× higher bandwidth

**When to use Blackwell:**
- Large model training (FP8/FP16)
- Extreme quantization inference (FP4/FP6)
- Memory-bound workloads (8 TB/s bandwidth)
- When cutting-edge performance needed

---

## Next Steps

- [Hopper SM90](./hopper_sm90.md) - Previous generation features
- [Choosing Atoms](./choosing_atoms.md) - Which atoms for which architecture
- [Advanced Examples](../10_examples/) - Real-world Blackwell kernels (when available)

---

## Further Reading

- [NVIDIA Blackwell Architecture Overview](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [Blackwell Whitepaper](https://resources.nvidia.com/en-us-blackwell) (when available)
- [PTX ISA for SM100](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS 4.x Blackwell Support](https://github.com/NVIDIA/cutlass)