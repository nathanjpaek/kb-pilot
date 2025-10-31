---
topic: "choosing_atoms"
difficulty: "intermediate"
related_topics: ["ampere_sm80", "hopper_sm90", "blackwell_sm100", "mma_atoms", "copy_atoms"]
---

# Choosing the Right Atoms for Your Workload

## Overview

With multiple GPU architectures (Ampere, Hopper, Blackwell) and many data types (FP16, BF16, FP8, FP4, etc.), choosing the right atoms for your kernel can be overwhelming. This guide helps you make informed decisions about which copy atoms and MMA atoms to use.

**Key factors:**
- Target GPU architecture
- Data types and precision requirements
- Performance vs accuracy trade-offs
- Memory bandwidth constraints
- Register pressure

---

## Quick Decision Tree

### For MMA Atoms
```
START
│
├─ What GPU architecture?
│  │
│  ├─ Ampere (SM80) ────────────────────┐
│  │                                      │
│  ├─ Hopper (SM90) ─────────────────────┤
│  │                                      │
│  └─ Blackwell (SM100) ─────────────────┤
│                                         │
└─> What precision do you need? <────────┘
    │
    ├─ Training (need accuracy)
    │  ├─ FP16/BF16 + FP32 accumulation ✓ (most common)
    │  └─ FP8 + FP16 accumulation (Hopper+, faster)
    │
    ├─ Inference (speed critical)
    │  ├─ FP8 (Hopper+, 4× faster than FP16)
    │  ├─ FP6 (Blackwell, even faster)
    │  └─ FP4 (Blackwell, maximum speed)
    │
    ├─ Scientific computing (need FP64)
    │  └─ FP64 MMA (Ampere+)
    │
    └─ Extreme quantization
       ├─ INT8 (all architectures)
       └─ INT4 (all architectures)
```

### For Copy Atoms
```
START
│
└─ What GPU architecture?
   │
   ├─ Ampere (SM80)
   │  └─ Use cp.async (async copy)
   │
   ├─ Hopper (SM90)
   │  └─ Use TMA (tensor memory accelerator)
   │
   └─ Blackwell (SM100)
      └─ Use TMA with TMEM support
```

---

## MMA Atom Selection Guide

### By Use Case

#### Deep Learning Training

**Best choice: FP16 inputs, FP32 accumulator**
```
# Ampere
mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
# 312 TFLOPS

# Hopper
mma = cute.make_mma_atom(cute.SM90_16x8x16_F16F16F32F32_TN)
# 989 TFLOPS

# Blackwell
mma = cute.make_mma_atom(cute.SM100_16x8x16_F16F16F32F32_TN)
# ~2000 TFLOPS
```

**Why:**
- FP16 is fast on tensor cores
- FP32 accumulator prevents precision loss during long accumulation
- Industry standard for training

**Alternative: BF16 for large models**
```
# Better dynamic range than FP16, same speed
mma = cute.make_mma_atom(cute.SM80_16x8x16_BF16BF16F32F32_TN)
```

**Why BF16:**
- Same range as FP32 (no overflow issues)
- Standard for large language models
- Slightly less precision than FP16 (usually okay)

#### Deep Learning Inference

**Best choice: FP8 (Hopper+)**
```
# Hopper
mma = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)
# 3958 TFLOPS (4× faster than FP16!)

# Blackwell
mma = cute.make_mma_atom(cute.SM100_16x8x32_E4M3E5M2F16_SS)
# ~8000 TFLOPS
```

**Why:**
- 4× faster than FP16 on Hopper
- Acceptable accuracy with proper scaling
- Standard for LLM inference

**Even faster: FP6/FP4 (Blackwell)**
```
# FP6 (Blackwell only)
mma = cute.make_mma_atom(cute.SM100_FP6_MMA_ATOM)
# ~10,000+ TFLOPS

# FP4 (Blackwell only)
mma = cute.make_mma_atom(cute.SM100_FP4_MMA_ATOM)
# ~16,000+ TFLOPS
```

**When to use:**
- Extreme quantization acceptable
- Maximum throughput needed
- Careful accuracy validation

#### Scientific Computing

**Best choice: FP64**
```
# Ampere (first to support FP64 tensor cores)
mma = cute.make_mma_atom(cute.SM80_16x8x4_F64F64F64F64_TN)
# 19.5 TFLOPS

# Hopper
mma = cute.make_mma_atom(cute.SM90_16x8x4_F64F64F64F64_TN)
# 67 TFLOPS

# Blackwell
mma = cute.make_mma_atom(cute.SM100_16x8x4_F64F64F64F64_TN)
# ~134 TFLOPS (estimated)
```

**Why:**
- Full precision needed
- Much faster than FP64 on CUDA cores
- Note: K dimension is only 4 (smaller than FP16's 16)

**Alternative: TF32 (faster, less precise)**
```
# TF32 on Ampere+ (automatic for FP32 operations)
mma = cute.make_mma_atom(cute.SM80_16x8x8_F32TF32TF32F32_TN)
# 156 TFLOPS (8× faster than FP64)
```

**When TF32 is acceptable:**
- 10-bit mantissa sufficient
- Need speed over precision
- Many iterations (precision loss accumulates)

#### Quantized Inference (INT8/INT4)

**INT8:**
```
# Ampere
mma = cute.make_mma_atom(cute.SM80_16x8x32_I8I8I32I32_TN)
# 624 TOPS

# Hopper
mma = cute.make_mma_atom(cute.SM90_16x8x64_I8I8I32I32_TN)
# 3958 TOPS (larger K dimension)

# Blackwell
mma = cute.make_mma_atom(cute.SM100_16x8x64_I8I8I32I32_TN)
# ~8000 TOPS
```

**INT4:**
```
# Ampere
mma = cute.make_mma_atom(cute.SM80_16x8x64_I4I4I32I32_TN)
# 1248 TOPS

# Hopper/Blackwell: Even higher throughput
```

**When to use:**
- Post-training quantization
- Need integer precision
- Have quantization/dequantization pipeline

---

## Copy Atom Selection Guide

### By Architecture

#### Ampere (SM80): Use cp.async

**Standard async copy:**
```
@cute.kernel
def ampere_copy():
    # Async copy atom
    async_atom = cute.make_ampere_async_copy_atom(
        element_type=cutlass.Float16,
        num_bits_per_copy=128  # Vectorized
    )
    
    # Create tiled copy
    tiled_copy = cute.make_tiled_copy(
        async_atom,
        thread_layout=cute.make_shape(32, 8),
        value_layout=cute.make_shape(4, 4)
    )
```

**Benefits:**
- Non-blocking (threads can do other work)
- Enables software pipelining
- Much faster than synchronous copy

**When to use:**
- All GMEM → SMEM transfers on Ampere
- Multi-stage pipelining (3+ stages recommended)

#### Hopper (SM90): Use TMA

**TMA copy:**
```
@cute.jit
def hopper_tma_copy(gmem: cute.Tensor):
    # Create TMA descriptor (once, outside kernel)
    tma_desc = cute.make_tma_descriptor(
        tensor=gmem,
        tile_shape=cute.make_shape(128, 64),
        element_type=cutlass.Float16
    )
    
    # Use in kernel
    @cute.kernel
    def kernel(tma_desc, gmem, smem):
        if cute.arch.thread_idx()[0] == 0:
            cute.copy(tma_desc, gmem[tile], smem)
        
        cute.arch.tma_wait()
        cute.arch.syncthreads()
```

**Benefits:**
- Hardware-accelerated transfer
- Minimal thread involvement
- Automatic tiling and swizzling
- Faster than cp.async

**When to use:**
- All GMEM → SMEM transfers on Hopper
- Large tiles (TMA excels at bulk transfers)

#### Blackwell (SM100): Use TMA with TMEM

**TMA with TMEM:**
```
@cute.kernel
def blackwell_tma_tmem(gmem, tma_desc):
    # Allocate TMEM
    tmem = cute.make_tmem_tensor(cute.make_shape(128, 64), cutlass.Float16)
    
    # TMA can load directly to TMEM
    if cute.arch.thread_idx()[0] == 0:
        cute.tma_load_to_tmem(tma_desc, gmem[tile], tmem)
    
    cute.arch.tma_wait()
    
    # Use data from TMEM
```

**When to use:**
- Large accumulators (TMEM reduces register pressure)
- GMEM → TMEM direct transfers
- Most GMEM transfers on Blackwell

---

## Data Type Selection

### FP16 vs BF16

**Use FP16 when:**
- General purpose training
- Need better precision (10-bit mantissa)
- Smaller models

**Use BF16 when:**
- Large language models
- Need FP32 range (avoid overflow)
- Slightly less precision acceptable

**Comparison:**

| Feature | FP16 | BF16 |
|---------|------|------|
| Mantissa | 10 bits | 7 bits |
| Exponent | 5 bits | 8 bits (same as FP32) |
| Range | ±65504 | ±3.4×10³⁸ (same as FP32) |
| Precision | Better | Worse |
| Overflow risk | Higher | Lower |
| Use case | General | LLMs |

### FP8 E4M3 vs E5M2

**Use E4M3 when:**
- Forward pass
- Need better precision
- Values are well-behaved

**Use E5M2 when:**
- Backward pass (gradients)
- Need larger range
- Have outliers

**Comparison:**

| Feature | E4M3 | E5M2 |
|---------|------|------|
| Mantissa | 3 bits | 2 bits |
| Exponent | 4 bits | 5 bits |
| Range | ±240 | ±57344 |
| Precision | Better | Worse |
| Use case | Forward | Backward |

### When to Use Lower Precision

**FP16 → FP8 migration checklist:**

✓ Hopper or Blackwell GPU  
✓ Inference workload (not training)  
✓ Can tolerate ~1-2% accuracy loss  
✓ Have scaling/quantization pipeline  
✓ Need maximum throughput  

**FP8 → FP6 migration checklist:**

✓ Blackwell GPU  
✓ Extreme quantization acceptable  
✓ Can tolerate ~3-5% accuracy loss  
✓ Block-scaled quantization available  
✓ Need even more throughput  

**FP6 → FP4 migration checklist:**

✓ Blackwell GPU  
✓ Ultra-low precision acceptable  
✓ Can tolerate ~5-10% accuracy loss  
✓ Careful validation done  
✓ Need absolute maximum throughput  

---

## Accumulator Type Selection

### FP16 vs FP32 Accumulator

**Use FP32 accumulator:**
```
# ✓ RECOMMENDED for most cases
mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
```

**When:**
- Long accumulation chains (large K dimension)
- Training (accuracy critical)
- Standard practice

**Use FP16 accumulator:**
```
# ⚠️ ONLY if needed
mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F16F16_TN)
```

**When:**
- Short accumulation chains (K < 1024)
- Register pressure critical
- Inference only
- Carefully validated

**Rule of thumb:** Always use FP32 accumulator unless you have a specific reason not to.

---

## Architecture-Specific Recommendations

### Ampere (A100)

**Recommended atoms:**
```
# Training: FP16 → FP32
mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
copy = cute.make_ampere_async_copy_atom(cutlass.Float16)

# Large models: BF16 → FP32
mma = cute.make_mma_atom(cute.SM80_16x8x16_BF16BF16F32F32_TN)
copy = cute.make_ampere_async_copy_atom(cutlass.BFloat16)

# Scientific: FP64
mma = cute.make_mma_atom(cute.SM80_16x8x4_F64F64F64F64_TN)

# Inference: INT8
mma = cute.make_mma_atom(cute.SM80_16x8x32_I8I8I32I32_TN)
```

**Best practices:**
- Always use cp.async for copies
- 3-stage pipeline minimum
- FP16 is sweet spot for performance

### Hopper (H100)

**Recommended atoms:**
```
# Training: FP16 → FP32 (2× faster than Ampere)
mma = cute.make_mma_atom(cute.SM90_16x8x16_F16F16F32F32_TN)
tma_desc = cute.make_tma_descriptor(...)

# Inference: FP8 → FP16 (4× faster than FP16!)
mma = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)
tma_desc = cute.make_tma_descriptor(...)

# Quantized: INT8 (larger K than Ampere)
mma = cute.make_mma_atom(cute.SM90_16x8x64_I8I8I32I32_TN)
```

**Best practices:**
- Always use TMA for copies
- FP8 is the killer feature (use it!)
- Warp specialization (producer/consumer)
- Consider clusters for large tiles

### Blackwell (B100/B200)

**Recommended atoms:**
```
# Training: FP16 → FP32 (2× faster than Hopper)
mma = cute.make_mma_atom(cute.SM100_16x8x16_F16F16F32F32_TN)

# Fast inference: FP8 (2× faster than Hopper)
mma = cute.make_mma_atom(cute.SM100_16x8x32_E4M3E5M2F16_SS)

# Extreme inference: FP6 (even faster)
mma = cute.make_mma_atom(cute.SM100_FP6_MMA_ATOM)

# Maximum throughput: FP4 (16000+ TFLOPS!)
mma = cute.make_mma_atom(cute.SM100_FP4_MMA_ATOM)
```

**Best practices:**
- Use TMEM for large accumulators
- FP4/FP6 for extreme quantization
- 2-CTA instructions for large tiles
- Block-scaled formats (MXFP) for better accuracy

---

## Common Scenarios

### Scenario 1: LLM Training (Hopper)
```
# Configuration
GPU: H100
Model: 70B parameter LLM
Batch size: 4
Sequence length: 4096

# Recommended atoms
mma = cute.make_mma_atom(cute.SM90_16x8x16_BF16BF16F32F32_TN)
# BF16 for range, FP32 accumulator for accuracy

tma_desc = cute.make_tma_descriptor(...)
# TMA for fastest data movement

# Why
- BF16 handles large model range
- FP32 accumulator critical for training
- TMA essential on Hopper
```

### Scenario 2: LLM Inference (Hopper)
```
# Configuration
GPU: H100
Model: 70B parameter LLM
Batch size: 16
Max tokens: 512

# Recommended atoms
mma = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)
# FP8 for 4× speedup

tma_desc = cute.make_tma_descriptor(...)

# With quantization
- Quantize weights to E5M2
- Activations in E4M3
- Per-channel scaling for accuracy
- 4× faster than FP16, minimal accuracy loss
```

### Scenario 3: Vision Model Training (Ampere)
```
# Configuration
GPU: A100
Model: ResNet-50 / ViT
Batch size: 256

# Recommended atoms
mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
# FP16 sufficient for vision

async_copy = cute.make_ampere_async_copy_atom(cutlass.Float16)

# Why
- FP16 is standard for vision
- Smaller models than LLMs (FP16 range sufficient)
- cp.async on Ampere
```

### Scenario 4: Quantized Inference (Blackwell)
```
# Configuration
GPU: B200
Model: Ultra-compressed LLM
Target: Maximum throughput

# Recommended atoms
mma = cute.make_mma_atom(cute.SM100_MXFP4_MMA_ATOM)
# Block-scaled FP4 for accuracy + speed

tma_desc = cute.make_tma_descriptor(...)

# With block scaling
- MXFP4 (microscaling FP4)
- Different scale per block (32 or 64 elements)
- Better accuracy than uniform FP4
- 16000+ TFLOPS throughput
```

---

## Performance vs Accuracy Trade-offs

### Precision Hierarchy (Slowest to Fastest)
```
FP64         ← Highest accuracy, slowest
  ↓
FP32 (CUDA)
  ↓
TF32         ← ~8× faster than FP32 CUDA cores
  ↓
BF16 / FP16  ← 2× faster than TF32
  ↓
FP8          ← 4× faster than FP16 (Hopper+)
  ↓
FP6          ← ~2× faster than FP8 (Blackwell)
  ↓
FP4          ← ~2× faster than FP6 (Blackwell)
  ↓          ← Lowest accuracy, fastest
```

### Typical Accuracy Impact

**From FP16 baseline:**

| Precision | Relative Accuracy | Use Case |
|-----------|-------------------|----------|
| FP32 | Reference | Scientific computing |
| BF16 | ~99.9% | LLM training |
| FP16 | 100% (baseline) | Standard training |
| FP8 (with scaling) | ~98-99% | LLM inference |
| FP6 (with block scaling) | ~95-97% | Aggressive inference |
| FP4 (with block scaling) | ~90-95% | Extreme inference |

**Note:** Accuracy highly workload-dependent. Always validate!

---

## Decision Matrix

### Quick Reference Table

| Workload | GPU | Recommended MMA | Recommended Copy |
|----------|-----|-----------------|------------------|
| LLM Training | H100 | BF16→FP32 | TMA |
| LLM Training | A100 | BF16→FP32 | cp.async |
| LLM Inference | H100 | FP8→FP16 | TMA |
| LLM Inference | B200 | FP4→FP16 | TMA+TMEM |
| Vision Training | A100 | FP16→FP32 | cp.async |
| Vision Inference | H100 | FP8→FP16 | TMA |
| Scientific | H100 | FP64→FP64 | TMA |
| Quantized | Any | INT8→INT32 | arch-specific |

---

## Validation Checklist

Before deploying a kernel with chosen atoms:

**✓ Correctness:**
- [ ] Reference CPU implementation matches GPU output
- [ ] Numerical stability verified (no NaN/Inf)
- [ ] Edge cases tested (zeros, very small/large values)

**✓ Performance:**
- [ ] Achieves >80% of theoretical peak
- [ ] Memory bandwidth efficiently utilized
- [ ] No unnecessary synchronization

**✓ Accuracy:**
- [ ] Meets accuracy requirements for application
- [ ] Tested on representative data
- [ ] Outliers handled properly

**✓ Generalization:**
- [ ] Works for different input sizes
- [ ] Tested on multiple problem sizes
- [ ] Handles edge dimensions correctly

---

## Summary

**Key decision factors:**

1. **Architecture first:** Different atoms for Ampere/Hopper/Blackwell
2. **Use case second:** Training vs inference vs scientific
3. **Precision third:** Balance speed and accuracy
4. **Always validate:** Profile and verify accuracy

**Golden rules:**

- **Ampere:** cp.async + FP16/BF16 + FP32 accumulator
- **Hopper:** TMA + FP8 (inference) or FP16/BF16 (training)
- **Blackwell:** TMA+TMEM + FP4/FP6 (inference) or FP16 (training)

**When in doubt:**
- Training: FP16→FP32 (safe default)
- Inference: FP8→FP16 on Hopper+ (best speed/accuracy)
- Scientific: FP64→FP64 (accuracy critical)

---

## Next Steps

- [Ampere SM80](./ampere_sm80.md) - Ampere-specific details
- [Hopper SM90](./hopper_sm90.md) - Hopper-specific details
- [Blackwell SM100](./blackwell_sm100.md) - Blackwell-specific details
- [MMA Atoms](../04_operations/mma_atoms.md) - Deep dive into MMA operations

---

## Further Reading

- [NVIDIA Data Type Comparison](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [FP8 Training Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [Quantization Best Practices](https://arxiv.org/abs/2208.07339)