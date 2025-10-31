---
topic: "ampere_sm80"
difficulty: "intermediate"
related_topics: ["mma_atoms", "copy_atoms", "choosing_atoms"]
---

# Ampere (SM80) Architecture Features

## Overview

**Ampere** (compute capability SM80) introduced several major improvements for deep learning workloads, including enhanced Tensor Cores, async copy instructions, and structural sparsity support. The flagship Ampere GPU is the **NVIDIA A100**.

**Key innovations:**
- 3rd generation Tensor Cores with TF32, FP64, and new data types
- Asynchronous copy (`cp.async`) for overlapping data movement and computation
- Structural sparsity support (2:4 sparsity)
- Multi-Instance GPU (MIG)

---

## Ampere Key Specifications

### A100 GPU Specifications

**Compute:**
- 108 SMs (Streaming Multiprocessors)
- 6912 CUDA cores
- 432 3rd generation Tensor Cores

**Memory:**
- 40GB or 80GB HBM2e
- Memory bandwidth: 1.5 TB/s (40GB) or 2.0 TB/s (80GB)
- 40MB L2 cache
- 192KB shared memory per SM (configurable)

**Performance (FP16 with Tensor Cores):**
- 312 TFLOPS (dense)
- 624 TFLOPS (sparse with 2:4 structured sparsity)

---

## Tensor Core Features

### Supported Data Types

**FP16 (Float16):**
```
@cute.kernel
def ampere_fp16_mma():
    # 16x8x16 FP16 MMA
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Inputs: FP16, Accumulator: FP32
    # 312 TFLOPS on A100
```

**BF16 (BFloat16):**
```
@cute.kernel
def ampere_bf16_mma():
    # 16x8x16 BF16 MMA
    mma = cute.make_mma_atom(cute.SM80_16x8x16_BF16BF16F32F32_TN)
    
    # Same performance as FP16 (312 TFLOPS)
    # Better dynamic range, slightly less precision
```

**TF32 (TensorFloat32):**
```
@cute.kernel
def ampere_tf32_mma():
    # 16x8x8 TF32 MMA (automatic for FP32)
    mma = cute.make_mma_atom(cute.SM80_16x8x8_F32TF32TF32F32_TN)
    
    # 10-bit mantissa (vs 23-bit for FP32)
    # 156 TFLOPS on A100
    # Automatically used for FP32 GEMM operations
```

**FP32 (Float32):**
```
@cute.kernel
def ampere_fp32_mma():
    # FP32 on CUDA cores (not Tensor Cores)
    # 19.5 TFLOPS on A100
    # Use TF32 on Tensor Cores instead for 8x speedup
    pass
```

**FP64 (Float64):**
```
@cute.kernel
def ampere_fp64_mma():
    # 16x8x4 FP64 MMA
    mma = cute.make_mma_atom(cute.SM80_16x8x4_F64F64F64F64_TN)
    
    # 19.5 TFLOPS on A100 (same as FP32 CUDA cores)
    # First Tensor Core support for FP64!
```

**INT8 (Signed 8-bit Integer):**
```
@cute.kernel
def ampere_int8_mma():
    # 16x8x32 INT8 MMA
    mma = cute.make_mma_atom(cute.SM80_16x8x32_I8I8I32I32_TN)
    
    # 624 TOPS on A100
    # For quantized inference
```

**INT4 (4-bit Integer):**
```
@cute.kernel
def ampere_int4_mma():
    # 16x8x64 INT4 MMA
    mma = cute.make_mma_atom(cute.SM80_16x8x64_I4I4I32I32_TN)
    
    # 1248 TOPS on A100
    # Ultra-low precision quantization
```

### MMA Atom Dimensions by Type

| Data Type | M×N×K | Throughput (A100) |
|-----------|-------|-------------------|
| FP16/BF16 | 16×8×16 | 312 TFLOPS |
| TF32      | 16×8×8  | 156 TFLOPS |
| FP64      | 16×8×4  | 19.5 TFLOPS |
| INT8      | 16×8×32 | 624 TOPS |
| INT4      | 16×8×64 | 1248 TOPS |

**Note the pattern:** Lower precision → larger K dimension → higher throughput.

---

## Async Copy (cp.async)

### What is cp.async?

**cp.async** allows **asynchronous copying from global memory to shared memory** without blocking threads.

**Traditional synchronous copy:**
```
@cute.kernel
def sync_copy(gmem: cute.Tensor, smem: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Thread blocked until data arrives
    smem[tid] = gmem[tid]
    
    # Can't do useful work while waiting
    cute.arch.syncthreads()
```

**Ampere async copy:**
```
@cute.kernel
def async_copy(gmem: cute.Tensor, smem: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Issue async copy (doesn't block thread!)
    async_atom = cute.make_ampere_async_copy_atom(cutlass.Float16)
    cute.copy(async_atom, gmem[tid], smem[tid])
    
    # Thread can do other work while copy happens in background
    do_useful_computation()
    
    # Wait for copy to complete
    cute.arch.cp_async_wait_all()
    cute.arch.syncthreads()
```

### cp.async Copy Atom
```
@cute.kernel
def create_async_copy_atom():
    # Create async copy atom for FP16
    async_atom = cute.make_ampere_async_copy_atom(
        element_type=cutlass.Float16,
        num_bits_per_copy=128  # 128-bit (8 FP16 values at once)
    )
    
    # Or for FP32
    async_atom_fp32 = cute.make_ampere_async_copy_atom(
        element_type=cutlass.Float32,
        num_bits_per_copy=128  # 4 FP32 values at once
    )
```

### Software Pipelining with cp.async

**Most important use case: overlap copy with compute.**
```
@cute.kernel
def pipelined_gemm(gmem_A: cute.Tensor, gmem_B: cute.Tensor, gmem_C: cute.Tensor):
    # Create async copy atoms
    async_copy = cute.make_ampere_async_copy_atom(cutlass.Float16)
    
    # Create MMA atom
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    # Allocate shared memory (double buffering)
    smem_A = [
        cute.make_smem_tensor(cute.make_shape(128, 32), cutlass.Float16),
        cute.make_smem_tensor(cute.make_shape(128, 32), cutlass.Float16)
    ]
    smem_B = [
        cute.make_smem_tensor(cute.make_shape(32, 128), cutlass.Float16),
        cute.make_smem_tensor(cute.make_shape(32, 128), cutlass.Float16)
    ]
    
    # Initialize accumulator
    frag_C = cute.make_rmem_tensor(cute.make_shape(128, 128), cutlass.Float32)
    frag_C.store(0.0)
    
    num_k_tiles = 16
    
    # ========================================
    # Prologue: Load first tile
    # ========================================
    write_stage = 0
    cute.copy(async_copy, gmem_A[0], smem_A[write_stage])
    cute.copy(async_copy, gmem_B[0], smem_B[write_stage])
    cute.arch.cp_async_commit_group()
    
    # ========================================
    # Main loop: Overlapped copy + compute
    # ========================================
    for k in range(num_k_tiles):
        read_stage = write_stage
        write_stage = 1 - write_stage  # Swap buffers
        
        # Issue async copy for NEXT tile (k+1)
        if k + 1 < num_k_tiles:
            cute.copy(async_copy, gmem_A[k + 1], smem_A[write_stage])
            cute.copy(async_copy, gmem_B[k + 1], smem_B[write_stage])
            cute.arch.cp_async_commit_group()
        
        # Wait for CURRENT tile (k) to arrive
        cute.arch.cp_async_wait_group(0)  # Wait for oldest group
        cute.arch.syncthreads()
        
        # Load from shared to registers
        frag_A = load_fragment(smem_A[read_stage])
        frag_B = load_fragment(smem_B[read_stage])
        
        # Compute MMA (while NEXT tile copies in background!)
        cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)
    
    # Store result
    store_result(frag_C, gmem_C)
```

**Key benefit:** Compute happens while next tile copies, hiding memory latency!

---

## Multi-Stage Pipeline

### 3-Stage Pipeline (Recommended)

**Even better: use 3+ buffers for deeper pipelining.**
```
@cute.kernel
def three_stage_pipeline(
    gmem_A: cute.Tensor,
    gmem_B: cute.Tensor,
    num_stages: cutlass.Constexpr = 3
):
    # Allocate multiple stages of shared memory
    smem_stages_A = [
        cute.make_smem_tensor(cute.make_shape(128, 32), cutlass.Float16)
        for _ in range(num_stages)
    ]
    smem_stages_B = [
        cute.make_smem_tensor(cute.make_shape(32, 128), cutlass.Float16)
        for _ in range(num_stages)
    ]
    
    async_copy = cute.make_ampere_async_copy_atom(cutlass.Float16)
    
    # ========================================
    # Prologue: Fill pipeline
    # ========================================
    for stage in range(num_stages - 1):
        cute.copy(async_copy, gmem_A[stage], smem_stages_A[stage])
        cute.copy(async_copy, gmem_B[stage], smem_stages_B[stage])
        cute.arch.cp_async_commit_group()
    
    # ========================================
    # Main loop
    # ========================================
    for k in range(num_k_tiles):
        stage_idx = k % num_stages
        
        # Issue copy for tile k + (num_stages - 1)
        if k + num_stages - 1 < num_k_tiles:
            cute.copy(async_copy, gmem_A[k + num_stages - 1], smem_stages_A[stage_idx])
            cute.copy(async_copy, gmem_B[k + num_stages - 1], smem_stages_B[stage_idx])
            cute.arch.cp_async_commit_group()
        
        # Wait for tile k to be ready
        cute.arch.cp_async_wait_group(num_stages - 2)
        cute.arch.syncthreads()
        
        # Compute with tile k
        compute_mma(smem_stages_A[stage_idx], smem_stages_B[stage_idx])
```

**Benefits of 3+ stages:**
- Even better latency hiding
- Keeps Tensor Cores fully utilized
- Requires more shared memory

---

## Shared Memory Configuration

### Ampere Shared Memory Sizes

**A100 shared memory per SM:**
- Maximum: 164 KB (configurable)
- Typical configurations: 8 KB, 16 KB, 32 KB, 48 KB, 64 KB, 100 KB, 164 KB

### Setting Shared Memory Size
```
@cute.jit
def launch_with_smem_config(gmem_A, gmem_B, gmem_C):
    # Calculate required shared memory
    smem_size_per_stage = 128 * 32 * 2 + 32 * 128 * 2  # FP16 = 2 bytes
    num_stages = 3
    total_smem = smem_size_per_stage * num_stages
    
    # total_smem = ~48 KB
    
    # Launch with explicit shared memory size
    my_kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[256, 1, 1],
        smem=total_smem
    )(gmem_A, gmem_B, gmem_C)
```

### Shared Memory vs Register Trade-off

**More shared memory = fewer registers per thread:**
```
# 48 KB shared memory config
# → More registers available per thread
# → Higher occupancy possible

# 164 KB shared memory config
# → Fewer registers available per thread
# → May reduce occupancy
```

**Profile to find optimal configuration!**

---

## Structured Sparsity (2:4 Sparsity)

### What is 2:4 Sparsity?

**2:4 sparsity:** In every group of 4 values, exactly 2 are non-zero (50% sparsity).

**Example:**
```
Dense weights:
[1.2, 0.3, -0.8, 2.1]

2:4 sparse weights (keep 2 largest):
[1.2, 0.0, 0.0, 2.1]  ← 2 non-zero, 2 zero
```

**Performance benefit:** 2× speedup with specialized Tensor Core instructions!

### Sparse MMA Atoms
```
@cute.kernel
def sparse_mma_ampere():
    # Sparse 16x8x32 MMA (note K=32, double the dense K=16)
    sparse_mma = cute.make_mma_atom(cute.SM80_16x8x32_F16F16F32F32_TN_SPARSE)
    
    # A matrix must be 2:4 sparse
    # B matrix is dense
    # 2× faster than dense: 624 TFLOPS vs 312 TFLOPS
```

**Note:** Requires preprocessing weights to 2:4 sparsity format and metadata.

---

## TF32 (TensorFloat32)

### What is TF32?

**TF32** is Ampere's automatic speedup for FP32 operations:
- **Range:** Same as FP32 (8-bit exponent)
- **Precision:** 10-bit mantissa (vs 23-bit for FP32)
- **Performance:** ~8× faster than FP32 CUDA cores

### Automatic TF32 Usage
```
@cute.kernel
def automatic_tf32():
    # When you use FP32 MMA atom, Ampere automatically uses TF32
    mma = cute.make_mma_atom(cute.SM80_16x8x8_F32TF32TF32F32_TN)
    
    # Inputs are FP32, but Tensor Cores use TF32 internally
    # 156 TFLOPS (vs 19.5 TFLOPS for FP32 CUDA cores)
```

### When to Use TF32

**✅ Use TF32 when:**
- Need more range than FP16
- Can tolerate slightly less precision than FP32
- Want faster training than FP32

**❌ Don't use TF32 when:**
- Need full FP32 precision (e.g., financial calculations)
- Can use FP16/BF16 instead (2× faster than TF32)

---

## Ampere-Specific Best Practices

### Practice 1: Use Async Copy

**Always use cp.async for GMEM → SMEM transfers:**
```
# ✓ GOOD: Async copy
async_atom = cute.make_ampere_async_copy_atom(cutlass.Float16)
cute.copy(async_atom, gmem, smem)

# ✗ BAD: Sync copy (wastes cycles)
for i in range(n):
    smem[i] = gmem[i]
```

### Practice 2: Pipeline with 3+ Stages

**Use multi-stage pipelining:**
```
# ✓ GOOD: 3-4 stage pipeline
num_stages = 3

# ✗ BAD: No pipelining (no overlap)
num_stages = 1
```

### Practice 3: Use FP16/BF16, Not FP32

**Prefer mixed precision:**
```
# ✓ GOOD: FP16 compute, FP32 accumulate
mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)

# ✗ BAD: FP32 compute (8× slower on Tensor Cores)
mma = cute.make_mma_atom(cute.SM80_16x8x8_F32F32F32F32_TN)
```

### Practice 4: Optimize Shared Memory Size

**Profile different shared memory configurations:**
```
# Try different tile sizes and stages
tile_sizes = [(128, 128, 32), (256, 128, 32), (128, 256, 32)]
num_stages_options = [2, 3, 4]

for tile in tile_sizes:
    for stages in num_stages_options:
        benchmark(tile, stages)
```

---

## Ampere Example: Optimized GEMM

### Complete Ampere GEMM Kernel
```
@cute.kernel
def ampere_optimized_gemm(
    gmem_A: cute.Tensor,  # M×K, FP16
    gmem_B: cute.Tensor,  # K×N, FP16
    gmem_C: cute.Tensor   # M×N, FP32
):
    # ========================================
    # Configuration
    # ========================================
    tile_m, tile_n, tile_k = 128, 128, 32
    num_stages = 3
    num_threads = 256
    
    # ========================================
    # Create Atoms
    # ========================================
    # Async copy atom (cp.async)
    async_copy_atom = cute.make_ampere_async_copy_atom(
        cutlass.Float16,
        num_bits_per_copy=128  # 8 FP16 values
    )
    
    # Create tiled copy
    tiled_copy = cute.make_tiled_copy(
        async_copy_atom,
        thread_layout=cute.make_shape(32, 8),
        value_layout=cute.make_shape(4, 4)
    )
    
    # MMA atom (FP16 inputs, FP32 accumulator)
    mma_atom = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        thread_layout=cute.make_shape(4, 4, 1)
    )
    
    # ========================================
    # Allocate Shared Memory (Multi-Stage)
    # ========================================
    smem_A_stages = [
        cute.make_smem_tensor(cute.make_shape(tile_m, tile_k + 8), cutlass.Float16)
        for _ in range(num_stages)
    ]
    smem_B_stages = [
        cute.make_smem_tensor(cute.make_shape(tile_k, tile_n + 8), cutlass.Float16)
        for _ in range(num_stages)
    ]
    
    # ========================================
    # Get Thread Slices
    # ========================================
    tid = cute.arch.thread_idx()[0]
    thread_copy = tiled_copy.get_slice(tid)
    thread_mma = tiled_mma.get_slice(tid)
    
    # ========================================
    # Create Register Fragments
    # ========================================
    frag_A = tiled_mma.make_fragment_A(cute.make_shape(tile_m, tile_k))
    frag_B = tiled_mma.make_fragment_B(cute.make_shape(tile_k, tile_n))
    frag_C = tiled_mma.make_fragment_C(cute.make_shape(tile_m, tile_n))
    
    frag_C.store(0.0)
    
    # ========================================
    # Get Block Tile
    # ========================================
    block_m = cute.arch.block_idx()[0]
    block_n = cute.arch.block_idx()[1]
    
    block_A = gmem_A[block_m]
    block_B = gmem_B[:, block_n]
    
    num_k_tiles = cute.size(gmem_A, mode=1) // tile_k
    
    # ========================================
    # Prologue: Fill Pipeline
    # ========================================
    for stage in range(num_stages - 1):
        if stage < num_k_tiles:
            # Partition and copy
            src_A = thread_copy.partition_S(block_A[:, stage])
            dst_A = thread_copy.partition_D(smem_A_stages[stage])
            
            src_B = thread_copy.partition_S(block_B[stage, :])
            dst_B = thread_copy.partition_D(smem_B_stages[stage])
            
            cute.copy(tiled_copy, src_A, dst_A)
            cute.copy(tiled_copy, src_B, dst_B)
            
            # Commit async copy group
            cute.arch.cp_async_commit_group()
    
    # ========================================
    # Main Loop: Pipelined Compute
    # ========================================
    for k in range(num_k_tiles):
        stage_idx = k % num_stages
        
        # Issue async copy for future tile
        if k + num_stages - 1 < num_k_tiles:
            future_k = k + num_stages - 1
            
            src_A = thread_copy.partition_S(block_A[:, future_k])
            dst_A = thread_copy.partition_D(smem_A_stages[stage_idx])
            
            src_B = thread_copy.partition_S(block_B[future_k, :])
            dst_B = thread_copy.partition_D(smem_B_stages[stage_idx])
            
            cute.copy(tiled_copy, src_A, dst_A)
            cute.copy(tiled_copy, src_B, dst_B)
            
            cute.arch.cp_async_commit_group()
        
        # Wait for current tile
        cute.arch.cp_async_wait_group(num_stages - 2)
        cute.arch.syncthreads()
        
        # Load from SMEM to registers
        thread_A = thread_mma.partition_A(smem_A_stages[stage_idx])
        thread_B = thread_mma.partition_B(smem_B_stages[stage_idx])
        
        cute.copy(thread_A, frag_A)
        cute.copy(thread_B, frag_B)
        
        # MMA (overlapped with next tile copy!)
        cute.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
    
    # ========================================
    # Epilogue: Store Results
    # ========================================
    block_C = gmem_C[block_m, block_n]
    thread_C = thread_mma.partition_C(block_C)
    
    cute.copy(frag_C, thread_C)
```

---

## Performance Tuning Tips

### Tile Size Selection

**Common Ampere tile sizes:**
```
# Smaller tiles (more blocks, better occupancy)
(64, 64, 32)   # Good for small matrices

# Medium tiles (balance)
(128, 128, 32)  # Most common, good default

# Larger tiles (fewer blocks, better data reuse)
(256, 128, 32)  # Good for large matrices
(128, 256, 32)
```

### Number of Stages

**Stage count trade-offs:**
```
num_stages = 2   # Minimal pipelining, less SMEM
num_stages = 3   # Recommended, good balance
num_stages = 4+  # More latency hiding, more SMEM
```

### Measuring Performance
```
@cute.jit
def benchmark_ampere_gemm():
    import time
    
    M, N, K = 4096, 4096, 4096
    
    # Create tensors
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    
    # Warm up
    for _ in range(10):
        ampere_gemm.launch(...)(A, B, C)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    num_iterations = 100
    for _ in range(num_iterations):
        ampere_gemm.launch(...)(A, B, C)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Calculate TFLOPS
    flops = 2 * M * N * K * num_iterations
    tflops = (flops / elapsed) / 1e12
    
    # A100 theoretical peak: 312 TFLOPS (FP16)
    efficiency = (tflops / 312.0) * 100
    
    print(f"Performance: {tflops:.2f} TFLOPS ({efficiency:.1f}% of peak)")
```

---

## Summary

**Ampere (SM80) key features:**
- 3rd generation Tensor Cores (FP16, BF16, TF32, FP64, INT8, INT4)
- Async copy (cp.async) for latency hiding
- 2:4 structured sparsity (2× speedup)
- Up to 164 KB shared memory per SM

**Performance tips:**
- Use async copy for all GMEM → SMEM transfers
- Pipeline with 3+ stages
- Prefer FP16/BF16 over FP32 (2× faster)
- Use TF32 if you need FP32 range
- Tune tile size and stages for your workload

**Typical Ampere GEMM performance:**
- Good implementation: 250-280 TFLOPS (80-90% of peak)
- Expert implementation: 290-310 TFLOPS (93-99% of peak)
- cuBLAS: ~310 TFLOPS (99% of peak)

---

## Next Steps

- [Hopper SM90](./hopper_sm90.md) - Next generation features (TMA, FP8)
- [Choosing Atoms](./choosing_atoms.md) - Which atoms for which architecture
- [MMA Atoms](../04_operations/mma_atoms.md) - Deep dive into MMA operations

---

## Further Reading

- [NVIDIA A100 Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [cp.async Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)