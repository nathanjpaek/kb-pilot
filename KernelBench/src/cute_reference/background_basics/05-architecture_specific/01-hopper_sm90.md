---
topic: "hopper_sm90"
difficulty: "advanced"
related_topics: ["ampere_sm80", "mma_atoms", "copy_atoms"]
---

# Hopper (SM90) Architecture Features

## Overview

**Hopper** (compute capability SM90) represents a massive leap in AI performance, introducing the Tensor Memory Accelerator (TMA), FP8 precision, and enhanced thread block clusters. The flagship Hopper GPU is the **NVIDIA H100**.

**Key innovations:**
- 4th generation Tensor Cores with FP8 support
- Tensor Memory Accelerator (TMA) for hardware-accelerated data movement
- Thread Block Clusters for cross-CTA cooperation
- Enhanced async pipelines
- Tensor Memory (TMEM) on later variants

---

## Hopper Key Specifications

### H100 GPU Specifications

**Compute:**
- 132 SMs (Streaming Multiprocessors)
- 16,896 CUDA cores
- 528 4th generation Tensor Cores

**Memory:**
- 80GB HBM3
- Memory bandwidth: 3.35 TB/s (2.2× A100)
- 60MB L2 cache
- 256KB shared memory per SM

**Performance:**
- FP16/BF16: 989 TFLOPS (3.17× A100)
- FP8: 3958 TFLOPS (12.7× A100!)
- TF32: 495 TFLOPS (3.17× A100)
- FP64: 67 TFLOPS (3.44× A100)
- INT8: 3958 TOPS

---

## FP8 Tensor Cores

### What is FP8?

**FP8** is an 8-bit floating-point format with two variants:
- **E4M3** (4-bit exponent, 3-bit mantissa): Better precision, for forward pass
- **E5M2** (5-bit exponent, 2-bit mantissa): Better range, for backward pass

### FP8 MMA Atoms
```
@cute.kernel
def hopper_fp8_mma():
    # 16x8x32 FP8 MMA (note K=32, double FP16's K=16)
    mma = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)
    
    # A: E4M3 (forward activations)
    # B: E5M2 (forward weights or gradients)
    # C/D: FP16 accumulator
    
    # 3958 TFLOPS - 4× faster than FP16!
```

### E4M3 vs E5M2

**E4M3 (4-bit exponent, 3-bit mantissa):**
```
Range: ~±240
Precision: Better than E5M2
Use for: Forward pass activations, intermediate values
```

**E5M2 (5-bit exponent, 2-bit mantissa):**
```
Range: ~±57344 (much larger)
Precision: Worse than E4M3
Use for: Backward pass gradients, weights
```

### FP8 Usage Example
```
@cute.kernel
def fp8_gemm(
    gmem_A: cute.Tensor,  # E4M3
    gmem_B: cute.Tensor,  # E5M2
    gmem_C: cute.Tensor   # FP16 output
):
    # FP8 MMA atom
    mma = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)
    
    tiled_mma = cute.make_tiled_mma(mma, ...)
    
    # Create fragments
    frag_A = tiled_mma.make_fragment_A(...)  # E4M3
    frag_B = tiled_mma.make_fragment_B(...)  # E5M2
    frag_C = tiled_mma.make_fragment_C(...)  # FP16 accumulator
    
    frag_C.store(0.0)
    
    # Load (with proper quantization/scaling)
    cute.copy(gmem_A, frag_A)
    cute.copy(gmem_B, frag_B)
    
    # MMA
    cute.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
    
    # Store
    cute.copy(frag_C, gmem_C)
```

**Important:** FP8 typically requires careful scaling to maintain accuracy!

---

## Tensor Memory Accelerator (TMA)

### What is TMA?

**TMA** is a hardware unit that performs bulk memory transfers from global memory to shared memory **without involving threads**. This is a major architectural change!

**Benefits:**
- Much faster than cp.async (Ampere)
- Minimal thread involvement
- Supports complex tiling and swizzling automatically
- Reduces register pressure

### Traditional Copy vs TMA

**Ampere (cp.async):**
```
@cute.kernel
def ampere_copy(gmem, smem):
    # Each thread issues async copy for its portion
    async_atom = cute.make_ampere_async_copy_atom(...)
    cute.copy(async_atom, gmem[thread_portion], smem[thread_portion])
    
    # Wait for completion
    cute.arch.cp_async_wait_all()
```

**Hopper (TMA):**
```
@cute.kernel
def hopper_tma_copy(gmem, smem):
    # TMA copies ENTIRE tile with minimal thread involvement
    tma_desc = cute.make_tma_descriptor(...)
    
    # Single "instruction" copies whole tile
    cute.copy(tma_desc, gmem[tile], smem)
    
    # Wait for TMA
    cute.arch.tma_wait()
```

### Creating TMA Descriptors

**TMA descriptor encapsulates all copy information:**
```
@cute.jit
def create_tma_descriptor(gmem_tensor: cute.Tensor):
    # Define tile shape to copy
    tile_shape = cute.make_shape(128, 64)
    
    # Create TMA descriptor
    tma_desc = cute.make_tma_descriptor(
        tensor=gmem_tensor,
        tile_shape=tile_shape,
        element_type=cutlass.Float16
    )
    
    return tma_desc
```

### TMA Copy in Kernel
```
@cute.kernel
def tma_load_example(
    gmem_A: cute.Tensor,
    smem_A: cute.Tensor,
    tma_desc_A: cute.TMADescriptor
):
    # TMA load from global to shared
    # Only one thread (typically thread 0) issues TMA
    if cute.arch.thread_idx()[0] == 0:
        cute.copy(tma_desc_A, gmem_A[block_tile], smem_A)
    
    # All threads wait for TMA to complete
    cute.arch.tma_arrive()
    cute.arch.tma_wait()
    cute.arch.syncthreads()
    
    # Now all threads can use smem_A
```

### TMA Store

**TMA can also store from shared to global:**
```
@cute.kernel
def tma_store_example(
    smem_C: cute.Tensor,
    gmem_C: cute.Tensor,
    tma_desc_C: cute.TMADescriptor
):
    # All threads write to SMEM
    compute_and_store_to_smem(smem_C)
    
    cute.arch.syncthreads()
    
    # TMA store from shared to global
    if cute.arch.thread_idx()[0] == 0:
        cute.copy(tma_desc_C, smem_C, gmem_C[block_tile])
    
    # Wait for store to complete
    cute.arch.tma_store_wait()
```

---

## Thread Block Clusters

### What are Clusters?

**Thread Block Clusters** allow multiple thread blocks (CTAs) to cooperate and share data through distributed shared memory.

**Traditional (pre-Hopper):**
- Thread blocks are independent
- No communication between blocks

**Hopper Clusters:**
- Up to 8 thread blocks form a cluster
- Can access each other's shared memory
- Synchronize across blocks in cluster

### Creating Clusters
```
@cute.jit
def launch_with_cluster(data: cute.Tensor):
    # Define cluster dimensions (2×2×1 = 4 CTAs per cluster)
    cluster_shape = cute.make_shape(2, 2, 1)
    
    # Launch with cluster
    my_kernel.launch(
        grid=[num_blocks_x, num_blocks_y, 1],
        block=[num_threads, 1, 1],
        cluster=cluster_shape
    )(data)
```

### Accessing Remote Shared Memory
```
@cute.kernel
def cluster_communication():
    # Allocate shared memory
    smem = cute.make_smem_tensor(cute.make_shape(128, 64), cutlass.Float32)
    
    # Get cluster coordinates
    cluster_id_x = cute.arch.cluster_idx()[0]
    cluster_id_y = cute.arch.cluster_idx()[1]
    
    # Get block's position within cluster
    block_rank_in_cluster = cute.arch.block_rank_in_cluster()
    
    # Access THIS block's shared memory (normal)
    local_val = smem[tid, 0]
    
    # Access ANOTHER block's shared memory in cluster
    remote_block_rank = 1  # Access block 1's SMEM
    remote_val = cute.cluster_smem_read(smem, remote_block_rank, tid, 0)
    
    # Cluster-wide barrier
    cute.arch.cluster_sync()
```

### Use Cases for Clusters

**1. Larger effective tiles:**
```
# Each CTA in 2×2 cluster processes 128×128 tile
# Cluster effectively processes 256×256 tile
# Can share partial results across CTAs
```

**2. Multi-CTA reductions:**
```
# Each CTA computes partial sum
# Cluster-wide reduction combines results
```

**3. Overlapped epilogue:**
```
# While one CTA computes, another can store results
```

---

## Enhanced Warp Specialization

### Warp Roles in Hopper

**Hopper enables explicit warp specialization:**
```
@cute.kernel
def warp_specialized_gemm():
    warp_id = cute.arch.thread_idx()[0] // 32
    
    # Assign roles to warps
    if warp_id == 0:
        # Warp 0: Producer (loads data with TMA)
        producer_warp()
    elif warp_id < 5:
        # Warps 1-4: Consumers (compute MMA)
        consumer_warp()
    else:
        # Warps 5-7: Additional producers or helpers
        helper_warp()
```

**Benefits:**
- Better pipeline utilization
- Reduced synchronization overhead
- More efficient resource usage

---

## Hopper Async Pipelines

### Pipeline API

**Hopper has enhanced pipeline abstractions:**
```
@cute.kernel
def hopper_pipelined_gemm():
    # Create pipeline
    pipeline = cute.make_pipeline(
        num_stages=3,
        pipeline_type=cute.PipelineType.TMA
    )
    
    # Producer state (for loading)
    producer_state = pipeline.make_producer_state()
    
    # Consumer state (for computing)
    consumer_state = pipeline.make_consumer_state()
    
    # Producer loop (warp 0)
    if warp_id == 0:
        for k in range(num_k_tiles):
            # Wait for stage to be available
            pipeline.producer_acquire(producer_state)
            
            # Issue TMA load
            tma_load(gmem[k], smem[producer_state.stage])
            
            # Commit stage
            pipeline.producer_commit(producer_state)
            producer_state.advance()
    
    # Consumer loop (warps 1-4)
    else:
        for k in range(num_k_tiles):
            # Wait for data to be ready
            pipeline.consumer_wait(consumer_state)
            
            # Compute MMA
            compute_mma(smem[consumer_state.stage])
            
            # Release stage
            pipeline.consumer_release(consumer_state)
            consumer_state.advance()
```

---

## Hopper MMA Atoms

### FP16/BF16 (Same as Ampere, but faster)
```
@cute.kernel
def hopper_fp16_mma():
    # Same atom as Ampere, but 2× faster hardware
    mma = cute.make_mma_atom(cute.SM90_16x8x16_F16F16F32F32_TN)
    
    # 989 TFLOPS vs Ampere's 312 TFLOPS
```

### FP8 (New in Hopper)
```
@cute.kernel
def hopper_fp8_mma():
    # E4M3 × E5M2 → FP16
    mma = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)
    
    # 3958 TFLOPS (4× FP16 speed!)
```

### INT8 (Enhanced)
```
@cute.kernel
def hopper_int8_mma():
    # 16x8x64 INT8 (larger K than Ampere's 16x8x32)
    mma = cute.make_mma_atom(cute.SM90_16x8x64_I8I8I32I32_TN)
    
    # 3958 TOPS
```

---

## Hopper-Specific Best Practices

### Practice 1: Use TMA for All GMEM Transfers
```
# ✓ GOOD: TMA on Hopper
tma_desc = cute.make_tma_descriptor(...)
cute.copy(tma_desc, gmem, smem)

# ✗ BAD: cp.async on Hopper (slower than TMA)
async_atom = cute.make_ampere_async_copy_atom(...)
cute.copy(async_atom, gmem, smem)
```

### Practice 2: Use FP8 When Possible
```
# ✓ GOOD: FP8 for max throughput (with proper scaling)
mma = cute.make_mma_atom(cute.SM90_16x8x32_E4M3E5M2F16_SS)

# ⚠️ OK: FP16 if accuracy critical
mma = cute.make_mma_atom(cute.SM90_16x8x16_F16F16F32F32_TN)
```

### Practice 3: Use Warp Specialization
```
# ✓ GOOD: Specialized warps
if warp_id == 0:
    producer_warp()  # Load data
else:
    consumer_warp()  # Compute MMA

# ✗ BAD: All warps do everything
load_data()
compute_mma()
```

### Practice 4: Consider Clusters for Large Tiles
```
# ✓ GOOD: Use clusters for very large tiles
cluster_shape = cute.make_shape(2, 2, 1)
kernel.launch(..., cluster=cluster_shape)

# ⚠️ OK: Single CTA for smaller tiles
```

---

## Hopper Example: TMA GEMM

### Complete Hopper GEMM with TMA
```
@cute.jit
def hopper_tma_gemm(
    A: cute.Tensor,  # M×K, FP16
    B: cute.Tensor,  # K×N, FP16
    C: cute.Tensor   # M×N, FP32
):
    # ========================================
    # Configuration
    # ========================================
    tile_m, tile_n, tile_k = 128, 128, 64
    num_stages = 3
    cluster_m, cluster_n = 2, 2
    
    # ========================================
    # Create TMA Descriptors (Host-side)
    # ========================================
    tma_desc_A = cute.make_tma_descriptor(
        tensor=A,
        tile_shape=cute.make_shape(tile_m, tile_k),
        element_type=cutlass.Float16
    )
    
    tma_desc_B = cute.make_tma_descriptor(
        tensor=B,
        tile_shape=cute.make_shape(tile_k, tile_n),
        element_type=cutlass.Float16
    )
    
    tma_desc_C = cute.make_tma_descriptor(
        tensor=C,
        tile_shape=cute.make_shape(tile_m, tile_n),
        element_type=cutlass.Float32
    )
    
    # ========================================
    # Launch Kernel
    # ========================================
    num_blocks_m = (cute.size(A, 0) + tile_m - 1) // tile_m
    num_blocks_n = (cute.size(B, 1) + tile_n - 1) // tile_n
    
    kernel.launch(
        grid=[num_blocks_m, num_blocks_n, 1],
        block=[256, 1, 1],
        cluster=[cluster_m, cluster_n, 1]
    )(A, B, C, tma_desc_A, tma_desc_B, tma_desc_C)

@cute.kernel
def kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    tma_desc_A: cute.TMADescriptor,
    tma_desc_B: cute.TMADescriptor,
    tma_desc_C: cute.TMADescriptor
):
    # ========================================
    # Shared Memory (Multi-Stage)
    # ========================================
    smem_A_stages = [
        cute.make_smem_tensor(cute.make_shape(tile_m, tile_k), cutlass.Float16)
        for _ in range(num_stages)
    ]
    smem_B_stages = [
        cute.make_smem_tensor(cute.make_shape(tile_k, tile_n), cutlass.Float16)
        for _ in range(num_stages)
    ]
    
    # ========================================
    # Create MMA
    # ========================================
    mma_atom = cute.make_mma_atom(cute.SM90_16x8x16_F16F16F32F32_TN)
    tiled_mma = cute.make_tiled_mma(mma_atom, ...)
    
    tid = cute.arch.thread_idx()[0]
    warp_id = tid // 32
    
    # ========================================
    # Register Fragments
    # ========================================
    thread_mma = tiled_mma.get_slice(tid)
    
    frag_A = tiled_mma.make_fragment_A(...)
    frag_B = tiled_mma.make_fragment_B(...)
    frag_C = tiled_mma.make_fragment_C(...)
    
    frag_C.store(0.0)
    
    # ========================================
    # Get Block Coordinates
    # ========================================
    block_m = cute.arch.block_idx()[0]
    block_n = cute.arch.block_idx()[1]
    
    num_k_tiles = cute.size(A, mode=1) // tile_k
    
    # ========================================
    # Warp Specialization
    # ========================================
    if warp_id == 0:
        # PRODUCER WARP: Load with TMA
        
        # Prologue: Fill pipeline
        for stage in range(num_stages - 1):
            if stage < num_k_tiles:
                # Issue TMA load
                cute.tma_load(
                    tma_desc_A,
                    A[block_m, stage],
                    smem_A_stages[stage]
                )
                cute.tma_load(
                    tma_desc_B,
                    B[stage, block_n],
                    smem_B_stages[stage]
                )
        
        # Main loop: Continue loading
        for k in range(num_stages - 1, num_k_tiles):
            stage_idx = k % num_stages
            
            cute.tma_load(
                tma_desc_A,
                A[block_m, k],
                smem_A_stages[stage_idx]
            )
            cute.tma_load(
                tma_desc_B,
                B[k, block_n],
                smem_B_stages[stage_idx]
            )
    
    else:
        # CONSUMER WARPS: Compute MMA
        
        for k in range(num_k_tiles):
            stage_idx = k % num_stages
            
            # Wait for data
            cute.arch.tma_wait(stage_idx)
            cute.arch.syncthreads()
            
            # Load SMEM → Registers
            thread_A = thread_mma.partition_A(smem_A_stages[stage_idx])
            thread_B = thread_mma.partition_B(smem_B_stages[stage_idx])
            
            cute.copy(thread_A, frag_A)
            cute.copy(thread_B, frag_B)
            
            # MMA
            cute.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
            
            cute.arch.syncthreads()
    
    # ========================================
    # Epilogue: Store with TMA
    # ========================================
    # Store registers to SMEM
    smem_C = cute.make_smem_tensor(cute.make_shape(tile_m, tile_n), cutlass.Float32)
    
    thread_C = thread_mma.partition_C(smem_C)
    cute.copy(frag_C, thread_C)
    
    cute.arch.syncthreads()
    
    # TMA store
    if tid == 0:
        cute.tma_store(
            tma_desc_C,
            smem_C,
            C[block_m, block_n]
        )
```

---

## FP8 Scaling and Quantization

### FP8 Requires Scaling

**FP8 has limited range, need scaling factors:**
```
@cute.jit
def fp8_with_scaling(
    A_fp16: cute.Tensor,
    B_fp16: cute.Tensor
):
    # Calculate scaling factors
    scale_A = torch.max(torch.abs(A_fp16)) / 240.0  # E4M3 max ~240
    scale_B = torch.max(torch.abs(B_fp16)) / 57344.0  # E5M2 max ~57344
    
    # Quantize to FP8
    A_fp8 = (A_fp16 / scale_A).to(torch.float8_e4m3fn)
    B_fp8 = (B_fp16 / scale_B).to(torch.float8_e5m2)
    
    # Compute (scaled)
    C_fp16 = fp8_gemm(A_fp8, B_fp8)
    
    # Dequantize (scale back)
    C_fp16 = C_fp16 * scale_A * scale_B
    
    return C_fp16
```

### Per-Tensor vs Per-Channel Scaling

**Per-tensor:** Single scale for entire tensor (simpler, less accurate)
**Per-channel:** Scale per row/column (more accurate, more complex)

---

## Performance Tuning

### Tile Size Selection

**Hopper can handle larger tiles than Ampere:**
```
# Small (more blocks)
(64, 64, 64)

# Medium (balanced)
(128, 128, 64)   # Good default for Hopper

# Large (fewer blocks, better reuse)
(256, 128, 64)
(128, 256, 64)
(256, 256, 64)   # Possible with Hopper's 256KB SMEM
```

### Cluster Configuration
```
# No cluster (default)
cluster = None

# Small cluster (2×2 = 4 CTAs)
cluster = [2, 2, 1]

# Large cluster (2×4 = 8 CTAs, maximum)
cluster = [2, 4, 1]
```

**Larger clusters → better cooperation, but may reduce occupancy**

### Measuring H100 Performance
```
@cute.jit
def benchmark_hopper():
    import time
    
    M, N, K = 8192, 8192, 8192
    
    # FP16 benchmark
    A_fp16 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B_fp16 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C_fp16 = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    
    # Warm up
    for _ in range(10):
        hopper_gemm_fp16(A_fp16, B_fp16, C_fp16)
    
    torch.cuda.synchronize()
    start = time.time()
    
    num_iters = 100
    for _ in range(num_iters):
        hopper_gemm_fp16(A_fp16, B_fp16, C_fp16)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    flops = 2 * M * N * K * num_iters
    tflops = (flops / elapsed) / 1e12
    
    # H100 theoretical peak: 989 TFLOPS (FP16)
    efficiency = (tflops / 989.0) * 100
    
    print(f"FP16: {tflops:.2f} TFLOPS ({efficiency:.1f}% of peak)")
    
    # FP8 benchmark
    A_fp8 = A_fp16.to(torch.float8_e4m3fn)
    B_fp8 = B_fp16.to(torch.float8_e5m2)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iters):
        hopper_gemm_fp8(A_fp8, B_fp8, C_fp16)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tflops_fp8 = (flops / elapsed) / 1e12
    
    # H100 theoretical peak: 3958 TFLOPS (FP8)
    efficiency_fp8 = (tflops_fp8 / 3958.0) * 100
    
    print(f"FP8:  {tflops_fp8:.2f} TFLOPS ({efficiency_fp8:.1f}% of peak)")
```

---

## Summary

**Hopper (SM90) key features:**
- 4th generation Tensor Cores with FP8 (4× faster than FP16)
- Tensor Memory Accelerator (TMA) for hardware-accelerated transfers
- Thread Block Clusters for cross-CTA cooperation
- Enhanced async pipelines and warp specialization
- Up to 256KB shared memory per SM

**Performance tips:**
- Always use TMA for global ↔ shared transfers
- Use FP8 when possible (with proper scaling)
- Leverage warp specialization (producer/consumer warps)
- Consider clusters for very large tiles
- Larger tiles than Ampere (more SMEM available)

**Typical Hopper GEMM performance:**
- FP16: 850-950 TFLOPS (86-96% of 989 peak)
- FP8: 3400-3800 TFLOPS (86-96% of 3958 peak)
- cuBLAS: ~950 TFLOPS (FP16), ~3900 TFLOPS (FP8)

**FP8 is a game-changer:** 4× speedup over FP16, enabling massive LLM training/inference acceleration.

---

## Next Steps

- [Blackwell SM100](./blackwell_sm100.md) - Latest architecture features
- [Ampere SM80](./ampere_sm80.md) - Previous generation comparison
- [Choosing Atoms](./choosing_atoms.md) - Which atoms for which GPU

---

## Further Reading

- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [FP8 Format Specification](https://arxiv.org/abs/2209.05433)
- [TMA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap)