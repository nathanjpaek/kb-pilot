---
topic: "pipelining_patterns"
difficulty: "advanced"
related_topics: ["tiling_strategies", "synchronization", "pipelining"]
---

# Pipelining Patterns in CuTe DSL

## Overview

**Software pipelining** overlaps data movement with computation to hide memory latency. This document covers advanced pipelining patterns specifically for optimization, building on the basic pipelining concepts.

**Key concepts:**
- Producer-consumer patterns
- Multi-stage pipelines
- Warp specialization
- Async copy pipelining
- TMA pipelining (Hopper)
- Performance tuning

---

## Basic Pipeline Review

### Simple Pipeline (Recap)

**Load next while computing current:**
```python
@cute.kernel
def basic_pipeline(gmem: cute.Tensor, num_tiles: cutlass.Int32):
    """Basic pipeline: load k+1 while computing k"""
    
    smem = cute.make_smem_tensor(cute.make_shape(TILE_SIZE), dtype)
    
    # Prologue: Load tile 0
    if num_tiles > 0:
        load_tile(gmem, smem, tile=0)
    
    # Main loop
    for k in range(num_tiles):
        # Issue load for next tile (async)
        if k + 1 < num_tiles:
            load_tile_async(gmem, smem, tile=k+1)
        
        # Wait for current tile
        wait_tile(tile=k)
        
        # Compute with current tile (while next loads!)
        compute_tile(smem)
    
    # Performance: overlap load and compute
```

---

## Double Buffering Pattern

### Ping-Pong Buffers

**Alternate between two buffers:**
```python
@cute.kernel
def double_buffer_pipeline(gmem: cute.Tensor, num_tiles: cutlass.Int32):
    """Double buffering: load and compute simultaneously"""
    
    # Allocate 2 buffers
    smem_buffers = [
        cute.make_smem_tensor(cute.make_shape(TILE_M, TILE_K), dtype),
        cute.make_smem_tensor(cute.make_shape(TILE_M, TILE_K), dtype)
    ]
    
    # Async copy atom
    async_copy = cute.make_ampere_async_copy_atom(cutlass.Float16)
    
    # Prologue: Load into buffer 0
    write_idx = 0
    cute.copy(async_copy, gmem[0], smem_buffers[write_idx])
    cute.arch.cp_async_commit_group()
    
    # Main loop
    for k in range(num_tiles):
        read_idx = write_idx
        write_idx = 1 - write_idx  # Toggle: 0↔1
        
        # Issue load for next tile (into write buffer)
        if k + 1 < num_tiles:
            cute.copy(async_copy, gmem[k+1], smem_buffers[write_idx])
            cute.arch.cp_async_commit_group()
        
        # Wait for current tile (in read buffer)
        cute.arch.cp_async_wait_group(0)
        cute.arch.syncthreads()
        
        # Compute with read buffer (write buffer loading in background!)
        compute_tile(smem_buffers[read_idx])
        
        # No sync needed before next iteration - different buffers!
```

**Key insight:** No sync between load and compute - they use different buffers!

---

## Multi-Stage Pipeline

### Triple Buffering

**Even deeper pipeline with 3+ buffers:**
```python
@cute.kernel
def triple_buffer_pipeline(
    gmem_A: cute.Tensor,
    gmem_B: cute.Tensor,
    result: cute.Tensor,
    num_k_tiles: cutlass.Int32
):
    """Triple buffering: maximum overlap"""
    
    NUM_STAGES = 3
    
    # Allocate 3 buffers for A and B
    smem_A = [
        cute.make_smem_tensor(cute.make_shape(TILE_M, TILE_K), cutlass.Float16)
        for _ in range(NUM_STAGES)
    ]
    smem_B = [
        cute.make_smem_tensor(cute.make_shape(TILE_K, TILE_N), cutlass.Float16)
        for _ in range(NUM_STAGES)
    ]
    
    async_copy = cute.make_ampere_async_copy_atom(cutlass.Float16)
    
    # Accumulator
    acc = cute.make_rmem_tensor(cute.make_shape(TILE_M, TILE_N), cutlass.Float32)
    acc.store(0.0)
    
    # ========================================
    # Prologue: Fill pipeline
    # ========================================
    for stage in range(NUM_STAGES - 1):
        if stage < num_k_tiles:
            cute.copy(async_copy, gmem_A[stage], smem_A[stage])
            cute.copy(async_copy, gmem_B[stage], smem_B[stage])
            cute.arch.cp_async_commit_group()
    
    # ========================================
    # Main loop: Process tiles
    # ========================================
    for k in range(num_k_tiles):
        stage_idx = k % NUM_STAGES
        
        # Issue load for tile k + (NUM_STAGES - 1)
        future_tile = k + NUM_STAGES - 1
        if future_tile < num_k_tiles:
            cute.copy(async_copy, gmem_A[future_tile], smem_A[stage_idx])
            cute.copy(async_copy, gmem_B[future_tile], smem_B[stage_idx])
            cute.arch.cp_async_commit_group()
        
        # Wait for tile k
        # (NUM_STAGES - 2) groups can be in flight
        cute.arch.cp_async_wait_group(NUM_STAGES - 2)
        cute.arch.syncthreads()
        
        # Compute with tile k (while k+1, k+2 loading!)
        mma_tile(acc, smem_A[stage_idx], smem_B[stage_idx])
    
    # Store result
    store_accumulator(acc, result)
```

**Pipeline state:**
```
Iteration k:
  Computing:  tile k
  Loading:    tile k+1, k+2
  
3 tiles in flight simultaneously!
```

---

## Warp-Specialized Pipeline

### Producer-Consumer Pattern

**Dedicate warps to loading (producers) and computing (consumers):**
```python
@cute.kernel
def warp_specialized_pipeline(
    gmem_A: cute.Tensor,
    gmem_B: cute.Tensor,
    result: cute.Tensor,
    num_k_tiles: cutlass.Int32
):
    """Warp specialization: producer/consumer separation"""
    
    tid = cute.arch.thread_idx()[0]
    warp_id = tid // 32
    lane_id = tid % 32
    
    NUM_STAGES = 3
    NUM_PRODUCER_WARPS = 1  # 1 warp loads
    NUM_CONSUMER_WARPS = 3  # 3 warps compute
    
    # Shared memory stages
    smem_A = [
        cute.make_smem_tensor(cute.make_shape(TILE_M, TILE_K), cutlass.Float16)
        for _ in range(NUM_STAGES)
    ]
    smem_B = [
        cute.make_smem_tensor(cute.make_shape(TILE_K, TILE_N), cutlass.Float16)
        for _ in range(NUM_STAGES)
    ]
    
    # ========================================
    # Producer warp (warp 0)
    # ========================================
    if warp_id < NUM_PRODUCER_WARPS:
        # Producer: Load all tiles
        
        for k in range(num_k_tiles):
            stage_idx = k % NUM_STAGES
            
            # Load tile k
            if lane_id == 0:  # Single thread issues TMA
                cute.tma_load(tma_desc_A, gmem_A[k], smem_A[stage_idx])
                cute.tma_load(tma_desc_B, gmem_B[k], smem_B[stage_idx])
            
            # Mark stage ready
            signal_stage_ready(stage_idx)
    
    # ========================================
    # Consumer warps (warps 1-3)
    # ========================================
    else:
        # Consumer: Compute all tiles
        
        consumer_id = warp_id - NUM_PRODUCER_WARPS
        
        # Accumulator
        acc = cute.make_rmem_tensor(
            cute.make_shape(WARP_M, WARP_N),
            cutlass.Float32
        )
        acc.store(0.0)
        
        for k in range(num_k_tiles):
            stage_idx = k % NUM_STAGES
            
            # Wait for producer to load
            wait_stage_ready(stage_idx)
            cute.arch.syncthreads()
            
            # Compute with this tile
            warp_mma(acc, smem_A[stage_idx], smem_B[stage_idx], consumer_id)
            
            cute.arch.syncthreads()
        
        # Store result (only consumer warps)
        store_warp_accumulator(acc, result, consumer_id)
```

**Benefits:**
- Producer warp never blocks on compute
- Consumer warps never block on load
- Better instruction-level parallelism
- Higher throughput

---

## Async Copy Patterns (Ampere)

### cp.async Pipeline

**Hardware-accelerated async copy:**
```python
@cute.kernel
def cp_async_pipeline(
    gmem: cute.Tensor,
    result: cute.Tensor,
    num_tiles: cutlass.Int32
):
    """Ampere cp.async with pipeline"""
    
    NUM_STAGES = 4
    
    smem = [
        cute.make_smem_tensor(cute.make_shape(TILE_SIZE), cutlass.Float16)
        for _ in range(NUM_STAGES)
    ]
    
    # Async copy atom
    async_copy = cute.make_ampere_async_copy_atom(cutlass.Float16)
    
    # Prologue: Issue first NUM_STAGES-1 loads
    for stage in range(NUM_STAGES - 1):
        if stage < num_tiles:
            cute.copy(async_copy, gmem[stage], smem[stage])
            cute.arch.cp_async_commit_group()  # Mark group boundary
    
    # Main loop
    for k in range(num_tiles):
        stage_idx = k % NUM_STAGES
        
        # Issue load for tile k + (NUM_STAGES - 1)
        future_tile = k + NUM_STAGES - 1
        if future_tile < num_tiles:
            cute.copy(async_copy, gmem[future_tile], smem[stage_idx])
            cute.arch.cp_async_commit_group()
        
        # Wait for tile k
        # Keep NUM_STAGES - 2 groups in flight
        cute.arch.cp_async_wait_group(NUM_STAGES - 2)
        cute.arch.syncthreads()
        
        # Compute
        compute_tile(smem[stage_idx])
    
    # Write result
    store_result(result)
```

**Key functions:**
- `cp_async_commit_group()`: Mark end of group
- `cp_async_wait_group(N)`: Wait for all but N newest groups
- Allows up to 8 groups in flight

---

## TMA Pipeline (Hopper)

### Hopper TMA with Barriers

**Tensor Memory Accelerator pipelining:**
```python
@cute.kernel
def tma_pipeline_hopper(
    gmem_A: cute.Tensor,
    gmem_B: cute.Tensor,
    result: cute.Tensor,
    tma_desc_A,
    tma_desc_B,
    num_k_tiles: cutlass.Int32
):
    """Hopper TMA with transaction-based pipelining"""
    
    tid = cute.arch.thread_idx()[0]
    warp_id = tid // 32
    
    NUM_STAGES = 3
    
    # Shared memory stages
    smem_A = [
        cute.make_smem_tensor(cute.make_shape(TILE_M, TILE_K), cutlass.Float16)
        for _ in range(NUM_STAGES)
    ]
    smem_B = [
        cute.make_smem_tensor(cute.make_shape(TILE_K, TILE_N), cutlass.Float16)
        for _ in range(NUM_STAGES)
    ]
    
    # Pipeline with transaction barriers
    pipeline = cute.make_pipeline(
        num_stages=NUM_STAGES,
        pipeline_type=cute.PipelineType.TMA
    )
    
    # ========================================
    # Producer warp (warp 0)
    # ========================================
    if warp_id == 0:
        producer_state = pipeline.make_producer_state()
        
        for k in range(num_k_tiles):
            # Acquire stage
            pipeline.producer_acquire(producer_state)
            
            stage_idx = producer_state.index()
            
            # Issue TMA (single thread)
            if tid == 0:
                cute.tma_load(tma_desc_A, gmem_A[k], smem_A[stage_idx])
                cute.tma_load(tma_desc_B, gmem_B[k], smem_B[stage_idx])
            
            # Commit stage
            pipeline.producer_commit(producer_state)
            
            # Advance to next stage
            producer_state.advance()
    
    # ========================================
    # Consumer warps (warps 1-3)
    # ========================================
    else:
        consumer_state = pipeline.make_consumer_state()
        
        acc = cute.make_rmem_tensor(
            cute.make_shape(WARP_M, WARP_N),
            cutlass.Float32
        )
        acc.store(0.0)
        
        for k in range(num_k_tiles):
            # Wait for stage to be ready
            pipeline.consumer_wait(consumer_state)
            
            stage_idx = consumer_state.index()
            
            # Compute
            warp_mma(acc, smem_A[stage_idx], smem_B[stage_idx], warp_id - 1)
            
            # Release stage
            pipeline.consumer_release(consumer_state)
            
            # Advance to next stage
            consumer_state.advance()
        
        # Store result
        store_warp_result(acc, result, warp_id - 1)
```

**Benefits:**
- Hardware-managed synchronization
- Transaction-based ordering
- More efficient than manual barriers

---

## Multi-Level Pipelining

### GEMM with Load and MMA Overlap

**Overlap memory, MMA, and prefetch:**
```python
@cute.kernel
def multi_level_pipeline_gemm(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor
):
    """
    Multi-level pipeline:
    - Level 1: Prefetch K tiles (memory)
    - Level 2: MMA computation (compute)
    - Level 3: Register prefetch (registers)
    """
    
    NUM_K_STAGES = 3  # K-dimension pipeline depth
    
    # Shared memory stages
    smem_A = [
        cute.make_smem_tensor(cute.make_shape(TILE_M, TILE_K), dtype)
        for _ in range(NUM_K_STAGES)
    ]
    smem_B = [
        cute.make_smem_tensor(cute.make_shape(TILE_K, TILE_N), dtype)
        for _ in range(NUM_K_STAGES)
    ]
    
    # Register fragments for double buffering
    frag_A = [
        cute.make_rmem_tensor(cute.make_shape(MMA_M, MMA_K), dtype),
        cute.make_rmem_tensor(cute.make_shape(MMA_M, MMA_K), dtype)
    ]
    frag_B = [
        cute.make_rmem_tensor(cute.make_shape(MMA_K, MMA_N), dtype),
        cute.make_rmem_tensor(cute.make_shape(MMA_K, MMA_N), dtype)
    ]
    
    acc = cute.make_rmem_tensor(cute.make_shape(TILE_M, TILE_N), cutlass.Float32)
    acc.store(0.0)
    
    # Prologue: Fill K-tile pipeline
    for stage in range(NUM_K_STAGES - 1):
        if stage < num_k_tiles:
            load_k_tile_async(smem_A[stage], smem_B[stage], stage)
    
    # Main K-tile loop
    for k_tile in range(num_k_tiles):
        k_stage = k_tile % NUM_K_STAGES
        
        # Load next K-tile (Level 1: memory pipeline)
        if k_tile + NUM_K_STAGES - 1 < num_k_tiles:
            load_k_tile_async(
                smem_A[k_stage],
                smem_B[k_stage],
                k_tile + NUM_K_STAGES - 1
            )
        
        # Wait for current K-tile
        wait_k_tile(k_tile)
        cute.arch.syncthreads()
        
        # Prefetch first MMA fragments (Level 3: register pipeline)
        frag_idx = 0
        load_mma_fragment(frag_A[frag_idx], smem_A[k_stage], mma_step=0)
        load_mma_fragment(frag_B[frag_idx], smem_B[k_stage], mma_step=0)
        
        # Inner loop: MMA operations within K-tile
        for mma_step in range(TILE_K // MMA_K):
            # Prefetch next MMA fragments (while computing current)
            next_frag_idx = 1 - frag_idx
            if mma_step + 1 < TILE_K // MMA_K:
                load_mma_fragment(frag_A[next_frag_idx], smem_A[k_stage], mma_step + 1)
                load_mma_fragment(frag_B[next_frag_idx], smem_B[k_stage], mma_step + 1)
            
            # MMA with current fragments (Level 2: compute pipeline)
            cute.gemm(mma_atom, acc, frag_A[frag_idx], frag_B[frag_idx], acc)
            
            # Swap fragment buffers
            frag_idx = next_frag_idx
        
        cute.arch.syncthreads()
    
    # Store result
    store_tile(C, acc)
```

**Three overlapped operations:**
1. Loading K-tile k+2 (memory)
2. Computing K-tile k (MMA)
3. Prefetching fragments for k (registers)

---

## Pipeline Tuning

### Choosing Number of Stages

**Trade-offs:**
```python
def choose_pipeline_depth(
    latency_cycles: int,
    compute_cycles: int,
    smem_available: int,
    smem_per_stage: int
) -> int:
    """Choose optimal pipeline depth"""
    
    # Need enough stages to hide latency
    min_stages = (latency_cycles + compute_cycles - 1) // compute_cycles
    
    # Limited by shared memory
    max_stages = smem_available // smem_per_stage
    
    # Practical limits
    min_stages = max(min_stages, 2)  # At least 2 for basic pipeline
    max_stages = min(max_stages, 8)  # Diminishing returns beyond 8
    
    optimal = min(min_stages + 1, max_stages)  # +1 for margin
    
    print(f"Latency requires: {min_stages} stages")
    print(f"Memory allows: {max_stages} stages")
    print(f"Recommended: {optimal} stages")
    
    return optimal

# Example: Ampere A100
choose_pipeline_depth(
    latency_cycles=400,    # Global memory latency
    compute_cycles=100,    # Compute per tile
    smem_available=164*1024,  # 164 KB
    smem_per_stage=32*1024    # 32 KB per stage
)
# Output:
# Latency requires: 4 stages
# Memory allows: 5 stages
# Recommended: 5 stages
```

### Measuring Pipeline Efficiency
```python
@cute.jit
def measure_pipeline_efficiency():
    """Measure pipeline performance"""
    
    import time
    
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    
    # Test different pipeline depths
    for num_stages in [1, 2, 3, 4, 5]:
        # Warm up
        for _ in range(10):
            gemm_kernel.launch(...)(A, B, C, num_stages=num_stages)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            gemm_kernel.launch(...)(A, B, C, num_stages=num_stages)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate performance
        flops = 2 * M * N * K * 100
        tflops = flops / elapsed / 1e12
        
        print(f"Stages {num_stages}: {tflops:6.2f} TFLOPS")
    
    # Typical output (A100):
    # Stages 1:  89.23 TFLOPS
    # Stages 2: 156.78 TFLOPS
    # Stages 3: 198.45 TFLOPS
    # Stages 4: 212.34 TFLOPS  ← Best
    # Stages 5: 210.12 TFLOPS  (diminishing returns)
```

---

## Common Pitfalls

### Pitfall 1: Over-Pipelining

**Too many stages hurts:**
```python
# ✗ BAD: 10 stages
NUM_STAGES = 10
# Uses too much shared memory
# Reduces occupancy
# Diminishing returns

# ✓ GOOD: 3-4 stages
NUM_STAGES = 4
# Good balance
```

### Pitfall 2: Incorrect Wait Points
```python
# ✗ BAD: Wait too early
cute.arch.cp_async_wait_group(NUM_STAGES - 1)  # Too conservative
# Not enough overlap

# ✗ BAD: Wait too late
cute.arch.cp_async_wait_group(NUM_STAGES + 1)  # Wrong!
# May compute before data ready

# ✓ GOOD: Wait at right time
cute.arch.cp_async_wait_group(NUM_STAGES - 2)
```

### Pitfall 3: Missing Synchronization
```python
# ✗ BAD: No sync before overwrite
compute_tile(smem[stage])
# Immediately start loading into same buffer!
load_tile_async(smem[stage], k+1)  # Race condition!

# ✓ GOOD: Sync before reuse
compute_tile(smem[stage])
cute.arch.syncthreads()  # Ensure all threads done
load_tile_async(smem[stage], k+1)
```

---

## Best Practices

### ✅ DO

**Use 3-4 stages for most cases:**
```python
NUM_STAGES = 3  # or 4 for Hopper
```

**Profile to find optimal depth:**
```python
for stages in [2, 3, 4, 5]:
    benchmark(stages)
```

**Use warp specialization on Hopper:**
```python
if warp_id == 0:
    producer()
else:
    consumer()
```

**Wait at correct point:**
```python
cute.arch.cp_async_wait_group(NUM_STAGES - 2)
```

### ❌ DON'T

**Don't pipeline everything:**
```python
# Pipeline only memory-bound kernels
# Compute-bound kernels don't benefit
```

**Don't exceed shared memory:**
```python
# NUM_STAGES * stage_size < max_smem
```

**Don't forget bounds checks:**
```python
if k + NUM_STAGES < num_tiles:
    load_tile(k + NUM_STAGES)
```

---

## Summary

**Pipelining patterns:**
- Double buffering: 2 stages
- Multi-stage: 3-5 stages
- Warp specialization: Producer/consumer
- Multi-level: Memory + compute + register

**Key techniques:**
- Ampere: cp.async with groups
- Hopper: TMA with barriers
- Always measure performance
- 3-4 stages usually optimal

**Benefits:**
- 1.5-3× speedup typical
- Higher for memory-bound kernels
- Essential for peak performance

**Key insight:** Pipelining hides latency by keeping GPU busy. The deeper the pipeline, the better the overlap, but diminishing returns and resource limits apply.

---

## Next Steps

- [Tiling Strategies](./tiling_strategies.md) - What to pipeline
- [Thread Layouts](./thread_layouts.md) - Organize for pipelining
- [Performance Tips](./performance_tips.md) - Additional optimizations

---

## Further Reading

- [Software Pipelining](https://en.wikipedia.org/wiki/Software_pipelining)
- [CUTLASS 3.x Pipelining](https://github.com/NVIDIA/cutlass/blob/main/media/docs/gemm_api_3x.md#pipelining)
- [Hopper TMA Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator-tma)