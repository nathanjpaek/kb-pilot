---
topic: "pipelining"
difficulty: "advanced"
related_topics: ["loops", "optimization", "memory_hierarchy"]
---

# Software Pipelining in CuTe DSL

## Overview

Software pipelining is a technique to overlap memory loads with computation, hiding memory latency and improving GPU utilization. CuTe DSL provides automatic pipelining support through the `cutlass.range()` function with the `prefetch_stages` parameter.

**Key idea:** While processing data from iteration `i`, simultaneously load data for iteration `i+k` where `k` is the number of prefetch stages.

---

## The Problem: Memory Latency

### Without Pipelining

**Traditional loop execution:**
```
@cute.jit
def unpipelined_loop(bound: cutlass.Int32, gmem, buffer):
    for i in range(bound):
        # 1. Load data (wait for memory)
        cute.copy(atom, gmem[i], buffer[i % 2])
        
        # 2. Wait for load to complete
        cute.arch.syncthreads()
        
        # 3. Compute (GPU cores idle during load!)
        use(buffer[i % 2])
```

**Timeline (sequential):**
```
Iteration 0: [Load 0] → [Wait] → [Compute 0]
Iteration 1: [Load 1] → [Wait] → [Compute 1]
Iteration 2: [Load 2] → [Wait] → [Compute 2]
             ↑ GPU compute idle during loads!
```

**Problem:** GPU compute units sit idle while waiting for memory loads.

---

## The Solution: Software Pipelining

### With Pipelining

**Pipelined execution overlaps load and compute:**
```
@cute.jit
def pipelined_loop(bound: cutlass.Int32, gmem, buffer):
    # CuTe DSL handles pipelining automatically!
    for i in cutlass.range(bound, prefetch_stages=2):
        cute.copy(atom, gmem[i], buffer[i % 3])
        use(buffer[i % 3])
```

**Timeline (overlapped):**
```
Prefetch:    [Load 0] [Load 1]
Iteration 0:                   [Compute 0] + [Load 2]
Iteration 1:                                 [Compute 1] + [Load 3]
Iteration 2:                                               [Compute 2] + [Load 4]
             ↑ Compute and load overlap!
```

**Benefit:** Computation and memory loads happen simultaneously, hiding latency.

---

## How CuTe DSL Implements Pipelining

### Manual Pipelining (Traditional CUDA)

**Without CuTe DSL, you'd write this manually:**
```
# Tedious manual pipelining
@cute.jit
def manual_pipeline(bound, gmem, buffer):
    prefetch_stages = 2
    total_stages = 3
    
    # Prefetch loop (load first 2 iterations)
    for i in range(prefetch_stages):
        cute.copy(atom, gmem[i], buffer[i])
    
    # Main loop (overlap load and compute)
    for i in range(bound):
        # Load next data (if available)
        if i + prefetch_stages < bound:
            cute.copy(atom, 
                     gmem[i + prefetch_stages], 
                     buffer[(i + prefetch_stages) % total_stages])
        
        # Use current data
        use(buffer[i % total_stages])
```

**Problems:**
- Error-prone indexing
- Hard to tune (change prefetch_stages requires rewriting)
- Easy to get wrong

### Automatic Pipelining (CuTe DSL)

**CuTe DSL does it for you:**
```
@cute.jit
def automatic_pipeline(bound, gmem, buffer):
    # Just specify prefetch_stages!
    for i in cutlass.range(bound, prefetch_stages=2):
        cute.copy(atom, gmem[i], buffer[i % 3])
        use(buffer[i % 3])
```

**What CuTe DSL generates:**
1. Prefetch loop (loads first `prefetch_stages` iterations)
2. Main loop (overlaps load of iteration `i+k` with compute of iteration `i`)
3. Epilogue (processes final iterations without loading)

**Benefit:** Simple API, compiler handles complexity.

---

## Using Software Pipelining

### Basic Usage
```
@cute.jit
def basic_pipelining(data_count: cutlass.Int32, gmem_src, buffer):
    num_stages = 3
    prefetch_stages = 2
    
    # Enable pipelining with prefetch_stages parameter
    for i in cutlass.range(data_count, prefetch_stages=prefetch_stages):
        # Load data (overlapped with previous iteration's compute)
        cute.copy(copy_atom, gmem_src[i], buffer[i % num_stages])
        
        # Use data (overlapped with next iteration's load)
        result = process(buffer[i % num_stages])
        store(result)
```

**Key points:**
- `prefetch_stages`: How many iterations to prefetch ahead
- `buffer[i % num_stages]`: Circular buffer (rotate through stages)
- `num_stages = prefetch_stages + 1` typically

### Choosing Buffer Size

**Rule of thumb:**
```
num_stages = prefetch_stages + 1
```

**Why?**
- `prefetch_stages` buffers for data being loaded
- 1 buffer for data being processed
- Total: `prefetch_stages + 1`

**Example:**
- `prefetch_stages=2` → need 3 buffers
- `prefetch_stages=3` → need 4 buffers

### Choosing Number of Prefetch Stages

**Factors to consider:**

1. **Memory latency vs compute time**
   - High latency, fast compute → more prefetch stages
   - Low latency, slow compute → fewer prefetch stages

2. **Buffer size constraints**
   - More stages = more shared memory needed
   - Limited by shared memory capacity

3. **Architecture**
   - Hopper (SM90+): Better support, use 2-4 stages
   - Ampere (SM80): Can use 1-2 stages
   - Older: May not benefit much

**Typical values:**
- `prefetch_stages=1`: Basic double buffering
- `prefetch_stages=2`: Good balance for most cases
- `prefetch_stages=3-4`: For high-latency operations
- `prefetch_stages>4`: Rarely beneficial (too much overhead)

---

## Complete Example: Pipelined GEMM Tile Loop
```
@cute.kernel
def pipelined_gemm_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    smem_A: cute.Tensor,
    smem_B: cute.Tensor,
    tiled_mma: cute.TiledMma
):
    # Configuration
    num_k_tiles = cute.size(gA, mode=2)
    prefetch_stages = 3
    total_stages = 4
    
    # Initialize accumulator
    acc = tiled_mma.make_fragment_C(gC.shape)
    acc.store(0.0)
    
    # Pipelined K-dimension loop
    for k_tile in cutlass.range(num_k_tiles, prefetch_stages=prefetch_stages):
        # Compute circular buffer index
        stage_idx = k_tile % total_stages
        
        # Load tiles from global to shared memory
        # (This load overlaps with previous iteration's compute!)
        cute.copy(tma_A, gA[:, :, k_tile], smem_A[:, :, stage_idx])
        cute.copy(tma_B, gB[:, :, k_tile], smem_B[:, :, stage_idx])
        
        # Synchronize before compute
        cute.arch.syncthreads()
        
        # MMA computation
        # (This compute overlaps with next iteration's load!)
        cute.gemm(tiled_mma, acc, smem_A[:, :, stage_idx], smem_B[:, :, stage_idx], acc)
    
    # Store result
    cute.copy(acc, gC)
```

**What happens:**
1. **Prefetch loop** (iterations 0-2): Load first 3 K-tiles
2. **Main loop**: For each iteration:
   - Compute MMA with current tile
   - Load next tile (overlapped!)
3. **Epilogue**: Process last few tiles without loading

---

## Architecture-Specific Considerations

### Hopper (SM90) and Later

**Best support for pipelining:**
- TMA (Tensor Memory Accelerator) optimized for pipelining
- Larger shared memory (256KB configurable)
- Hardware support for async operations

**Recommended:**
```
for i in cutlass.range(bound, prefetch_stages=3):
    # Use TMA for loads
    cute.copy(tma_atom, gmem[i], smem[i % 4])
    use(smem[i % 4])
```

### Ampere (SM80)

**Good support:**
- Async copy instructions available
- 164KB shared memory max
- Works well with 1-2 prefetch stages

**Recommended:**
```
for i in cutlass.range(bound, prefetch_stages=2):
    cute.copy(async_atom, gmem[i], smem[i % 3])
    use(smem[i % 3])
```

### Older Architectures

**Limited benefit:**
- No async copy hardware support
- Smaller shared memory
- May not see performance gains

**May still work but less effective.**

---

## Common Patterns

### Pattern 1: GEMM K-Loop Pipeline
```
@cute.jit
def gemm_k_loop(num_k_tiles, gA, gB, smem_A, smem_B, acc):
    for k in cutlass.range(num_k_tiles, prefetch_stages=2):
        stage = k % 3
        
        # Load next tiles
        cute.copy(tma_A, gA[:, :, k], smem_A[:, :, stage])
        cute.copy(tma_B, gB[:, :, k], smem_B[:, :, stage])
        
        # Synchronize
        cute.arch.syncthreads()
        
        # Compute with current tiles
        cute.gemm(mma, acc, smem_A[:, :, stage], smem_B[:, :, stage], acc)
```

### Pattern 2: Conv2D Input Pipeline
```
@cute.jit
def conv2d_pipeline(num_tiles, input_gmem, filter_gmem, smem_input, smem_filter):
    for tile in cutlass.range(num_tiles, prefetch_stages=2):
        stage = tile % 3
        
        # Load input tile and filter
        cute.copy(tma, input_gmem[tile], smem_input[stage])
        cute.copy(tma, filter_gmem[tile], smem_filter[stage])
        
        # Synchronize
        cute.arch.syncthreads()
        
        # Convolve
        convolve(smem_input[stage], smem_filter[stage])
```

### Pattern 3: Reduction Pipeline
```
@cute.jit
def pipelined_reduction(num_chunks, gmem_data, smem_buffer):
    partial_sum = 0.0
    
    for chunk in cutlass.range(num_chunks, prefetch_stages=1):
        stage = chunk % 2
        
        # Load chunk
        cute.copy(async_atom, gmem_data[chunk], smem_buffer[stage])
        
        # Reduce chunk
        partial_sum += reduce_chunk(smem_buffer[stage])
    
    return partial_sum
```

---

## Performance Benefits

### Theoretical Speedup

**Ideal case (compute fully overlaps load):**
```
Without pipelining:
Total time = N × (load_time + compute_time)

With pipelining:
Total time ≈ prefetch_time + N × max(load_time, compute_time)

Speedup = (load_time + compute_time) / max(load_time, compute_time)
```

**Example:**
- Load time: 100 μs
- Compute time: 80 μs
- Without pipelining: 180 μs per iteration
- With pipelining: 100 μs per iteration (limited by load)
- **Speedup: 1.8×**

### Real-World Performance

**Typical speedups:**
- GEMM: 1.2-1.5× (compute-bound)
- Bandwidth-limited kernels: 1.5-2× (memory-bound)
- Irregular access patterns: 1.1-1.3× (limited overlap)

**Diminishing returns:**
- `prefetch_stages=1→2`: Significant improvement
- `prefetch_stages=2→3`: Moderate improvement
- `prefetch_stages=3→4`: Small improvement
- `prefetch_stages>4`: Negligible or negative (overhead)

---

## Debugging Pipelined Loops

### Common Issues

**Issue 1: Buffer size mismatch**
```
# ❌ WRONG: Buffer too small
for i in cutlass.range(bound, prefetch_stages=3):
    cute.copy(atom, gmem[i], buffer[i % 3])  # Need 4 stages, not 3!

# ✅ CORRECT
for i in cutlass.range(bound, prefetch_stages=3):
    cute.copy(atom, gmem[i], buffer[i % 4])  # 4 stages = prefetch + 1
```

**Issue 2: Missing synchronization**
```
# ❌ WRONG: Race condition
for i in cutlass.range(bound, prefetch_stages=2):
    cute.copy(atom, gmem[i], smem[i % 3])
    # Missing syncthreads!
    use(smem[i % 3])  # May use stale data!

# ✅ CORRECT
for i in cutlass.range(bound, prefetch_stages=2):
    cute.copy(atom, gmem[i], smem[i % 3])
    cute.arch.syncthreads()  # Ensure load completes
    use(smem[i % 3])
```

**Issue 3: Not enough shared memory**
```
# Check shared memory usage
prefetch_stages = 3
stage_size = tile_m * tile_n * element_size
total_smem = (prefetch_stages + 1) * stage_size

# Ampere max: 164KB
# Hopper max: 256KB
assert total_smem <= max_smem, "Reduce prefetch_stages or tile size"
```

### Verification
```
@cute.jit
def debug_pipeline(bound, gmem, buffer):
    cute.printf("Starting pipelined loop\n")
    
    for i in cutlass.range(bound, prefetch_stages=2):
        cute.printf("Iteration %d, stage %d\n", i, i % 3)
        cute.copy(atom, gmem[i], buffer[i % 3])
        use(buffer[i % 3])
    
    cute.printf("Pipeline complete\n")
```

---

## Limitations

### Not Supported

❌ **Available only on SM90+ (Hopper and later)** for full automatic pipelining
- Older architectures may not benefit
- Manual pipelining still possible

❌ **Requires async copy support**
- TMA on Hopper
- Async copy on Ampere
- Won't work well with synchronous loads

❌ **Limited by shared memory**
- More stages = more shared memory
- May reduce occupancy

### When NOT to Use Pipelining

**Don't use pipelining when:**

1. **Compute is very fast** (< memory latency)
   - Load time dominates
   - Overlap benefit minimal

2. **Shared memory limited**
   - Can't fit multiple stages
   - Would reduce occupancy too much

3. **Irregular memory access**
   - Hard to prefetch effectively
   - May cause cache thrashing

4. **Simple kernels**
   - Overhead not worth complexity
   - Profile first!

---

## Best Practices

### ✅ DO

**Start with 2 prefetch stages:**
```
for i in cutlass.range(bound, prefetch_stages=2):
    # Good default
```

**Measure before and after:**
```
# Profile both versions
unpipelined_time = benchmark(unpipelined_kernel)
pipelined_time = benchmark(pipelined_kernel)
speedup = unpipelined_time / pipelined_time
```

**Use circular buffers:**
```
num_stages = prefetch_stages + 1
for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
    stage = i % num_stages  # Circular indexing
```

**Synchronize appropriately:**
```
for i in cutlass.range(bound, prefetch_stages=2):
    cute.copy(atom, gmem[i], smem[i % 3])
    cute.arch.syncthreads()  # Ensure load done before use
    use(smem[i % 3])
```

### ❌ DON'T

**Don't use too many stages:**
```
# ❌ BAD: Probably too many
for i in cutlass.range(bound, prefetch_stages=10):
    pass
```

**Don't forget buffer size:**
```
# ❌ BAD: Forgot +1
prefetch_stages = 3
for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
    buffer[i % prefetch_stages]  # Should be % (prefetch_stages + 1)
```

**Don't assume always faster:**
```
# Profile! Sometimes overhead > benefit
```

---

## Summary

**Software pipelining:**
- Overlaps memory loads with computation
- Hides memory latency
- Improves GPU utilization

**CuTe DSL makes it easy:**
- Automatic: `cutlass.range(bound, prefetch_stages=N)`
- Compiler handles prefetch loop, main loop, epilogue
- Just specify how many stages to prefetch

**Key parameters:**
- `prefetch_stages`: How far to prefetch (typically 1-3)
- `num_stages = prefetch_stages + 1`: Circular buffer size
- Use `i % num_stages` for indexing

**Best for:**
- Memory-bound kernels
- Hopper (SM90+) and Ampere (SM80) GPUs
- GEMM, convolution, large data movement

**Profile to verify benefit!**

---

## Next Steps

- [Loops](./loops.md) - Understanding loop types
- [Memory Hierarchy](../03_memory_and_layouts/memory_hierarchy.md) - GMEM, SMEM, registers
- [GEMM Examples](../10_examples/advanced/persistent_gemm.md) - Real pipelining use

---

## Further Reading

- [NVIDIA Software Pipelining Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- [Hopper TMA Documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)