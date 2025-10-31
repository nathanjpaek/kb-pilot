---
topic: "performance_tips"
difficulty: "intermediate"
related_topics: ["tiling_strategies", "thread_layouts", "profiling"]
---

# Performance Tips for CuTe DSL

## Overview

This guide covers practical performance optimization techniques for CuTe DSL kernels. These are battle-tested tips for achieving the best performance on modern GPUs.

**Key concepts:**
- Occupancy optimization
- Memory bandwidth optimization
- Compute throughput maximization
- Register pressure management
- Instruction-level parallelism
- Common performance pitfalls

---

## Understanding Performance Bottlenecks

### The Three Limits

**Every kernel is limited by one of three factors:**

1. **Memory Bandwidth**
   - Limited by DRAM throughput
   - ~1-2 TB/s on modern GPUs
   - Example: Element-wise operations

2. **Compute Throughput**
   - Limited by FLOPS
   - ~100-300 TFLOPS on modern GPUs
   - Example: Matrix multiplication

3. **Latency**
   - Limited by instruction latency
   - Example: Reductions, small problems

### Identifying Your Bottleneck
```python
def identify_bottleneck(kernel_name: str):
    """Simple heuristic to identify bottleneck"""
    
    # Measure achieved performance
    achieved_flops = measure_flops(kernel_name)
    achieved_bandwidth = measure_bandwidth(kernel_name)
    
    # Compare to peak
    peak_flops = 312.0  # TFLOPS (A100)
    peak_bandwidth = 1555.0  # GB/s (A100)
    
    compute_utilization = achieved_flops / peak_flops
    memory_utilization = achieved_bandwidth / peak_bandwidth
    
    print(f"Compute: {compute_utilization*100:.1f}% of peak")
    print(f"Memory:  {memory_utilization*100:.1f}% of peak")
    
    if compute_utilization > 0.8:
        print("✓ Compute bound - good!")
    elif memory_utilization > 0.8:
        print("✓ Memory bound - expected for this kernel")
    else:
        print("⚠ Low utilization - check for:")
        print("  - Synchronization overhead")
        print("  - Small problem size")
        print("  - Non-coalesced memory access")
        print("  - Bank conflicts")
```

---

## Occupancy Optimization

### What is Occupancy?

**Occupancy = Active warps / Max possible warps per SM**
```
High occupancy (80-100%):
  - More warps to hide latency
  - Better instruction-level parallelism
  - But: May reduce per-thread resources

Low occupancy (25-50%):
  - Fewer warps
  - More registers/shared memory per thread
  - Can be optimal for compute-bound kernels
```

### Measuring Occupancy
```python
def calculate_occupancy(
    threads_per_block: int,
    registers_per_thread: int,
    shared_mem_per_block: int,
    arch: str = "sm_80"  # Ampere
) -> float:
    """Calculate theoretical occupancy"""
    
    if arch == "sm_80":  # Ampere
        max_threads_per_sm = 2048
        max_blocks_per_sm = 32
        max_registers_per_sm = 65536
        max_shared_mem_per_sm = 164 * 1024
        max_registers_per_thread = 255
    elif arch == "sm_90":  # Hopper
        max_threads_per_sm = 2048
        max_blocks_per_sm = 32
        max_registers_per_sm = 65536
        max_shared_mem_per_sm = 228 * 1024
        max_registers_per_thread = 255
    
    # Check if configuration is valid
    if threads_per_block > 1024:
        return 0.0
    if registers_per_thread > max_registers_per_thread:
        return 0.0
    
    # Blocks limited by threads
    blocks_by_threads = max_threads_per_sm // threads_per_block
    blocks_by_threads = min(blocks_by_threads, max_blocks_per_sm)
    
    # Blocks limited by registers
    regs_per_block = threads_per_block * registers_per_thread
    if regs_per_block > 0:
        blocks_by_regs = max_registers_per_sm // regs_per_block
    else:
        blocks_by_regs = max_blocks_per_sm
    
    # Blocks limited by shared memory
    if shared_mem_per_block > 0:
        blocks_by_smem = max_shared_mem_per_sm // shared_mem_per_block
    else:
        blocks_by_smem = max_blocks_per_sm
    
    # Actual blocks per SM
    blocks_per_sm = min(
        blocks_by_threads,
        blocks_by_regs,
        blocks_by_smem,
        max_blocks_per_sm
    )
    
    # Occupancy
    active_warps = blocks_per_sm * (threads_per_block // 32)
    max_warps = max_threads_per_sm // 32
    occupancy = active_warps / max_warps
    
    print(f"Blocks per SM: {blocks_per_sm}")
    print(f"  Limited by threads: {blocks_by_threads}")
    print(f"  Limited by registers: {blocks_by_regs}")
    print(f"  Limited by shared memory: {blocks_by_smem}")
    print(f"Occupancy: {occupancy*100:.1f}%")
    
    return occupancy

# Example
calculate_occupancy(
    threads_per_block=256,
    registers_per_thread=32,
    shared_mem_per_block=48*1024
)
```

### Optimizing Occupancy

**For memory-bound kernels (maximize occupancy):**
```python
# ✓ GOOD: High occupancy
THREADS_PER_BLOCK = 256  # 8 blocks per SM
REGISTERS_PER_THREAD = 32  # Moderate
SHARED_MEM = 16*1024  # Small

# Occupancy: ~100%
```

**For compute-bound kernels (balance resources):**
```python
# ✓ GOOD: Lower occupancy, more resources
THREADS_PER_BLOCK = 128  # Fewer threads
REGISTERS_PER_THREAD = 128  # More registers
SHARED_MEM = 96*1024  # Larger shared memory

# Occupancy: ~50% (but higher throughput!)
```

---

## Memory Bandwidth Optimization

### Tip 1: Coalesce Memory Access

**Always access consecutive memory:**
```python
# ✗ BAD: Strided access (non-coalesced)
@cute.kernel
def strided_access(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    stride = 32
    idx = tid
    while idx < cute.size(data):
        data[idx] = data[idx] * 2.0
        idx += stride
    # Bandwidth: ~50 GB/s (3% of peak!)

# ✓ GOOD: Coalesced access
@cute.kernel
def coalesced_access(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    idx = bid * 256 + tid
    if idx < cute.size(data):
        data[idx] = data[idx] * 2.0
    # Bandwidth: ~1400 GB/s (90% of peak!)
```

### Tip 2: Vectorize Memory Access

**Load multiple elements per instruction:**
```python
# ✗ BAD: Scalar loads
@cute.kernel
def scalar_loads(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    base = tid * 4
    
    a = data[base + 0]
    b = data[base + 1]
    c = data[base + 2]
    d = data[base + 3]
    # 4 separate load instructions

# ✓ GOOD: Vectorized load
@cute.kernel
def vectorized_loads(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Single instruction loads 4 floats
    vec = cute.make_rmem_tensor(cute.make_shape(4), cutlass.Float32)
    cute.copy(data[tid*4:(tid+1)*4], vec)
    
    # Process
    for i in range(4):
        vec[i] = vec[i] * 2.0
    
    # Single instruction stores 4 floats
    cute.copy(vec, data[tid*4:(tid+1)*4])
    # 2 instructions instead of 8!
```

### Tip 3: Use Shared Memory for Reuse

**Cache in shared memory:**
```python
@cute.kernel
def with_reuse(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    """Cache data with reuse in shared memory"""
    
    # Shared memory cache
    smem_A = cute.make_smem_tensor(cute.make_shape(TILE_M, TILE_K), dtype)
    smem_B = cute.make_smem_tensor(cute.make_shape(TILE_K, TILE_N), dtype)
    
    # Load once from global memory
    load_tile(smem_A, A)
    load_tile(smem_B, B)
    cute.arch.syncthreads()
    
    # Reuse many times from shared memory (fast!)
    for i in range(TILE_M):
        for j in range(TILE_N):
            acc = 0.0
            for k in range(TILE_K):
                acc += smem_A[i, k] * smem_B[k, j]  # Reused!
            C[i, j] = acc
    
    # Arithmetic intensity: TILE_M * TILE_N * TILE_K / (TILE_M*TILE_K + TILE_K*TILE_N)
    # Example: 128*128*32 / (128*32 + 32*128) = 64 FLOPs/byte (excellent!)
```

### Tip 4: Minimize Redundant Loads
```python
# ✗ BAD: Redundant loads
@cute.kernel
def redundant_loads(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    result = data[tid] * 2.0  # Load 1
    result = result + data[tid] * 3.0  # Load 2 (same element!)
    data[tid] = result

# ✓ GOOD: Load once, reuse
@cute.kernel
def single_load(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    val = data[tid]  # Load once
    result = val * 2.0 + val * 3.0  # Reuse from register
    data[tid] = result
```

---

## Compute Throughput Optimization

### Tip 5: Use Tensor Cores

**Leverage specialized hardware:**
```python
# ✗ BAD: Manual loops (slow)
@cute.kernel
def manual_gemm(A, B, C):
    # Scalar operations
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    # ~1 TFLOPS

# ✓ GOOD: Tensor cores (fast)
@cute.kernel
def tensor_core_gemm(A, B, C):
    # Use MMA instructions
    mma = cute.make_mma_atom(cute.SM80_16x8x16_F16F16F32F32_TN)
    
    frag_A = load_fragment_A()
    frag_B = load_fragment_B()
    frag_C = cute.make_rmem_tensor(mma.frag_C_shape(), cutlass.Float32)
    
    cute.gemm(mma, frag_C, frag_A, frag_B, frag_C)
    # ~200 TFLOPS!
```

### Tip 6: Maximize Instruction-Level Parallelism

**Interleave independent operations:**
```python
# ✗ BAD: Sequential dependencies
@cute.kernel
def sequential():
    a = load_a()
    b = compute_b(a)  # Depends on a
    c = compute_c(b)  # Depends on b
    d = compute_d(c)  # Depends on c
    # Long dependency chain

# ✓ GOOD: Independent operations
@cute.kernel
def parallel():
    # Load all inputs first
    a0 = load_a(0)
    a1 = load_a(1)
    a2 = load_a(2)
    a3 = load_a(3)
    
    # Compute independently (parallel!)
    b0 = compute(a0)
    b1 = compute(a1)
    b2 = compute(a2)
    b3 = compute(a3)
    
    # Store
    store(b0, b1, b2, b3)
```

### Tip 7: Unroll Loops

**Let compiler generate more instructions:**
```python
# ✗ BAD: Small dynamic loop
@cute.kernel
def dynamic_loop():
    for i in range(n):  # n is runtime value
        process(i)
    # Loop overhead

# ✓ GOOD: Unrolled loop
@cute.kernel
def unrolled_loop():
    # Compile-time loop (unrolls automatically)
    for i in cutlass.range_constexpr(8):
        process(i)
    # No loop overhead, more ILP
```

---

## Register Pressure Management

### Tip 8: Balance Register Usage

**Too many registers → low occupancy:**
```python
def analyze_register_usage():
    """Check register usage"""
    
    # High register usage
    # 128 registers/thread × 256 threads = 32,768 registers
    # Only 2 blocks fit per SM (65,536 total registers)
    # Occupancy: 50%
    
    # Lower register usage
    # 64 registers/thread × 256 threads = 16,384 registers
    # 4 blocks fit per SM
    # Occupancy: 100%
```

**Optimize register usage:**
```python
# ✗ BAD: Large arrays in registers
@cute.kernel
def high_register_pressure():
    # 256 registers! (assuming float32)
    temp = cute.make_rmem_tensor(cute.make_shape(64), cutlass.Float32)
    # Low occupancy

# ✓ GOOD: Use shared memory for large arrays
@cute.kernel
def low_register_pressure():
    # Shared memory (not registers)
    temp = cute.make_smem_tensor(cute.make_shape(64), cutlass.Float32)
    # Higher occupancy
```

### Tip 9: Spill Prevention

**Avoid register spilling to local memory:**
```python
# ✗ BAD: Too many live values
@cute.kernel
def register_spill():
    a = load_a()
    b = load_b()
    c = load_c()
    d = load_d()
    e = load_e()
    f = load_f()
    # ... (100+ live variables)
    # Compiler spills to local memory (slow!)

# ✓ GOOD: Limit live values
@cute.kernel
def no_spill():
    # Process in batches
    a = load_a()
    b = load_b()
    result1 = compute(a, b)
    store(result1)
    
    c = load_c()
    d = load_d()
    result2 = compute(c, d)
    store(result2)
    # Fewer live values at once
```

---

## Shared Memory Optimization

### Tip 10: Avoid Bank Conflicts

**Pad to eliminate conflicts:**
```python
# ✗ BAD: Bank conflicts
smem = cute.make_smem_tensor(
    cute.make_shape(32, 32),
    cutlass.Float32
)
# Stride = 32 floats = 128 bytes
# Wraps to same banks → conflicts!

# ✓ GOOD: Padded to avoid conflicts
smem = cute.make_smem_tensor(
    cute.make_shape(32, 32 + 8),  # +8 padding
    cutlass.Float32
)
# Stride = 40 floats = 160 bytes
# No conflicts!
```

### Tip 11: Optimize Shared Memory Layout

**Choose layout for access pattern:**
```python
# Access pattern: row-wise
# Use row-major layout
smem_row_major = cute.make_smem_tensor(
    cute.make_shape(M, N),
    cutlass.Float32,
    cute.make_stride(N, 1)  # Row-major
)

# Access pattern: column-wise
# Use column-major layout
smem_col_major = cute.make_smem_tensor(
    cute.make_shape(M, N),
    cutlass.Float32,
    cute.make_stride(1, M)  # Column-major
)
```

### Tip 12: Minimize Shared Memory Usage

**Higher occupancy with less shared memory:**
```python
def optimize_smem_usage():
    """Find minimum shared memory needed"""
    
    # Option 1: Large tiles, low occupancy
    TILE_SIZE = 256
    smem_usage = TILE_SIZE * TILE_SIZE * 2  # 128 KB
    blocks_per_sm = 164*1024 // smem_usage  # 1 block
    occupancy = "50%"
    
    # Option 2: Smaller tiles, higher occupancy
    TILE_SIZE = 128
    smem_usage = TILE_SIZE * TILE_SIZE * 2  # 32 KB
    blocks_per_sm = 164*1024 // smem_usage  # 5 blocks
    occupancy = "100%"
    
    # Profile both and choose best!
```

---

## Synchronization Optimization

### Tip 13: Minimize Synchronization
```python
# ✗ BAD: Too much synchronization
@cute.kernel
def excessive_sync():
    for i in range(1000):
        load_data()
        cute.arch.syncthreads()  # Sync 1
        process_data()
        cute.arch.syncthreads()  # Sync 2
        store_data()
        cute.arch.syncthreads()  # Sync 3
    # 3000 synchronizations!

# ✓ GOOD: Minimal synchronization
@cute.kernel
def minimal_sync():
    for i in range(1000):
        load_data()
        cute.arch.syncthreads()  # Only when needed
        process_data()
        # No sync needed here
        store_data()
    # 1000 synchronizations
```

### Tip 14: Use Warp-Level Primitives

**Avoid block-level sync when warp-level suffices:**
```python
# ✗ BAD: Block sync for warp operations
@cute.kernel
def block_sync():
    value = compute()
    cute.arch.syncthreads()  # Overkill!
    
    # Use value within warp
    shared_value = warp_shuffle(value)

# ✓ GOOD: Warp sync only
@cute.kernel
def warp_sync():
    value = compute()
    # No block sync needed
    
    # Warp-level operations
    shared_value = warp_shuffle(value)
```

---

## Launch Configuration Optimization

### Tip 15: Tune Block Size

**Find optimal block size:**
```python
def tune_block_size(kernel, data):
    """Find best block size"""
    
    best_time = float('inf')
    best_block_size = None
    
    for block_size in [64, 128, 256, 512, 1024]:
        if block_size > 1024:
            continue
        
        # Launch configuration
        grid = [(data.numel() + block_size - 1) // block_size, 1, 1]
        block = [block_size, 1, 1]
        
        # Benchmark
        time = benchmark_kernel(kernel, grid, block, data)
        
        print(f"Block size {block_size}: {time:.3f} ms")
        
        if time < best_time:
            best_time = time
            best_block_size = block_size
    
    print(f"\nBest: {best_block_size} threads/block")
    return best_block_size

# Typical output:
# Block size 64:   5.234 ms
# Block size 128:  3.456 ms
# Block size 256:  2.789 ms  ← Best
# Block size 512:  3.012 ms
# Block size 1024: 3.567 ms
```

### Tip 16: Maximize Grid Size

**Use all SMs:**
```python
def calculate_grid_size(problem_size: int, block_size: int, num_sms: int = 108):
    """Calculate grid size to saturate GPU"""
    
    # Minimum blocks to cover problem
    min_blocks = (problem_size + block_size - 1) // block_size
    
    # Want at least 4 blocks per SM for good occupancy
    target_blocks = num_sms * 4
    
    # Use larger of the two
    num_blocks = max(min_blocks, target_blocks)
    
    print(f"Problem size: {problem_size}")
    print(f"Block size: {block_size}")
    print(f"Minimum blocks: {min_blocks}")
    print(f"Target blocks: {target_blocks}")
    print(f"Using: {num_blocks} blocks")
    
    return num_blocks
```

---

## Profiling-Guided Optimization

### Tip 17: Profile Before Optimizing

**Always measure first:**
```python
import torch.profiler as profiler

def profile_kernel(kernel, *args):
    """Profile kernel to find bottlenecks"""
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True
    ) as prof:
        # Run kernel
        for _ in range(100):
            kernel.launch(*args)
        torch.cuda.synchronize()
    
    # Print results
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))
    
    # Look for:
    # - High memory time → optimize memory access
    # - Low occupancy → increase occupancy
    # - High sync time → reduce synchronization
```

### Tip 18: Use NSight Compute

**Detailed GPU profiling:**
```bash
# Profile kernel
ncu --set full -o profile python script.py

# View metrics:
# - Memory throughput
# - Compute throughput
# - Occupancy
# - Bank conflicts
# - Cache hit rates
```

---

## Common Performance Pitfalls

### Pitfall 1: Small Problem Size
```python
# ✗ BAD: GPU overhead dominates
n = 100  # Too small!
kernel.launch(grid=[1,1,1], block=[100,1,1])(data)
# Time: launch overhead >> compute

# ✓ GOOD: Amortize overhead
n = 100000  # Large enough
kernel.launch(grid=[391,1,1], block=[256,1,1])(data)
# Time: compute >> launch overhead
```

### Pitfall 2: Unbalanced Work
```python
# ✗ BAD: Some threads idle
@cute.kernel
def unbalanced():
    tid = cute.arch.thread_idx()[0]
    
    if tid < 32:
        # Only first warp works
        expensive_computation()
    # Other warps idle!

# ✓ GOOD: Balanced work
@cute.kernel
def balanced():
    tid = cute.arch.thread_idx()[0]
    
    # All threads work
    idx = tid
    while idx < work_size:
        computation(idx)
        idx += 256
```

### Pitfall 3: Ignoring Alignment
```python
# ✗ BAD: Misaligned access
data_ptr = some_address  # Not aligned!
kernel(data_ptr)  # Slower

# ✓ GOOD: Ensure alignment
assert data_ptr % 128 == 0  # Aligned to cache line
kernel(data_ptr)  # Faster
```

---

## Quick Checklist

### Memory Optimization
- [ ] Memory access coalesced?
- [ ] Using vectorized loads/stores?
- [ ] Data reused from shared memory?
- [ ] No bank conflicts?
- [ ] Shared memory padded if needed?

### Compute Optimization
- [ ] Using tensor cores if available?
- [ ] Loops unrolled?
- [ ] Good instruction-level parallelism?
- [ ] Minimize dependencies?

### Occupancy
- [ ] Block size multiple of 32?
- [ ] Register usage reasonable?
- [ ] Shared memory usage reasonable?
- [ ] Occupancy > 50%?

### Launch Configuration
- [ ] Grid size saturates GPU?
- [ ] Block size tuned?
- [ ] Problem size large enough?
- [ ] Work balanced across threads?

### Synchronization
- [ ] Minimal synchronization?
- [ ] Using warp primitives when possible?
- [ ] No unnecessary barriers?

---

## Summary

**Top 10 tips:**
1. **Coalesce memory access** (10-30× speedup)
2. **Use shared memory for reuse** (5-20× speedup)
3. **Avoid bank conflicts** (2-4× speedup)
4. **Use tensor cores** (50-100× speedup)
5. **Maximize occupancy** (memory-bound)
6. **Balance occupancy/resources** (compute-bound)
7. **Vectorize memory access** (2-4× speedup)
8. **Minimize synchronization** (1.5-2× speedup)
9. **Tune block size** (1.5-3× speedup)
10. **Profile before optimizing** (find real bottlenecks)

**Optimization order:**
1. Get correctness first
2. Profile to find bottleneck
3. Fix biggest bottleneck
4. Repeat 2-3

**Key insight:** The right optimization can give 10-100× speedup. Always profile to find the real bottleneck before optimizing.

---

## Next Steps

- [Profiling](./profiling.md) - Measure performance
- [Autotuning](./autotuning.md) - Automatic optimization
- [Case Studies](../09_case_studies/) - Real examples

---

## Further Reading

- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU Performance Optimization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidelines)
- [CUTLASS Performance](https://github.com/NVIDIA/cutlass/blob/main/media/docs/profiling.md)