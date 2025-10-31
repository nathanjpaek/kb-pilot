---
topic: "profiling"
difficulty: "intermediate"
related_topics: ["performance_tips", "kernel_launch", "debugging"]
---

# Profiling CuTe DSL Kernels

## Overview

Profiling is essential for understanding and optimizing GPU kernel performance. This guide covers tools and techniques for measuring, analyzing, and improving CuTe DSL kernel performance.

**Key concepts:**
- Basic timing measurements
- PyTorch profiler integration
- NSight Systems (timeline analysis)
- NSight Compute (kernel metrics)
- Interpreting profiler output
- Performance metrics

---

## Basic Timing

### Simple Timing with PyTorch

**Manual timing:**
```python
import torch
import time

def time_kernel(kernel, grid, block, *args, num_iters=100, warmup=10):
    """Time a CuTe kernel"""
    
    # Warm up
    for _ in range(warmup):
        kernel.launch(grid=grid, block=block)(*args)
    torch.cuda.synchronize()
    
    # Time
    start = time.time()
    for _ in range(num_iters):
        kernel.launch(grid=grid, block=block)(*args)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time = elapsed / num_iters
    print(f"Average time: {avg_time*1000:.3f} ms")
    
    return avg_time
```

### CUDA Events for Precise Timing

**More accurate GPU timing:**
```python
def time_kernel_cuda_events(kernel, grid, block, *args, num_iters=100):
    """Time kernel using CUDA events (more accurate)"""
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warm up
    for _ in range(10):
        kernel.launch(grid=grid, block=block)(*args)
    torch.cuda.synchronize()
    
    # Time
    start_event.record()
    for _ in range(num_iters):
        kernel.launch(grid=grid, block=block)(*args)
    end_event.record()
    
    # Wait for completion
    torch.cuda.synchronize()
    
    # Calculate time
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters
    
    print(f"Average time: {avg_time_ms:.3f} ms")
    
    return avg_time_ms / 1000  # Return in seconds
```

### Calculating Throughput

**FLOPS and bandwidth:**
```python
def calculate_metrics(
    kernel_time: float,
    problem_size: tuple,
    dtype_size: int = 2,  # Float16 = 2 bytes
    operation: str = "gemm"
):
    """Calculate performance metrics"""
    
    if operation == "gemm":
        M, N, K = problem_size
        
        # FLOPs: 2*M*N*K (multiply-add)
        flops = 2 * M * N * K
        tflops = flops / kernel_time / 1e12
        
        # Bytes transferred (A, B, C)
        bytes_transferred = (M*K + K*N + M*N) * dtype_size
        bandwidth_gb_s = bytes_transferred / kernel_time / 1e9
        
        # Arithmetic intensity
        arithmetic_intensity = flops / bytes_transferred
        
        print(f"\nPerformance Metrics:")
        print(f"  Time: {kernel_time*1000:.3f} ms")
        print(f"  Throughput: {tflops:.2f} TFLOPS")
        print(f"  Bandwidth: {bandwidth_gb_s:.1f} GB/s")
        print(f"  Arithmetic Intensity: {arithmetic_intensity:.1f} FLOP/byte")
        
        return {
            'time': kernel_time,
            'tflops': tflops,
            'bandwidth': bandwidth_gb_s,
            'arithmetic_intensity': arithmetic_intensity
        }
    
    elif operation == "elementwise":
        N = problem_size[0]
        
        # 1 operation per element
        ops = N
        gops = ops / kernel_time / 1e9
        
        # Bytes: read + write
        bytes_transferred = N * dtype_size * 2
        bandwidth_gb_s = bytes_transferred / kernel_time / 1e9
        
        print(f"\nPerformance Metrics:")
        print(f"  Time: {kernel_time*1000:.3f} ms")
        print(f"  Throughput: {gops:.2f} GOPS")
        print(f"  Bandwidth: {bandwidth_gb_s:.1f} GB/s")
        
        return {
            'time': kernel_time,
            'gops': gops,
            'bandwidth': bandwidth_gb_s
        }

# Example usage
time = time_kernel_cuda_events(gemm_kernel, grid, block, A, B, C)
metrics = calculate_metrics(time, (4096, 4096, 4096), operation="gemm")
```

---

## PyTorch Profiler

### Basic Profiling

**Profile with PyTorch profiler:**
```python
import torch.profiler as profiler

def profile_kernel_torch(kernel, grid, block, *args, num_iters=100):
    """Profile kernel with PyTorch profiler"""
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(num_iters):
            kernel.launch(grid=grid, block=block)(*args)
            torch.cuda.synchronize()
    
    # Print results
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
    
    # Export for Chrome
    prof.export_chrome_trace("trace.json")
    print("\nTrace exported to trace.json")
    print("Open chrome://tracing and load the file")
    
    return prof
```

### Analyzing Profiler Output

**Key metrics to look for:**
```python
def analyze_profile(prof):
    """Analyze profiler output"""
    
    # Get kernel stats
    key_averages = prof.key_averages()
    
    total_cuda_time = 0
    total_cpu_time = 0
    kernel_times = []
    
    for event in key_averages:
        if event.device_type == profiler.DeviceType.CUDA:
            total_cuda_time += event.cuda_time_total
            kernel_times.append({
                'name': event.key,
                'time': event.cuda_time_total / 1000,  # Convert to ms
                'calls': event.count
            })
        elif event.device_type == profiler.DeviceType.CPU:
            total_cpu_time += event.cpu_time_total
    
    # Sort by time
    kernel_times.sort(key=lambda x: x['time'], reverse=True)
    
    print("\nTop Kernels by Time:")
    for i, kernel in enumerate(kernel_times[:10]):
        print(f"{i+1}. {kernel['name']}")
        print(f"   Time: {kernel['time']:.2f} ms ({kernel['calls']} calls)")
    
    print(f"\nTotal GPU time: {total_cuda_time/1000:.2f} ms")
    print(f"Total CPU time: {total_cpu_time/1000:.2f} ms")
```

### Memory Profiling

**Track memory usage:**
```python
def profile_memory(kernel, grid, block, *args):
    """Profile memory usage"""
    
    # Reset peak memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Measure before
    mem_before = torch.cuda.memory_allocated()
    
    # Run kernel
    kernel.launch(grid=grid, block=block)(*args)
    torch.cuda.synchronize()
    
    # Measure after
    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()
    
    print(f"Memory Usage:")
    print(f"  Before: {mem_before / 1024**2:.1f} MB")
    print(f"  After:  {mem_after / 1024**2:.1f} MB")
    print(f"  Peak:   {mem_peak / 1024**2:.1f} MB")
    print(f"  Delta:  {(mem_after - mem_before) / 1024**2:.1f} MB")
```

---

## NSight Systems

### Timeline Profiling

**Capture timeline:**
```bash
# Profile with NSight Systems
nsys profile \
    --stats=true \
    --trace=cuda,nvtx,osrt \
    --output=profile \
    python your_script.py

# View with GUI
nsys-ui profile.nsys-rep

# Or generate report
nsys stats profile.nsys-rep
```

### Adding NVTX Markers

**Annotate code for better profiling:**
```python
import torch.cuda.nvtx as nvtx

@cute.jit
def annotated_workflow():
    """Workflow with NVTX markers"""
    
    # Data preparation
    nvtx.range_push("Data Preparation")
    data = torch.randn(4096, 4096, device='cuda')
    nvtx.range_pop()
    
    # Kernel 1
    nvtx.range_push("Preprocessing Kernel")
    preprocess_kernel.launch(grid, block)(data)
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    # Kernel 2
    nvtx.range_push("Main Computation")
    compute_kernel.launch(grid, block)(data)
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    # Kernel 3
    nvtx.range_push("Postprocessing")
    postprocess_kernel.launch(grid, block)(data)
    torch.cuda.synchronize()
    nvtx.range_pop()

# Now timeline will show labeled regions
```

### Interpreting NSight Systems Output

**Key things to look for:**

1. **GPU Utilization**
   - Should be high (>80%)
   - Gaps indicate CPU overhead or synchronization

2. **Kernel Duration**
   - Compare kernel times
   - Identify bottlenecks

3. **Memory Transfers**
   - H2D (Host to Device)
   - D2H (Device to Host)
   - Should be minimal

4. **CPU Activity**
   - Python overhead
   - Framework overhead

---

## NSight Compute

### Detailed Kernel Analysis

**Profile single kernel:**
```bash
# Profile specific kernel
ncu \
    --set full \
    --target-processes all \
    --kernel-name "my_kernel" \
    --launch-skip 0 \
    --launch-count 1 \
    --output profile \
    python your_script.py

# View in GUI
ncu-ui profile.ncu-rep

# Or command line report
ncu --print-summary per-kernel \
    --import profile.ncu-rep
```

### Key Metrics in NSight Compute

**Metrics to examine:**
```python
def interpret_ncu_metrics(metrics: dict):
    """Interpret NSight Compute metrics"""
    
    print("=== NSight Compute Metrics Analysis ===\n")
    
    # 1. Memory Throughput
    mem_throughput = metrics.get('memory_throughput', 0)
    peak_mem_bw = metrics.get('peak_memory_bandwidth', 1555)  # GB/s for A100
    mem_utilization = (mem_throughput / peak_mem_bw) * 100
    
    print(f"Memory:")
    print(f"  Throughput: {mem_throughput:.1f} GB/s")
    print(f"  Utilization: {mem_utilization:.1f}%")
    
    if mem_utilization < 60:
        print("  ⚠️  Low memory utilization - check coalescing")
    
    # 2. Compute Throughput
    compute_throughput = metrics.get('compute_throughput', 0)
    peak_flops = metrics.get('peak_flops', 312)  # TFLOPS for A100
    compute_utilization = (compute_throughput / peak_flops) * 100
    
    print(f"\nCompute:")
    print(f"  Throughput: {compute_throughput:.1f} TFLOPS")
    print(f"  Utilization: {compute_utilization:.1f}%")
    
    if compute_utilization < 60:
        print("  ⚠️  Low compute utilization - check occupancy")
    
    # 3. Occupancy
    occupancy = metrics.get('occupancy', 0)
    
    print(f"\nOccupancy:")
    print(f"  Achieved: {occupancy:.1f}%")
    
    if occupancy < 50:
        print("  ⚠️  Low occupancy")
        print("     - Check register usage")
        print("     - Check shared memory usage")
        print("     - Try smaller block size")
    
    # 4. L1/L2 Cache
    l1_hit_rate = metrics.get('l1_cache_hit_rate', 0)
    l2_hit_rate = metrics.get('l2_cache_hit_rate', 0)
    
    print(f"\nCache:")
    print(f"  L1 Hit Rate: {l1_hit_rate:.1f}%")
    print(f"  L2 Hit Rate: {l2_hit_rate:.1f}%")
    
    if l1_hit_rate < 50:
        print("  ⚠️  Low L1 cache hit rate - check data reuse")
    
    # 5. Bank Conflicts
    bank_conflicts = metrics.get('shared_memory_bank_conflicts', 0)
    
    print(f"\nShared Memory:")
    print(f"  Bank Conflicts: {bank_conflicts}")
    
    if bank_conflicts > 0:
        print("  ⚠️  Bank conflicts detected - add padding")
    
    # 6. Warp Execution Efficiency
    warp_efficiency = metrics.get('warp_execution_efficiency', 0)
    
    print(f"\nWarp Execution:")
    print(f"  Efficiency: {warp_efficiency:.1f}%")
    
    if warp_efficiency < 80:
        print("  ⚠️  Low warp efficiency - check for divergence")
```

### Roofline Analysis in NSight Compute

**Understanding performance bounds:**
```python
def roofline_analysis(kernel_time, flops, bytes_transferred):
    """Generate roofline analysis"""
    
    # A100 specs
    peak_flops = 312e12  # 312 TFLOPS
    peak_bandwidth = 1555e9  # 1555 GB/s
    
    # Achieved
    achieved_flops = flops / kernel_time
    achieved_bandwidth = bytes_transferred / kernel_time
    arithmetic_intensity = flops / bytes_transferred
    
    # Ridge point (where compute and memory bound meet)
    ridge_point = peak_flops / peak_bandwidth
    
    print("=== Roofline Analysis ===\n")
    print(f"Arithmetic Intensity: {arithmetic_intensity:.2f} FLOP/byte")
    print(f"Ridge Point: {ridge_point:.2f} FLOP/byte")
    
    if arithmetic_intensity < ridge_point:
        print("\n📊 Kernel is MEMORY BOUND")
        print(f"   Achieved: {achieved_bandwidth/1e9:.1f} GB/s")
        print(f"   Peak: {peak_bandwidth/1e9:.1f} GB/s")
        print(f"   Efficiency: {achieved_bandwidth/peak_bandwidth*100:.1f}%")
        print("\n   Optimization suggestions:")
        print("   - Increase data reuse (tiling)")
        print("   - Improve memory coalescing")
        print("   - Use shared memory caching")
    else:
        print("\n🖥️  Kernel is COMPUTE BOUND")
        print(f"   Achieved: {achieved_flops/1e12:.1f} TFLOPS")
        print(f"   Peak: {peak_flops/1e12:.1f} TFLOPS")
        print(f"   Efficiency: {achieved_flops/peak_flops*100:.1f}%")
        print("\n   Optimization suggestions:")
        print("   - Use tensor cores")
        print("   - Increase instruction-level parallelism")
        print("   - Unroll loops")
```

---

## Comparative Profiling

### Comparing Multiple Implementations
```python
def compare_implementations(implementations: dict, *args):
    """Compare multiple kernel implementations"""
    
    import pandas as pd
    
    results = []
    
    for name, kernel_func in implementations.items():
        print(f"\n=== Testing {name} ===")
        
        # Time kernel
        time = time_kernel_cuda_events(kernel_func, *args)
        
        # Calculate metrics
        metrics = calculate_metrics(time, problem_size=(4096, 4096, 4096))
        
        results.append({
            'Implementation': name,
            'Time (ms)': time * 1000,
            'TFLOPS': metrics['tflops'],
            'Bandwidth (GB/s)': metrics['bandwidth']
        })
    
    # Create comparison table
    df = pd.DataFrame(results)
    df = df.sort_values('Time (ms)')
    
    print("\n=== Comparison ===")
    print(df.to_string(index=False))
    
    # Calculate speedups
    baseline_time = df.iloc[-1]['Time (ms)']
    df['Speedup'] = baseline_time / df['Time (ms)']
    
    print("\n=== Speedups vs Slowest ===")
    print(df[['Implementation', 'Speedup']].to_string(index=False))
    
    return df

# Usage
implementations = {
    'Naive': naive_gemm,
    'Tiled': tiled_gemm,
    'Pipelined': pipelined_gemm,
    'Tensor Core': tensor_core_gemm
}

comparison = compare_implementations(implementations, A, B, C)
```

### A/B Testing Optimizations
```python
def ab_test_optimization(baseline_kernel, optimized_kernel, *args):
    """Compare baseline vs optimized version"""
    
    print("=== A/B Test ===\n")
    
    # Baseline
    print("Baseline:")
    baseline_time = time_kernel_cuda_events(baseline_kernel, *args)
    baseline_metrics = calculate_metrics(baseline_time, (4096, 4096, 4096))
    
    # Optimized
    print("\nOptimized:")
    optimized_time = time_kernel_cuda_events(optimized_kernel, *args)
    optimized_metrics = calculate_metrics(optimized_time, (4096, 4096, 4096))
    
    # Compare
    speedup = baseline_time / optimized_time
    tflops_improvement = optimized_metrics['tflops'] - baseline_metrics['tflops']
    
    print("\n=== Comparison ===")
    print(f"Speedup: {speedup:.2f}×")
    print(f"TFLOPS Improvement: +{tflops_improvement:.1f}")
    
    if speedup > 1.1:
        print("✓ Optimization successful!")
    elif speedup > 0.95:
        print("≈ No significant change")
    else:
        print("✗ Optimization made it slower!")
    
    return speedup
```

---

## Automated Performance Testing

### Regression Testing
```python
import json
from datetime import datetime

class PerformanceTracker:
    """Track kernel performance over time"""
    
    def __init__(self, log_file="performance_log.json"):
        self.log_file = log_file
        self.load_history()
    
    def load_history(self):
        """Load performance history"""
        try:
            with open(self.log_file, 'r') as f:
                self.history = json.load(f)
        except FileNotFoundError:
            self.history = {}
    
    def save_history(self):
        """Save performance history"""
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record(self, kernel_name: str, metrics: dict):
        """Record performance metrics"""
        
        if kernel_name not in self.history:
            self.history[kernel_name] = []
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        self.history[kernel_name].append(entry)
        self.save_history()
    
    def check_regression(self, kernel_name: str, current_time: float, threshold: float = 1.1):
        """Check for performance regression"""
        
        if kernel_name not in self.history or len(self.history[kernel_name]) == 0:
            print(f"No history for {kernel_name}")
            return False
        
        # Get last recorded time
        last_entry = self.history[kernel_name][-1]
        last_time = last_entry['metrics']['time']
        
        # Compare
        ratio = current_time / last_time
        
        if ratio > threshold:
            print(f"⚠️  REGRESSION DETECTED for {kernel_name}!")
            print(f"   Previous: {last_time*1000:.3f} ms")
            print(f"   Current:  {current_time*1000:.3f} ms")
            print(f"   Slowdown: {ratio:.2f}×")
            return True
        elif ratio < 0.9:
            print(f"✓ Performance improved for {kernel_name}!")
            print(f"   Previous: {last_time*1000:.3f} ms")
            print(f"   Current:  {current_time*1000:.3f} ms")
            print(f"   Speedup:  {1/ratio:.2f}×")
        else:
            print(f"Performance stable for {kernel_name}")
        
        return False

# Usage
tracker = PerformanceTracker()

# Benchmark kernel
time = time_kernel_cuda_events(my_kernel, grid, block, *args)
metrics = {'time': time, 'tflops': 150.5}

# Record
tracker.record('my_kernel', metrics)

# Check for regression
tracker.check_regression('my_kernel', time)
```

---

## Profiling Best Practices

### ✅ DO

**Always warm up:**
```python
# Warm up GPU
for _ in range(10):
    kernel.launch(grid, block)(*args)
torch.cuda.synchronize()

# Then profile
```

**Use CUDA events for accuracy:**
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
kernel.launch(grid, block)(*args)
end.record()

torch.cuda.synchronize()
time = start.elapsed_time(end)
```

**Profile multiple iterations:**
```python
# Average over many runs
for _ in range(100):
    kernel.launch(grid, block)(*args)
```

**Annotate with NVTX:**
```python
nvtx.range_push("Critical Section")
kernel.launch(grid, block)(*args)
nvtx.range_pop()
```

### ❌ DON'T

**Don't profile debug builds:**
```python
# ✗ BAD: Debug mode is slow
# Profile release builds only
```

**Don't forget to synchronize:**
```python
# ✗ BAD: Timing without sync
start = time.time()
kernel.launch(grid, block)(*args)
end = time.time()  # Wrong! Kernel still running

# ✓ GOOD: Sync first
kernel.launch(grid, block)(*args)
torch.cuda.synchronize()
end = time.time()
```

**Don't profile cold cache:**
```python
# ✗ BAD: First run is always slower
time = time_kernel(kernel, ...)  # Cold cache

# ✓ GOOD: Warm up first
warm_up(kernel, ...)
time = time_kernel(kernel, ...)
```

---

## Summary

**Profiling tools:**
- **Basic timing:** Python `time` or CUDA events
- **PyTorch profiler:** Timeline and memory
- **NSight Systems:** CPU/GPU timeline
- **NSight Compute:** Detailed kernel metrics

**Key metrics:**
- Time (ms)
- Throughput (TFLOPS/GOPS)
- Bandwidth (GB/s)
- Occupancy (%)
- Cache hit rates
- Bank conflicts

**Profiling workflow:**
1. Basic timing to identify slow kernels
2. PyTorch profiler for overview
3. NSight Systems for timeline
4. NSight Compute for deep dive
5. Optimize bottlenecks
6. Repeat

**Key insight:** Profile before optimizing. Measure the actual bottleneck, not what you think it is. 90% of the time is spent in 10% of the code - find that 10%.

---

## Next Steps

- [Performance Tips](./performance_tips.md) - What to optimize
- [Autotuning](./autotuning.md) - Automated optimization
- [Case Studies](../09_case_studies/) - Real examples

---

## Further Reading

- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NSight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [NSight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)