---
topic: "debugging_techniques"
difficulty: "intermediate"
related_topics: ["common_errors", "error_messages", "profiling"]
---

# Debugging Techniques for CuTe DSL

## Overview

Debugging GPU kernels is challenging because code runs on a different processor with limited visibility. This guide covers practical techniques for finding and fixing bugs in CuTe DSL kernels.

**Key concepts:**
- Print debugging
- Assertions and validation
- Isolating problems
- Reducing test cases
- Comparing with reference
- Using debuggers
- Incremental development

---

## Debugging Strategy

### The Scientific Method

**Systematic debugging approach:**

1. **Observe** - What's wrong? Crash? Wrong output? Slow?
2. **Hypothesize** - What might cause this?
3. **Test** - Add prints/checks to test hypothesis
4. **Analyze** - Look at results
5. **Fix** - Make targeted change
6. **Verify** - Confirm fix works

### Start Simple

**Always start with the simplest possible test:**
```python
# ✗ BAD: Debug with huge problem
A = torch.randn(8192, 8192, device='cuda')
kernel.launch(...)(A)  # Where's the bug??

# ✓ GOOD: Debug with tiny problem
A = torch.randn(8, 8, device='cuda')
print(A)
kernel.launch(...)(A)
print(A)  # Easy to see what happened
```

---

## Print Debugging

### Basic Printf in Kernels

**Print from device code:**
```python
@cute.kernel
def debug_print(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    # Print from thread 0 only (avoid spam)
    if tid == 0 and bid == 0:
        cute.printf("Block %d, Thread %d: data[0] = %f\n", bid, tid, data[0])
    
    # Print from specific thread
    if tid == 5 and bid == 0:
        cute.printf("Thread 5: processing data[5] = %f\n", data[5])

# Launch
debug_print.launch(grid=[4,1,1], block=[256,1,1])(cute.from_dlpack(data))
torch.cuda.synchronize()

# Output:
# Block 0, Thread 0: data[0] = 1.234
# Thread 5: processing data[5] = 5.678
```

### Conditional Printing

**Print only when conditions met:**
```python
@cute.kernel
def conditional_debug(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    idx = bid * 256 + tid
    
    if idx < cute.size(data):
        value = data[idx]
        
        # Print problematic values
        if value < 0.0:
            cute.printf("WARNING: Negative value at idx=%d: %f\n", idx, value)
        
        # Print out of range values
        if value > 100.0:
            cute.printf("WARNING: Large value at idx=%d: %f\n", idx, value)
        
        data[idx] = process(value)
```

### Print Array Contents

**Dump array for inspection:**
```python
@cute.kernel
def print_array(data: cute.Tensor, n: cutlass.Int32):
    tid = cute.arch.thread_idx()[0]
    
    # Only thread 0 prints
    if tid == 0:
        cute.printf("Array contents:\n")
        for i in range(n):
            cute.printf("  [%d] = %f\n", i, data[i])

# Print small array
small_data = torch.tensor([1.0, 2.0, 3.0, -4.0, 5.0], device='cuda')
print_array.launch(grid=[1,1,1], block=[1,1,1])(
    cute.from_dlpack(small_data), 
    n=5
)
torch.cuda.synchronize()

# Output:
# Array contents:
#   [0] = 1.000000
#   [1] = 2.000000
#   [2] = 3.000000
#   [3] = -4.000000
#   [4] = 5.000000
```

### Limiting Print Output

**Control print spam:**
```python
@cute.kernel
def limited_prints(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    # Method 1: Only first block
    if bid == 0 and tid < 10:
        cute.printf("Thread %d: %f\n", tid, data[tid])
    
    # Method 2: Only specific threads
    if tid % 32 == 0:  # Every 32nd thread (warp leaders)
        cute.printf("Warp leader %d\n", tid)
    
    # Method 3: Only first N total prints
    global_tid = bid * cute.arch.block_dim_x() + tid
    if global_tid < 10:
        cute.printf("First 10 threads: %d\n", global_tid)
```

---

## Assertions and Validation

### Runtime Assertions in Kernels

**Check invariants:**
```python
@cute.kernel
def with_assertions(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Assert dimensions match
    M_A, K_A = cute.shape(A)
    K_B, N_B = cute.shape(B)
    M_C, N_C = cute.shape(C)
    
    if tid == 0:
        # Check K dimensions match
        cute.assert_(K_A == K_B, "K dimension mismatch")
        # Check output dimensions match
        cute.assert_(M_A == M_C, "M dimension mismatch")
        cute.assert_(N_B == N_C, "N dimension mismatch")
    
    cute.arch.syncthreads()
    
    # Continue with computation
    # ...
```

### Host-Side Validation

**Validate before and after kernel:**
```python
def validate_and_launch(kernel, data, expected_min=None, expected_max=None):
    """Launch kernel with validation"""
    
    # Pre-launch validation
    print("Pre-launch validation:")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Device: {data.device}")
    print(f"  Is contiguous: {data.is_contiguous()}")
    print(f"  Min: {data.min().item():.6f}")
    print(f"  Max: {data.max().item():.6f}")
    print(f"  Mean: {data.mean().item():.6f}")
    
    # Check for NaN/Inf
    if torch.isnan(data).any():
        print("  ⚠️  WARNING: Input contains NaN!")
    if torch.isinf(data).any():
        print("  ⚠️  WARNING: Input contains Inf!")
    
    # Save copy for comparison
    data_before = data.clone()
    
    # Launch kernel
    print("\nLaunching kernel...")
    kernel.launch(grid=[10,1,1], block=[256,1,1])(cute.from_dlpack(data))
    torch.cuda.synchronize()
    
    # Post-launch validation
    print("\nPost-launch validation:")
    print(f"  Min: {data.min().item():.6f}")
    print(f"  Max: {data.max().item():.6f}")
    print(f"  Mean: {data.mean().item():.6f}")
    
    # Check for NaN/Inf
    if torch.isnan(data).any():
        print("  ⚠️  ERROR: Output contains NaN!")
        # Find where
        nan_indices = torch.where(torch.isnan(data))
        print(f"  NaN at indices: {nan_indices}")
    
    if torch.isinf(data).any():
        print("  ⚠️  ERROR: Output contains Inf!")
    
    # Check expected range
    if expected_min is not None:
        if data.min() < expected_min:
            print(f"  ⚠️  ERROR: Min {data.min().item()} < expected {expected_min}")
    
    if expected_max is not None:
        if data.max() > expected_max:
            print(f"  ⚠️  ERROR: Max {data.max().item()} > expected {expected_max}")
    
    # Check if data changed
    if torch.equal(data, data_before):
        print("  ⚠️  WARNING: Data unchanged!")
    else:
        print(f"  ✓ Data modified ({torch.sum(data != data_before).item()} elements)")
    
    print("\n✓ Validation complete")

# Usage
data = torch.randn(1024, device='cuda')
validate_and_launch(my_kernel, data, expected_min=-10.0, expected_max=10.0)
```

---

## Isolating Problems

### Binary Search for Bug Location

**Narrow down where bug occurs:**
```python
@cute.kernel
def complex_kernel(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Step 1: Load
    value = data[tid]
    cute.printf("After load: %f\n", value)  # Check point 1
    
    # Step 2: Process A
    value = process_a(value)
    cute.printf("After process_a: %f\n", value)  # Check point 2
    
    # Step 3: Process B
    value = process_b(value)
    cute.printf("After process_b: %f\n", value)  # Check point 3
    
    # Step 4: Store
    data[tid] = value
    cute.printf("After store: %f\n", data[tid])  # Check point 4

# Run and see where output becomes wrong
# If wrong after process_a but OK after load → bug in process_a
```

### Disable Code Sections

**Comment out code to isolate:**
```python
@cute.kernel
def debug_by_disabling(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    # Section 1: Works fine
    result = data[tid] * 2.0
    
    # Section 2: Suspected bug here
    # result = buggy_function(result)  # DISABLED
    result = result  # Bypass for now
    
    # Section 3: More processing
    result = result + 1.0
    
    data[tid] = result

# If output correct with section disabled → bug in that section
```

### Simplify to Minimal Example

**Reduce to smallest failing case:**
```python
# ✗ COMPLEX: Hard to debug
@cute.kernel
def complex_gemm(A, B, C):
    # 500 lines of optimized code
    # Tiling, pipelining, tensor cores, etc.
    pass

# ✓ SIMPLE: Easy to debug
@cute.kernel
def simple_gemm(A, B, C):
    # Naive implementation
    tid = cute.arch.thread_idx()[0]
    
    for i in range(cute.shape(C)[0]):
        for j in range(cute.shape(C)[1]):
            acc = 0.0
            for k in range(cute.shape(A)[1]):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc

# Get simple version working first
# Then add optimizations one at a time
```

---

## Comparing with Reference

### Reference Implementation

**Compare against known-good implementation:**
```python
def test_kernel_against_reference(kernel, A, B):
    """Test kernel output against PyTorch reference"""
    
    # Kernel output
    C_kernel = torch.zeros(A.shape[0], B.shape[1], device='cuda', dtype=A.dtype)
    kernel.launch(grid=grid, block=block)(
        cute.from_dlpack(A),
        cute.from_dlpack(B),
        cute.from_dlpack(C_kernel)
    )
    torch.cuda.synchronize()
    
    # Reference output
    C_reference = torch.matmul(A, B)
    
    # Compare
    print("Comparison Results:")
    print(f"  Max absolute difference: {(C_kernel - C_reference).abs().max().item()}")
    print(f"  Mean absolute difference: {(C_kernel - C_reference).abs().mean().item()}")
    
    # Check tolerance
    if torch.allclose(C_kernel, C_reference, rtol=1e-3, atol=1e-5):
        print("  ✓ Results match!")
    else:
        print("  ✗ Results DO NOT match!")
        
        # Find worst mismatches
        diff = (C_kernel - C_reference).abs()
        worst_indices = torch.topk(diff.flatten(), k=5).indices
        
        print("\n  Worst 5 mismatches:")
        for idx in worst_indices:
            i = idx // C_kernel.shape[1]
            j = idx % C_kernel.shape[1]
            print(f"    [{i}, {j}]: kernel={C_kernel[i,j].item():.6f}, "
                  f"reference={C_reference[i,j].item():.6f}, "
                  f"diff={diff[i,j].item():.6f}")

# Usage
A = torch.randn(64, 64, device='cuda', dtype=torch.float32)
B = torch.randn(64, 64, device='cuda', dtype=torch.float32)
test_kernel_against_reference(my_gemm_kernel, A, B)
```

### Element-by-Element Comparison

**Check specific elements:**
```python
def debug_specific_elements(kernel_output, reference_output, threshold=1e-3):
    """Debug specific mismatched elements"""
    
    diff = (kernel_output - reference_output).abs()
    mismatched = diff > threshold
    
    if not mismatched.any():
        print("✓ All elements match within threshold")
        return
    
    num_mismatched = mismatched.sum().item()
    print(f"⚠️  {num_mismatched} elements mismatched")
    
    # Get indices of mismatches
    indices = torch.where(mismatched)
    
    # Print first 10 mismatches
    print("\nFirst 10 mismatches:")
    for i in range(min(10, num_mismatched)):
        idx = tuple(ind[i].item() for ind in indices)
        kernel_val = kernel_output[idx].item()
        ref_val = reference_output[idx].item()
        diff_val = diff[idx].item()
        
        print(f"  Index {idx}:")
        print(f"    Kernel:    {kernel_val:.8f}")
        print(f"    Reference: {ref_val:.8f}")
        print(f"    Diff:      {diff_val:.8f}")

# Usage
debug_specific_elements(C_kernel, C_reference)
```

---

## Incremental Development

### Test Each Component

**Build and test incrementally:**
```python
# Step 1: Test data loading
@cute.kernel
def test_load(A: cute.Tensor, temp: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    temp[tid] = A[tid]  # Just copy
    # Verify: temp should equal A

# Step 2: Test processing
@cute.kernel
def test_process(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    value = data[tid]
    processed = process_function(value)
    data[tid] = processed
    # Verify: check if processing correct

# Step 3: Test storage
@cute.kernel
def test_store(temp: cute.Tensor, output: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    output[tid] = temp[tid]
    # Verify: output should equal temp

# Step 4: Combine (now we know each part works)
@cute.kernel
def combined(A: cute.Tensor, output: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    value = A[tid]
    processed = process_function(value)
    output[tid] = processed
```

### Add Features One at a Time
```python
# Version 1: Basic (works)
@cute.kernel
def v1_basic(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    data[tid] = data[tid] * 2.0

# Test v1
test(v1_basic)  # ✓ Works

# Version 2: Add tiling (test again)
@cute.kernel
def v2_tiled(data: cute.Tensor):
    smem = cute.make_smem_tensor(cute.make_shape(256), cutlass.Float32)
    tid = cute.arch.thread_idx()[0]
    
    smem[tid] = data[tid]
    cute.arch.syncthreads()
    data[tid] = smem[tid] * 2.0

# Test v2
test(v2_tiled)  # ✓ Works

# Version 3: Add pipelining (test again)
@cute.kernel
def v3_pipelined(data: cute.Tensor):
    # Add pipelining...
    pass

# Test v3
test(v3_pipelined)  # If fails, bug is in new code

# This way, you always know which change introduced the bug
```

---

## Debugging Tools

### CUDA Memcheck

**Detect memory errors:**
```bash
# Run with cuda-memcheck
cuda-memcheck python your_script.py

# Common errors detected:
# - Out of bounds access
# - Race conditions
# - Misaligned access
# - Uninitialized memory

# Example output:
# ========= Invalid __global__ write of size 4
# =========     at 0x00000148 in kernel(float*)
# =========     by thread (256,0,0) in block (0,0,0)
```

### Compute Sanitizer

**Modern replacement for cuda-memcheck:**
```bash
# Run with compute-sanitizer
compute-sanitizer python your_script.py

# More detailed than cuda-memcheck
# Shows exact line numbers
# Better error messages

# Example:
# ========= Invalid __global__ write of size 4 bytes
# =========     at file.cu:23
# =========     by thread (256,0,0) in block (0,0,0)
# =========     Address 0x7f1234567890 is 256 bytes after a 1024 byte allocation
```

### CUDA-GDB

**Interactive debugger:**
```bash
# Compile with debug info
export CUDA_DEBUGGER_ENABLE=1

# Run cuda-gdb
cuda-gdb python

# GDB commands:
# (cuda-gdb) break kernel_name
# (cuda-gdb) run your_script.py
# (cuda-gdb) cuda thread (0,0,0) (0,0)  # Switch to thread
# (cuda-gdb) print variable_name
# (cuda-gdb) continue
```

---

## Debugging Specific Issues

### Debugging NaN/Inf

**Track down where NaN appears:**
```python
@cute.kernel
def debug_nan(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    
    value = data[tid]
    
    # Check for NaN at each step
    if cute.isnan(value):
        cute.printf("NaN at input: tid=%d\n", tid)
    
    result = process1(value)
    if cute.isnan(result):
        cute.printf("NaN after process1: tid=%d, input=%f\n", tid, value)
    
    result = process2(result)
    if cute.isnan(result):
        cute.printf("NaN after process2: tid=%d\n", tid)
    
    data[tid] = result

# Common causes:
# - Division by zero
# - Square root of negative
# - Log of negative/zero
# - Overflow/underflow
```

### Debugging Race Conditions

**Detect races with validation:**
```python
@cute.kernel
def detect_race(data: cute.Tensor):
    smem = cute.make_smem_tensor(cute.make_shape(256), cutlass.Float32)
    
    tid = cute.arch.thread_idx()[0]
    
    # Write to shared memory
    smem[tid] = tid
    
    # MISSING SYNC → RACE CONDITION
    # cute.arch.syncthreads()  # Should be here!
    
    # Read from shared memory (may read stale data)
    value = smem[(tid + 1) % 256]
    
    # Validate: value should be (tid + 1) % 256
    expected = cutlass.Float32((tid + 1) % 256)
    if value != expected:
        cute.printf("RACE DETECTED: tid=%d, expected=%f, got=%f\n",
                   tid, expected, value)

# Run multiple times - races are non-deterministic
for i in range(10):
    print(f"Run {i+1}")
    kernel.launch(...)(data)
    torch.cuda.synchronize()
```

### Debugging Performance Issues

**Identify bottlenecks:**
```python
def debug_performance(kernel, *args):
    """Profile different parts of kernel"""
    
    # Time overall
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    kernel.launch(...)(*args)
    end.record()
    torch.cuda.synchronize()
    
    total_time = start.elapsed_time(end)
    print(f"Total time: {total_time:.3f} ms")
    
    # Use NSight Compute for detailed breakdown
    # ncu --set full python script.py
    
    # Check:
    # - Memory bandwidth utilization
    # - Compute utilization
    # - Occupancy
    # - Bank conflicts
    # - Cache hit rates
```

---

## Debugging Workflow

### Complete Debugging Session
```python
def debug_kernel_systematically(kernel, *args):
    """Systematic debugging approach"""
    
    print("=== SYSTEMATIC KERNEL DEBUGGING ===\n")
    
    # 1. Validate inputs
    print("1. Validating inputs...")
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            print(f"   Arg {i}: shape={arg.shape}, dtype={arg.dtype}, "
                  f"device={arg.device}, contiguous={arg.is_contiguous()}")
            if torch.isnan(arg).any():
                print(f"   ⚠️  WARNING: Arg {i} contains NaN")
            if torch.isinf(arg).any():
                print(f"   ⚠️  WARNING: Arg {i} contains Inf")
    
    # 2. Try with smallest possible input
    print("\n2. Testing with minimal input...")
    small_args = [arg[:8] if isinstance(arg, torch.Tensor) else arg 
                  for arg in args]
    try:
        kernel.launch(grid=[1,1,1], block=[8,1,1])(*small_args)
        torch.cuda.synchronize()
        print("   ✓ Minimal test passed")
    except Exception as e:
        print(f"   ✗ Minimal test failed: {e}")
        return
    
    # 3. Compare with reference
    print("\n3. Comparing with reference...")
    # (implement reference comparison)
    
    # 4. Run with full input
    print("\n4. Running with full input...")
    try:
        kernel.launch(grid=grid, block=block)(*args)
        torch.cuda.synchronize()
        print("   ✓ Full test passed")
    except Exception as e:
        print(f"   ✗ Full test failed: {e}")
        return
    
    # 5. Validate output
    print("\n5. Validating output...")
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            if torch.isnan(arg).any():
                print(f"   ⚠️  ERROR: Output {i} contains NaN")
            if torch.isinf(arg).any():
                print(f"   ⚠️  ERROR: Output {i} contains Inf")
    
    print("\n=== DEBUGGING COMPLETE ===")

# Usage
debug_kernel_systematically(my_kernel, A, B, C)
```

---

## Best Practices

### ✅ DO

**Start simple:**
```python
# Test with 8 elements before 8192
small_data = torch.randn(8, device='cuda')
kernel.launch(grid=[1,1,1], block=[8,1,1])(cute.from_dlpack(small_data))
```

**Print liberally during development:**
```python
# Add prints everywhere initially
# Remove after bug found
```

**Compare with reference:**
```python
# Always have a reference to compare against
ref = torch.matmul(A, B)
assert torch.allclose(output, ref)
```

**Build incrementally:**
```python
# Test each new feature
# Don't add everything at once
```

### ❌ DON'T

**Don't debug in production:**
```python
# ✗ BAD: Debug with full dataset
# ✓ GOOD: Debug with tiny subset
```

**Don't make multiple changes:**
```python
# ✗ BAD: Add 5 features, then debug
# ✓ GOOD: Add 1 feature, test, repeat
```

**Don't ignore warnings:**
```python
# Every warning is a potential bug
# Fix warnings immediately
```

---

## Summary

**Debugging strategy:**
1. Start with smallest possible input
2. Add prints liberally
3. Compare with reference implementation
4. Isolate problem (binary search)
5. Fix incrementally
6. Validate thoroughly

**Essential techniques:**
- Print debugging (cuda.printf)
- Assertions and validation
- Reference comparison
- Incremental development
- Systematic approach

**Tools:**
- cuda-memcheck / compute-sanitizer
- cuda-gdb
- NSight Compute/Systems
- PyTorch profiler

**Key insight:** Most bugs are found fastest by starting simple, using prints strategically, and comparing with a known-good reference. Tools are helpful but systematic thinking is more important.

---

## Next Steps

- [Common Errors](./common_errors.md) - What errors mean
- [Error Messages](./error_messages.md) - Decoding errors
- [Validation](./validation.md) - Testing strategies

---

## Further Reading

- [CUDA-GDB User Guide](https://docs.nvidia.com/cuda/cuda-gdb/)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/)
- [NSight Compute](https://docs.nvidia.com/nsight-compute/)