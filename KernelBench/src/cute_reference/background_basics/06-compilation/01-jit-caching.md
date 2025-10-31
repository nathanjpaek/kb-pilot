---
topic: "jit_caching"
difficulty: "intermediate"
related_topics: ["jit_compilation", "code_generation", "static_vs_dynamic"]
---

# JIT Caching in CuTe DSL

## Overview

CuTe DSL caches compiled kernels to avoid recompiling identical code. Understanding the caching mechanism is crucial for development workflow, debugging, and deployment.

**Key concepts:**
- Kernel cache location and structure
- Cache key generation
- When recompilation occurs
- Cache invalidation
- Precompilation with `cute.compile()`
- Managing cache in production

---

## How Caching Works

### First Compilation

**When you first run a kernel:**
```python
@cute.kernel
def my_kernel(data: cute.Tensor, n: cutlass.Int32):
    tid = cute.arch.thread_idx()[0]
    if tid < n:
        data[tid] = 0.0

# First call: compiles (slow, 1-10 seconds)
my_kernel.launch(grid=[4,1,1], block=[256,1,1])(data, n=1000)
# Kernel compiled and cached
```

**What happens:**
1. Parse Python AST
2. Generate MLIR IR
3. Compile to CUDA C++
4. Invoke NVCC to generate PTX/SASS
5. **Save to cache**
6. Load and execute

**First call timing:**
```
Total time: ~2-10 seconds
- Compilation: 1.5-9 seconds
- Execution: 0.5-1 seconds
```

### Subsequent Calls

**Same kernel, later calls:**
```python
# Second call: uses cache (fast, microseconds)
my_kernel.launch(grid=[4,1,1], block=[256,1,1])(data, n=1000)
# Kernel loaded from cache

# Third call: still cached
my_kernel.launch(grid=[4,1,1], block=[256,1,1])(data, n=2000)
# n is dynamic parameter, same cached kernel
```

**Cached call timing:**
```
Total time: ~0.5-1 milliseconds
- Cache lookup: 0.001 milliseconds
- Execution: 0.5-1 milliseconds
```

**Speedup:** 1000-10000× faster!

---

## Cache Location

### Default Cache Directory

**Where cached kernels live:**
```bash
# Linux/Mac
~/.cache/cutlass/
├── kernels/
│   ├── my_kernel_abc123def.cubin    # Compiled binary
│   ├── my_kernel_abc123def.cu       # Generated CUDA C++
│   ├── my_kernel_abc123def.ptx      # PTX assembly
│   └── metadata.json                # Cache metadata
```

**Windows:**
```
C:\Users\<username>\AppData\Local\cutlass\cache\
```

### Inspecting Cache

**Check cache size:**
```bash
du -sh ~/.cache/cutlass/
# 245M    .cache/cutlass/
```

**List cached kernels:**
```bash
ls -lh ~/.cache/cutlass/kernels/
# -rw-r--r-- 1 user user  42K my_kernel_abc123.cubin
# -rw-r--r-- 1 user user  15K my_kernel_abc123.cu
# -rw-r--r-- 1 user user  28K my_kernel_abc123.ptx
```

**View generated CUDA C++:**
```bash
cat ~/.cache/cutlass/kernels/my_kernel_abc123.cu
```

---

## Cache Key Generation

### What Determines Cache Key

**Cache key based on:**

1. **Function source code**
```python
   @cute.kernel
   def kernel_v1(data: cute.Tensor):
       data[0] = 1.0  # Version 1
   
   @cute.kernel
   def kernel_v1(data: cute.Tensor):
       data[0] = 2.0  # Version 2 - different cache key
```

2. **Parameter types**
```python
   kernel.launch(...)(float32_tensor)  # Cache key A
   kernel.launch(...)(float16_tensor)  # Cache key B (different!)
```

3. **Constexpr values**
```python
   kernel.launch(...)(data, tile_size=128)  # Cache key A
   kernel.launch(...)(data, tile_size=256)  # Cache key B (different!)
```

4. **Compilation options**
```python
   os.environ['CUTLASS_DEBUG'] = '1'
   kernel.launch(...)(data)  # Cache key A
   
   os.environ['CUTLASS_DEBUG'] = '0'
   kernel.launch(...)(data)  # Cache key B (different!)
```

5. **Target architecture**
```python
   os.environ['CUTLASS_NVCC_ARCHS'] = '80'  # Ampere
   kernel.launch(...)(data)  # Cache key A
   
   os.environ['CUTLASS_NVCC_ARCHS'] = '90'  # Hopper
   kernel.launch(...)(data)  # Cache key B (different!)
```

### What Does NOT Affect Cache Key

**✓ Same cache key:**

1. **Runtime parameter values:**
```python
   kernel.launch(...)(data, n=100)  # Cache once
   kernel.launch(...)(data, n=200)  # Use cache
   kernel.launch(...)(data, n=300)  # Use cache
```

2. **Launch configuration:**
```python
   kernel.launch(grid=[16,1,1], block=[256,1,1])(data)  # Cache once
   kernel.launch(grid=[32,1,1], block=[256,1,1])(data)  # Use cache
```

3. **Tensor data:**
```python
   kernel.launch(...)(tensor_a)  # Cache once
   kernel.launch(...)(tensor_b)  # Use cache (same type)
```

4. **Function name:**
```python
   # Name doesn't matter, only content
   @cute.kernel
   def name1(data): data[0] = 1.0
   
   @cute.kernel
   def name2(data): data[0] = 1.0
   # Same cache key if function bodies identical
```

---

## When Recompilation Occurs

### Triggers for Recompilation

**1. Code changes:**
```python
@cute.kernel
def my_kernel(data: cute.Tensor):
    data[0] = 1.0  # Version 1

my_kernel.launch(...)(data)  # Compile

# Edit function
@cute.kernel
def my_kernel(data: cute.Tensor):
    data[0] = 2.0  # Version 2

my_kernel.launch(...)(data)  # Recompile!
```

**2. Type changes:**
```python
tensor_fp32 = torch.randn(1024, dtype=torch.float32, device="cuda")
kernel.launch(...)(cute.from_dlpack(tensor_fp32))  # Compile

tensor_fp16 = torch.randn(1024, dtype=torch.float16, device="cuda")
kernel.launch(...)(cute.from_dlpack(tensor_fp16))  # Recompile!
```

**3. Constexpr value changes:**
```python
kernel.launch(...)(data, tile_size=128)  # Compile
kernel.launch(...)(data, tile_size=256)  # Recompile!
```

**4. Environment variable changes:**
```python
os.environ['CUTLASS_NVCC_ARCHS'] = '80'
kernel.launch(...)(data)  # Compile

os.environ['CUTLASS_NVCC_ARCHS'] = '90'
kernel.launch(...)(data)  # Recompile!
```

**5. Python session restart:**
```python
# First Python session
kernel.launch(...)(data)  # Compile

# Exit Python

# New Python session
kernel.launch(...)(data)  # Use cache (no recompile!)
```

---

## Precompilation with cute.compile()

### Eager Compilation

**Compile without executing:**
```python
@cute.kernel
def my_kernel(data: cute.Tensor, tile_size: cutlass.Constexpr):
    for i in cutlass.range_constexpr(tile_size):
        data[i] = 0.0

# Compile ahead of time
my_kernel.compile(
    data=cute.Tensor,  # Type annotation
    tile_size=128      # Constexpr value
)

# Later execution uses cache (no compilation delay)
my_kernel.launch(...)(data, tile_size=128)
```

### Warming Up Cache

**Precompile all variants:**
```python
@cute.jit
def warm_cache():
    """Compile all kernel variants ahead of time"""
    
    dummy_data = torch.zeros(1024, device="cuda")
    cute_data = cute.from_dlpack(dummy_data)
    
    # Compile all tile sizes we'll use
    for tile_size in [64, 128, 256]:
        print(f"Precompiling tile_size={tile_size}...")
        my_kernel.compile(data=cute_data, tile_size=tile_size)
    
    print("Cache warmed!")

# Run once at application startup
warm_cache()

# Now all kernel variants are cached
for size in [64, 128, 256]:
    my_kernel.launch(...)(data, tile_size=size)  # All fast!
```

### Build-Time Compilation

**Compile during package build:**
```python
# In setup.py or build script
import cutlass.cute as cute

@cute.kernel
def my_kernel(data: cute.Tensor):
    # Production kernel
    pass

# Force compilation
my_kernel.compile(data=cute.Tensor)

# Kernel now in cache for deployment
```

---

## Cache Management

### Clearing Cache

**Clear all cached kernels:**
```python
import shutil
import os

cache_dir = os.path.expanduser("~/.cache/cutlass")
shutil.rmtree(cache_dir)
print("Cache cleared")
```

**Clear specific kernel:**
```bash
# Find kernel hash
ls ~/.cache/cutlass/kernels/ | grep my_kernel

# Remove specific files
rm ~/.cache/cutlass/kernels/my_kernel_abc123.*
```

### Cache Size Management

**Automatic cache cleanup (if supported):**
```python
import os

# Set maximum cache size (in MB)
os.environ['CUTLASS_CACHE_MAX_SIZE'] = '1024'  # 1 GB

# Enable LRU eviction
os.environ['CUTLASS_CACHE_LRU'] = '1'
```

**Manual cache cleanup:**
```python
import os
import time
from pathlib import Path

def clean_old_cache(max_age_days=30):
    """Remove cache files older than max_age_days"""
    cache_dir = Path.home() / ".cache" / "cutlass" / "kernels"
    
    now = time.time()
    cutoff = now - (max_age_days * 86400)
    
    for file in cache_dir.glob("*"):
        if file.stat().st_mtime < cutoff:
            file.unlink()
            print(f"Removed old cache file: {file.name}")

clean_old_cache(30)
```

---

## Development Workflow

### Development Mode

**Disable caching during development:**
```python
import os

# Force recompilation every time (for debugging)
os.environ['CUTLASS_DISABLE_CACHE'] = '1'

@cute.kernel
def my_kernel(data: cute.Tensor):
    # Kernel under development
    pass

# Always recompiles (no cache used)
my_kernel.launch(...)(data)
```

**Or clear cache before each run:**
```python
import shutil
import os

# Clear cache at start of development session
cache_dir = os.path.expanduser("~/.cache/cutlass")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Development: cache cleared")
```

### Production Mode

**Enable aggressive caching:**
```python
import os

# Ensure caching enabled (default)
os.environ.pop('CUTLASS_DISABLE_CACHE', None)

# Precompile all variants
warm_cache()

# Use cached kernels for best performance
```

---

## Cache Persistence

### Across Python Sessions

**Cache persists:**
```python
# Session 1
@cute.kernel
def kernel(data: cute.Tensor):
    data[0] = 1.0

kernel.launch(...)(data)  # Compile, save to cache
# Exit Python

# Session 2 (later)
@cute.kernel
def kernel(data: cute.Tensor):
    data[0] = 1.0

kernel.launch(...)(data)  # Use cache (no compilation!)
```

### Across System Reboots

**Cache survives reboots:**
```bash
# Before reboot
python my_script.py  # Compiles and caches

# After reboot
python my_script.py  # Uses cache (fast!)
```

### Across CuTe DSL Updates

**Cache may be invalidated on updates:**
```bash
# Old version
pip install nvidia-cutlass-dsl==4.0.0
python my_script.py  # Compile

# Upgrade
pip install --upgrade nvidia-cutlass-dsl
python my_script.py  # May recompile if format changed
```

---

## Multi-GPU Caching

### Per-Device Caching

**Different GPUs may have different caches:**
```python
# Device 0 (A100 - SM80)
torch.cuda.set_device(0)
kernel.launch(...)(data_on_gpu0)  # Compile for SM80

# Device 1 (H100 - SM90)
torch.cuda.set_device(1)
kernel.launch(...)(data_on_gpu1)  # Compile for SM90 (different arch!)

# Two separate cached kernels
```

### Shared Cache

**Same architecture GPUs share cache:**
```python
# Both GPUs are A100s (SM80)
torch.cuda.set_device(0)
kernel.launch(...)(data0)  # Compile once

torch.cuda.set_device(1)
kernel.launch(...)(data1)  # Use cache (same architecture)
```

---

## Debugging Cache Issues

### Cache Misses

**Check if cache is being used:**
```python
import os
os.environ['CUTLASS_VERBOSE'] = '1'

@cute.kernel
def kernel(data: cute.Tensor):
    data[0] = 1.0

# Look for cache messages in output:
# "Cache hit: kernel_abc123"
# or
# "Cache miss: compiling..."
kernel.launch(...)(data)
```

### Cache Corruption

**If cache seems corrupted:**
```python
import shutil
import os

# Clear cache
cache_dir = os.path.expanduser("~/.cache/cutlass")
shutil.rmtree(cache_dir, ignore_errors=True)
print("Cache cleared due to corruption")

# Recompile fresh
kernel.launch(...)(data)
```

### Verification

**Verify cache contents:**
```python
from pathlib import Path
import hashlib

def verify_cache():
    """Check cache integrity"""
    cache_dir = Path.home() / ".cache" / "cutlass" / "kernels"
    
    for cubin in cache_dir.glob("*.cubin"):
        # Check file is readable and non-zero
        if cubin.stat().st_size == 0:
            print(f"Warning: Empty cache file {cubin}")
            cubin.unlink()

verify_cache()
```

---

## Best Practices

### ✅ DO

**Warm cache in production:**
```python
# At application startup
warm_cache()
```

**Use cache in benchmarking:**
```python
# Warm up before timing
for _ in range(10):
    kernel.launch(...)(data)

# Now benchmark
benchmark_kernel()
```

**Clear cache when changing CuTe DSL versions:**
```bash
pip install --upgrade nvidia-cutlass-dsl
rm -rf ~/.cache/cutlass
```

**Monitor cache size:**
```python
import os
from pathlib import Path

cache_size = sum(f.stat().st_size for f in 
                 Path.home().glob(".cache/cutlass/**/*") 
                 if f.is_file())
print(f"Cache size: {cache_size / 1024**2:.1f} MB")
```

### ❌ DON'T

**Don't disable cache in production:**
```python
# BAD: Slow startup
os.environ['CUTLASS_DISABLE_CACHE'] = '1'
```

**Don't assume cache works across versions:**
```python
# Cache may be incompatible after upgrade
```

**Don't include cache in version control:**
```bash
# .gitignore
.cache/
```

**Don't share cache across users:**
```bash
# Each user needs their own cache
# ~/.cache is user-specific
```

---

## Performance Impact

### First Call vs Cached

**Benchmark:**
```python
import time

@cute.kernel
def test_kernel(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    data[tid] = 0.0

data = torch.randn(1024, device="cuda")
cute_data = cute.from_dlpack(data)

# First call (compile + execute)
start = time.time()
test_kernel.launch(grid=[4,1,1], block=[256,1,1])(cute_data)
torch.cuda.synchronize()
first_call = time.time() - start

# Second call (cached)
start = time.time()
test_kernel.launch(grid=[4,1,1], block=[256,1,1])(cute_data)
torch.cuda.synchronize()
cached_call = time.time() - start

print(f"First call:  {first_call*1000:.1f} ms (compile + execute)")
print(f"Cached call: {cached_call*1000:.3f} ms (execute only)")
print(f"Speedup: {first_call/cached_call:.0f}×")

# Typical output:
# First call:  2347.3 ms (compile + execute)
# Cached call: 0.523 ms (execute only)
# Speedup: 4487×
```

---

## Summary

**JIT caching:**
- Compiled kernels saved to `~/.cache/cutlass/`
- Cache persists across Python sessions
- Recompilation only when code/types/Constexpr change

**Cache key based on:**
- Function source code
- Parameter types
- Constexpr values
- Compilation options
- Target architecture

**Cache management:**
- `cute.compile()` for precompilation
- Clear with `rm -rf ~/.cache/cutlass`
- Monitor size for production deployments

**Best practices:**
- Warm cache at startup
- Disable during development
- Clear after CuTe DSL updates
- Use cached kernels in benchmarks

**Performance:**
- First call: 1-10 seconds (compile + execute)
- Cached calls: <1 millisecond (execute only)
- 1000-10000× speedup from caching

---

## Next Steps

- [JIT Compilation](./jit_compilation.md) - How compilation works
- [Compilation Options](./compilation_options.md) - Tuning compilation
- [Code Generation](./code_generation.md) - What gets cached

---

## Further Reading

- [CUDA JIT Caching](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#just-in-time-compilation)
- [NVRTC Caching](https://docs.nvidia.com/cuda/nvrtc/index.html)