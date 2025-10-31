---
topic: "compilation_options"
difficulty: "intermediate"
related_topics: ["jit_compilation", "jit_caching", "code_generation"]
---

# Compilation Options in CuTe DSL

## Overview

CuTe DSL provides various compilation options to control code generation, optimization level, debugging support, and target architecture. Understanding these options is essential for development, debugging, and production deployment.

**Key concepts:**
- Environment variables for compilation control
- Optimization levels
- Debug vs release builds
- Architecture targeting
- Compiler flags
- Assertion control

---

## Setting Compilation Options

### Environment Variables

**Primary method for controlling compilation:**
```python
import os

# Set options BEFORE importing cutlass
os.environ['CUTLASS_DEBUG'] = '1'
os.environ['CUTLASS_NVCC_ARCHS'] = '80'

import cutlass
import cutlass.cute as cute

@cute.kernel
def my_kernel(data: cute.Tensor):
    # Compiled with debug mode, targeting SM80
    pass
```

**Important:** Most options must be set before importing `cutlass`.

---

## Optimization Levels

### CUTLASS_OPT_LEVEL

**Control optimization level:**
```python
import os

# No optimization (fastest compile, slowest execution)
os.environ['CUTLASS_OPT_LEVEL'] = '0'

# Basic optimization
os.environ['CUTLASS_OPT_LEVEL'] = '1'

# Moderate optimization (default)
os.environ['CUTLASS_OPT_LEVEL'] = '2'

# Aggressive optimization (slowest compile, fastest execution)
os.environ['CUTLASS_OPT_LEVEL'] = '3'

import cutlass
```

**Trade-offs:**

| Level | Compile Time | Runtime Speed | Use When |
|-------|--------------|---------------|----------|
| -O0   | Fast (1×)    | Slow (1×)     | Development, debugging |
| -O1   | Medium (2×)  | Medium (3×)   | Quick testing |
| -O2   | Medium (3×)  | Fast (8×)     | Default, balanced |
| -O3   | Slow (4-5×)  | Fastest (10×) | Production |

### Example: Development vs Production

**Development:**
```python
import os
os.environ['CUTLASS_OPT_LEVEL'] = '0'  # Fast compile for iteration
os.environ['CUTLASS_DEBUG'] = '1'       # Enable debugging

import cutlass.cute as cute

@cute.kernel
def dev_kernel(data: cute.Tensor):
    # Quick compile, easy to debug
    data[0] = 1.0
```

**Production:**
```python
import os
os.environ['CUTLASS_OPT_LEVEL'] = '3'  # Maximum optimization
os.environ['CUTLASS_DEBUG'] = '0'       # Disable debugging overhead

import cutlass.cute as cute

@cute.kernel
def prod_kernel(data: cute.Tensor):
    # Slower compile, maximum performance
    data[0] = 1.0
```

---

## Debug Mode

### CUTLASS_DEBUG

**Enable debug symbols and checks:**
```python
import os

# Debug mode (default in development)
os.environ['CUTLASS_DEBUG'] = '1'

# Release mode (default in production)
os.environ['CUTLASS_DEBUG'] = '0'

import cutlass
```

**Debug mode enables:**
- Line number information
- Bounds checking (if available)
- Assertion checks
- More verbose error messages
- Symbol names in profiling

**Debug mode disables:**
- Aggressive optimizations
- Function inlining
- Dead code elimination

### Example: Debug Workflow
```python
import os
os.environ['CUTLASS_DEBUG'] = '1'
os.environ['CUTLASS_VERBOSE'] = '1'

import cutlass.cute as cute

@cute.kernel
def debug_kernel(data: cute.Tensor, n: cutlass.Int32):
    tid = cute.arch.thread_idx()[0]
    
    # In debug mode, better error messages
    if tid < n:
        data[tid] = 1.0 / tid  # Division by zero caught with line number!

# Error message includes:
# - File name
# - Line number
# - Function name
# - Helpful context
```

---

## Architecture Targeting

### CUTLASS_NVCC_ARCHS

**Specify target GPU architecture:**
```python
import os

# Ampere (A100, RTX 3090)
os.environ['CUTLASS_NVCC_ARCHS'] = '80'

# Hopper (H100)
os.environ['CUTLASS_NVCC_ARCHS'] = '90'

# Blackwell (B100, B200)
os.environ['CUTLASS_NVCC_ARCHS'] = '100'

# Multiple architectures (fat binary)
os.environ['CUTLASS_NVCC_ARCHS'] = '80;90;100'

import cutlass
```

**Architecture codes:**

| Architecture | Compute Capability | NVCC Arch | Example GPUs |
|--------------|-------------------|-----------|--------------|
| Volta        | 7.0, 7.2         | 70, 72    | V100 |
| Turing       | 7.5              | 75        | RTX 2080, T4 |
| Ampere       | 8.0, 8.6         | 80, 86    | A100, RTX 3090 |
| Ada          | 8.9              | 89        | RTX 4090, L40 |
| Hopper       | 9.0              | 90        | H100 |
| Blackwell    | 10.0             | 100       | B100, B200 |

### Auto-Detection

**Let CuTe detect current GPU:**
```python
import os

# Auto-detect (default)
os.environ.pop('CUTLASS_NVCC_ARCHS', None)

import cutlass
# Automatically targets current GPU architecture
```

### Multi-GPU Support

**Compile for multiple architectures:**
```python
import os

# Fat binary supporting multiple GPUs
os.environ['CUTLASS_NVCC_ARCHS'] = '80;90'  # A100 and H100

import cutlass.cute as cute

@cute.kernel
def multi_gpu_kernel(data: cute.Tensor):
    # Single kernel works on both A100 and H100
    data[0] = 1.0

# Larger binary, but portable
```

---

## Verbosity Control

### CUTLASS_VERBOSE

**Control logging verbosity:**
```python
import os

# Silent (no output)
os.environ['CUTLASS_VERBOSE'] = '0'

# Basic info (default)
os.environ['CUTLASS_VERBOSE'] = '1'

# Detailed compilation info
os.environ['CUTLASS_VERBOSE'] = '2'

# Everything (very noisy)
os.environ['CUTLASS_VERBOSE'] = '3'

import cutlass
```

**Output at different levels:**

**Level 0 (silent):**
```
# No output
```

**Level 1 (basic):**
```
Compiling kernel 'my_kernel'...
Compilation complete (2.3s)
```

**Level 2 (detailed):**
```
Compiling kernel 'my_kernel'...
- Parsing AST
- Generating MLIR
- Optimizing MLIR
- Generating CUDA C++
- Invoking NVCC
- Compilation complete (2.3s)
```

**Level 3 (debug):**
```
Compiling kernel 'my_kernel'...
- Parsing AST
  - Found 3 parameters
  - Return type: None
- Generating MLIR
  - 47 operations
  - 12 basic blocks
- Optimizing MLIR
  - Pass 1: Constant folding
  - Pass 2: Dead code elimination
  - Pass 3: Loop unrolling
- Generating CUDA C++
  - 123 lines
  - 45 unique tensor accesses
- Invoking NVCC
  - Target: sm_80
  - Optimization: -O2
- Compilation complete (2.3s)
- Generated code at: ~/.cache/cutlass/kernels/my_kernel_abc.cu
```

---

## Code Generation Options

### CUTLASS_DUMP_CODE

**Save generated CUDA C++ code:**
```python
import os
os.environ['CUTLASS_DUMP_CODE'] = '1'

import cutlass.cute as cute

@cute.kernel
def my_kernel(data: cute.Tensor):
    data[0] = 1.0

# Compilation prints:
# "Generated code saved to: ~/.cache/cutlass/kernels/my_kernel_abc123.cu"
```

**Use case:** Inspect generated code for optimization.

### CUTLASS_KEEP_PTX

**Keep PTX assembly:**
```python
import os
os.environ['CUTLASS_KEEP_PTX'] = '1'

import cutlass.cute as cute

@cute.kernel
def my_kernel(data: cute.Tensor):
    data[0] = 1.0

# PTX saved alongside CUBIN
# ~/.cache/cutlass/kernels/my_kernel_abc123.ptx
```

**Use case:** Analyze low-level code generation, instruction counts.

### CUTLASS_KEEP_ALL

**Keep all intermediate files:**
```python
import os
os.environ['CUTLASS_KEEP_ALL'] = '1'

import cutlass

# Keeps:
# - .cu (CUDA C++)
# - .ptx (PTX assembly)
# - .cubin (binary)
# - .mlir (MLIR IR)
```

---

## Assertion Control

### CUTLASS_ENABLE_ASSERTIONS

**Enable runtime assertions:**
```python
import os

# Enable assertions (default in debug)
os.environ['CUTLASS_ENABLE_ASSERTIONS'] = '1'

# Disable assertions (faster, but less safe)
os.environ['CUTLASS_ENABLE_ASSERTIONS'] = '0'

import cutlass.cute as cute

@cute.kernel
def kernel_with_assert(data: cute.Tensor, n: cutlass.Int32):
    tid = cute.arch.thread_idx()[0]
    
    # Assertion checked at runtime (if enabled)
    assert tid < n, "Thread index out of bounds"
    
    data[tid] = 1.0
```

**Assertions enabled:**
```
# If assertion fails:
AssertionError: Thread index out of bounds
  File "my_kernel.py", line 8, in kernel_with_assert
  tid=256, n=100
```

**Assertions disabled:**
```
# If assertion fails:
# Undefined behavior (may crash or corrupt data)
```

---

## Cache Control

### CUTLASS_DISABLE_CACHE

**Disable JIT cache:**
```python
import os

# Force recompilation every time
os.environ['CUTLASS_DISABLE_CACHE'] = '1'

import cutlass.cute as cute

@cute.kernel
def always_compile(data: cute.Tensor):
    data[0] = 1.0

# Always recompiles (slow, useful for debugging)
```

### CUTLASS_CACHE_DIR

**Custom cache location:**
```python
import os

# Use custom cache directory
os.environ['CUTLASS_CACHE_DIR'] = '/tmp/my_cutlass_cache'

import cutlass

# Kernels cached in /tmp/my_cutlass_cache
```

---

## NVCC Flags

### CUTLASS_NVCC_FLAGS

**Pass additional flags to NVCC:**
```python
import os

# Add custom NVCC flags
os.environ['CUTLASS_NVCC_FLAGS'] = '--use_fast_math --maxrregcount=64'

import cutlass.cute as cute

@cute.kernel
def kernel_with_flags(data: cute.Tensor):
    # Compiled with fast math and register limit
    data[0] = 1.0
```

**Common flags:**

| Flag | Description | Use When |
|------|-------------|----------|
| `--use_fast_math` | Fast (less accurate) math | Performance > precision |
| `--ftz=true` | Flush denormals to zero | Extra performance |
| `--prec-div=false` | Fast division | Don't need full precision |
| `--prec-sqrt=false` | Fast square root | Don't need full precision |
| `--maxrregcount=N` | Limit registers per thread | Increase occupancy |
| `--fmad=true` | Use fused multiply-add | Better precision + speed |

### Example: Maximum Performance
```python
import os

os.environ['CUTLASS_NVCC_FLAGS'] = ' '.join([
    '--use_fast_math',
    '--ftz=true',
    '--prec-div=false',
    '--prec-sqrt=false',
    '--fmad=true'
])

import cutlass
```

---

## Warning Control

### CUTLASS_WARNINGS

**Control warning verbosity:**
```python
import os

# Show all warnings (default)
os.environ['CUTLASS_WARNINGS'] = 'all'

# Show only errors
os.environ['CUTLASS_WARNINGS'] = 'error'

# Suppress all warnings
os.environ['CUTLASS_WARNINGS'] = 'none'

import cutlass
```

---

## Profiling and Analysis

### CUTLASS_ENABLE_PROFILING

**Enable profiling hooks:**
```python
import os
os.environ['CUTLASS_ENABLE_PROFILING'] = '1'

import cutlass.cute as cute

@cute.kernel
def profiled_kernel(data: cute.Tensor):
    data[0] = 1.0

# Kernel includes profiling markers
# Can be analyzed with Nsight Compute/Systems
```

### CUTLASS_PRINT_STATISTICS

**Print compilation statistics:**
```python
import os
os.environ['CUTLASS_PRINT_STATISTICS'] = '1'

import cutlass.cute as cute

@cute.kernel
def kernel_with_stats(data: cute.Tensor):
    data[0] = 1.0

# Prints:
# Compilation Statistics:
# - MLIR operations: 47
# - CUDA lines: 123
# - Estimated registers: 32
# - Estimated shared memory: 0 bytes
```

---

## Configuration File

### .cutlassrc

**Store options in config file:**
```bash
# ~/.cutlassrc or ./.cutlassrc
CUTLASS_OPT_LEVEL=3
CUTLASS_DEBUG=0
CUTLASS_NVCC_ARCHS=80;90
CUTLASS_VERBOSE=1
```

**Load from file (if supported):**
```python
import cutlass
# Automatically reads ~/.cutlassrc
```

---

## Common Configuration Patterns

### Pattern 1: Development
```python
import os

# Fast iteration, easy debugging
os.environ['CUTLASS_OPT_LEVEL'] = '0'
os.environ['CUTLASS_DEBUG'] = '1'
os.environ['CUTLASS_VERBOSE'] = '2'
os.environ['CUTLASS_ENABLE_ASSERTIONS'] = '1'
os.environ['CUTLASS_DUMP_CODE'] = '1'

import cutlass.cute as cute
```

### Pattern 2: Production
```python
import os

# Maximum performance, minimal overhead
os.environ['CUTLASS_OPT_LEVEL'] = '3'
os.environ['CUTLASS_DEBUG'] = '0'
os.environ['CUTLASS_VERBOSE'] = '0'
os.environ['CUTLASS_ENABLE_ASSERTIONS'] = '0'
os.environ['CUTLASS_NVCC_FLAGS'] = '--use_fast_math'

import cutlass.cute as cute
```

### Pattern 3: Benchmarking
```python
import os

# Accurate timing, profiling enabled
os.environ['CUTLASS_OPT_LEVEL'] = '3'
os.environ['CUTLASS_DEBUG'] = '0'
os.environ['CUTLASS_ENABLE_PROFILING'] = '1'
os.environ['CUTLASS_VERBOSE'] = '0'

import cutlass.cute as cute
```

### Pattern 4: Debugging Performance
```python
import os

# Inspect generated code, understand optimization
os.environ['CUTLASS_OPT_LEVEL'] = '3'
os.environ['CUTLASS_DUMP_CODE'] = '1'
os.environ['CUTLASS_KEEP_PTX'] = '1'
os.environ['CUTLASS_VERBOSE'] = '2'
os.environ['CUTLASS_PRINT_STATISTICS'] = '1'

import cutlass.cute as cute
```

---

## Platform-Specific Options

### Linux/Mac
```bash
# Set environment variables in shell
export CUTLASS_OPT_LEVEL=3
export CUTLASS_NVCC_ARCHS=90
python my_script.py
```

### Windows
```cmd
REM Set environment variables in cmd
set CUTLASS_OPT_LEVEL=3
set CUTLASS_NVCC_ARCHS=90
python my_script.py
```
```powershell
# PowerShell
$env:CUTLASS_OPT_LEVEL = "3"
$env:CUTLASS_NVCC_ARCHS = "90"
python my_script.py
```

---

## Runtime Option Changes

### Changing Options After Import

**Most options are read at import time:**
```python
import cutlass.cute as cute

# This has NO effect (too late)
os.environ['CUTLASS_OPT_LEVEL'] = '3'

@cute.kernel
def kernel(data: cute.Tensor):
    pass
# Still uses default optimization level
```

**Workaround: Use subprocess:**
```python
import subprocess
import os

# Set environment for subprocess
env = os.environ.copy()
env['CUTLASS_OPT_LEVEL'] = '3'

subprocess.run(['python', 'my_kernel_script.py'], env=env)
```

---

## Checking Current Options

### Print Active Configuration
```python
import os
import cutlass

def print_config():
    """Print current CuTe compilation configuration"""
    
    options = {
        'CUTLASS_OPT_LEVEL': os.environ.get('CUTLASS_OPT_LEVEL', 'default'),
        'CUTLASS_DEBUG': os.environ.get('CUTLASS_DEBUG', 'default'),
        'CUTLASS_NVCC_ARCHS': os.environ.get('CUTLASS_NVCC_ARCHS', 'auto'),
        'CUTLASS_VERBOSE': os.environ.get('CUTLASS_VERBOSE', 'default'),
        'CUTLASS_DISABLE_CACHE': os.environ.get('CUTLASS_DISABLE_CACHE', '0'),
    }
    
    print("CuTe Compilation Configuration:")
    for key, value in options.items():
        print(f"  {key}: {value}")

print_config()
```

---

## Best Practices

### ✅ DO

**Set options before import:**
```python
import os
os.environ['CUTLASS_OPT_LEVEL'] = '3'
import cutlass  # Now options take effect
```

**Use different configs for dev/prod:**
```python
if os.environ.get('ENV') == 'production':
    os.environ['CUTLASS_OPT_LEVEL'] = '3'
else:
    os.environ['CUTLASS_OPT_LEVEL'] = '0'
```

**Document your configuration:**
```python
# Production configuration
# - Optimization: Maximum (-O3)
# - Debug: Disabled
# - Target: Hopper (H100)
os.environ['CUTLASS_OPT_LEVEL'] = '3'
os.environ['CUTLASS_DEBUG'] = '0'
os.environ['CUTLASS_NVCC_ARCHS'] = '90'
```

**Use verbose mode when debugging:**
```python
os.environ['CUTLASS_VERBOSE'] = '2'
```

### ❌ DON'T

**Don't set options after import:**
```python
import cutlass
os.environ['CUTLASS_OPT_LEVEL'] = '3'  # Too late!
```

**Don't use -O0 in production:**
```python
# BAD: 10× slower
os.environ['CUTLASS_OPT_LEVEL'] = '0'
```

**Don't disable assertions in development:**
```python
# BAD: Harder to debug
os.environ['CUTLASS_ENABLE_ASSERTIONS'] = '0'
```

**Don't target wrong architecture:**
```python
# BAD: Won't run on H100
os.environ['CUTLASS_NVCC_ARCHS'] = '80'  # A100 only
# Kernel will fail on H100
```

---

## Summary

**Key compilation options:**
- `CUTLASS_OPT_LEVEL`: Optimization level (0-3)
- `CUTLASS_DEBUG`: Debug mode (0/1)
- `CUTLASS_NVCC_ARCHS`: Target GPU architecture
- `CUTLASS_VERBOSE`: Logging verbosity
- `CUTLASS_DUMP_CODE`: Save generated code

**Configuration patterns:**
- Development: -O0, debug=1, verbose=2
- Production: -O3, debug=0, verbose=0
- Benchmarking: -O3, profiling=1
- Debugging: -O3, dump_code=1, keep_ptx=1

**Best practices:**
- Set options before importing cutlass
- Use -O3 for production
- Enable debug mode for development
- Target correct architecture
- Document your configuration

---

## Next Steps

- [JIT Compilation](./jit_compilation.md) - How compilation works
- [JIT Caching](./jit_caching.md) - Cache behavior
- [Type Safety](./type_safety.md) - Type annotations and validation

---

## Further Reading

- [NVCC Compiler Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU Architecture Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)