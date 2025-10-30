---
topic: "quick_start"
difficulty: "beginner"
related_topics: ["overview", "decorators"]
---

# Quick Start Guide for Developers

## Prerequisites
Install CUTLASS Python which includes CuTe DSL:
pip install nvidia-cutlass

**Requirements:**
- NVIDIA GPU (Ampere/Hopper/Blackwell recommended)
- CUDA 11.8+
- Python 3.8+
- PyTorch or JAX (for framework integration)

---

## Hello World: Vector Addition

Let's write your first CuTe DSL kernel - a simple vector addition.

### Step 1: Import Libraries

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

### Step 2: Define the Kernel

Define a GPU kernel that adds two vectors element-wise. Each thread processes one element using thread and block indices to calculate the global position.

### Step 3: Create Host Wrapper

Create a host function that calculates grid dimensions and launches the vector addition kernel with appropriate configuration.

### Step 4: Run from Python

Create PyTorch tensors, convert them to CuTe tensors using DLPack (zero-copy), run the kernel, and verify results.

### Complete Example

The complete working example includes the kernel definition, host wrapper, tensor creation, kernel execution, and result verification.

---

## Example 2: Matrix Transpose

A slightly more complex example demonstrating 2D indexing.

Define a transpose kernel using 2D thread blocks for efficiency. Use 2D thread and block indices to calculate global row and column positions. Launch with 16x16 thread blocks in a 2D grid configuration.

---

## Example 3: Element-wise Operation with Dynamic Arguments

Shows how to pass runtime parameters to kernels.

Define a kernel that takes scalar parameters (scale and bias) at runtime. The kernel performs: output = scale * x + y + bias

---

## Example 4: Using Compile-Time Constants

Demonstrates the power of cutlass.Constexpr for specialization.

Define a kernel with a compile-time flag to choose between ReLU and GELU activation. The unused branch is completely eliminated from compiled code, resulting in different specialized kernels for each activation type.

---

## Common Patterns

### Pattern 1: Bounds Checking

Always check if your thread index is within array bounds before accessing memory.

### Pattern 2: Grid-Stride Loop

For arrays larger than your grid size, use a grid-stride loop to process all elements.

### Pattern 3: 2D/3D Indexing

For multi-dimensional data, calculate row and column indices from 2D block and thread indices.

---

## Debugging Tips

### Enable Logging

Set environment variables to enable console logging and control verbosity:
export CUTE_DSL_LOG_TO_CONSOLE=1
export CUTE_DSL_LOG_LEVEL=10

### Print from Kernels

Use cute.printf() for runtime debugging (use sparingly as it impacts performance).

### Verify with PyTorch

Always compare results against a reference implementation using torch.testing.assert_close().

---

## Performance Tips

1. **Choose appropriate block size**: 128-256 threads is typical
2. **Minimize divergence**: Avoid if statements within warps when possible
3. **Coalesce memory access**: Adjacent threads should access adjacent memory
4. **Use shared memory** for data reuse (coming in advanced tutorials)
5. **Profile first**: Use nsys or ncu to find bottlenecks

---

## Next Steps

Now that you've written basic kernels:

1. **Learn about layouts** - [Layouts and Tensors](../01_fundamentals/layouts.md)
2. **Explore control flow** - [Loops and Conditionals](../02_control_flow/loops.md)
3. **See real examples** - [GitHub Examples](../10_examples/github_examples.md)
4. **Optimize performance** - [Tiling Strategies](../08_optimization/tiling_strategies.md)

---

## Common Errors and Solutions

### Error: "expects argument #1 to be Tensor"
**Solution**: Convert PyTorch tensors with from_dlpack() first

### Error: "kernel launch failed: invalid configuration"
**Solution**: Check that grid/block dimensions are valid (not zero, not too large)

### Error: "Changing type of a variable not allowed"
**Solution**: Don't reassign variables to different types in kernels

### Results don't match reference
**Solution**: Add bounds checking, verify indexing logic, check for race conditions