---
topic: "overview"
difficulty: "beginner"
related_topics: ["decorators", "quick_start"]
---

# CuTe DSL Overview

## What is CuTe DSL?

CuTe DSL is a Python-based domain-specific language (DSL) designed for dynamic compilation of numeric and GPU-oriented code. It provides a high-level, Pythonic interface for writing high-performance GPU kernels while maintaining the power and control of CUDA programming.

## Philosophy

CuTe DSL bridges the gap between Python's ease of use and GPU's performance requirements by:

- **Expressing GPU kernels with full hardware control** while using familiar Python syntax
- **Just-in-time (JIT) compilation** that generates optimized GPU code on-the-fly
- **Seamless framework integration** through DLPack protocol for PyTorch, JAX, etc.
- **Compile-time optimization** through optional static analysis and caching

## Primary Goals

### 1. Consistency with CuTe C++
CuTe DSL maintains compatibility with CuTe C++ abstractions, allowing users to leverage existing CuTe knowledge and patterns. The same layout algebra, tensor operations, and hardware abstractions are available in Python.

### 2. JIT Compilation for Host and Device
Both host-side orchestration and GPU kernel code are compiled just-in-time.

### 3. DLPack Integration
Seamless interoperability with major ML frameworks through zero-copy tensor conversion.

### 4. Intelligent Caching
JIT-compiled kernels are cached automatically, so repeated calls with the same configuration reuse compiled code without recompilation overhead.

### 5. Type Inference and Native Types
CuTe DSL provides native type support with automatic inference, reducing boilerplate.

### 6. Optional Lower-Level Control
When needed, you can access GPU-specific features directly for fine-grained control over hardware.

## Key Advantages

### For Python Users
- Write GPU kernels in familiar Python syntax
- No need to learn CUDA C++
- Tight integration with PyTorch/JAX workflows

### For CUDA Experts  
- Full control over memory hierarchies (GMEM, SMEM, TMEM, registers)
- Access to architecture-specific instructions (Tensor Cores, TMA, etc.)
- Explicit tiling and layout control

### For Performance Engineers
- Compile-time specialization for different input sizes
- Software pipelining with minimal code
- Auto-tuning friendly design

## When to Use CuTe DSL

**Good fit:**
- Custom GPU kernels for ML operations
- High-performance numeric computation
- Operations not well-served by existing libraries
- Rapid prototyping of GPU algorithms
- Research into novel GPU optimization techniques

**Less ideal:**
- Simple element-wise operations (use PyTorch/JAX)
- Operations already optimized in cuBLAS/cuDNN
- Applications without GPU acceleration needs

## Architecture Support

CuTe DSL supports modern NVIDIA GPU architectures:

- **Ampere (SM80)**: A100, RTX 3090
- **Hopper (SM90)**: H100, H200
- **Blackwell (SM100)**: B100, B200 (latest features)

Each architecture offers progressively more advanced features:
- Ampere: Tensor Cores, async copy
- Hopper: TMA (Tensor Memory Accelerator), warp specialization
- Blackwell: 2-CTA instructions, TMEM, Programmatic Dependent Launch (PDL)

## Comparison with Alternatives

| Feature | CuTe DSL | Triton | Raw CUDA |
|---------|----------|--------|----------|
| Language | Python | Python | C++ |
| Learning Curve | Medium | Medium | Steep |
| Control Level | High | Medium | Highest |
| CuTe Compatibility | Native | None | Via C++ |
| Framework Integration | Excellent | Excellent | Manual |
| Debugging | Python + CUDA tools | Python + Triton tools | CUDA tools |

## Next Steps
- [Decorators](./decorators.md) - Learn about @jit and @kernel
- [Quick Start](./quick_start.md) - Write your first CuTe DSL kernel
- [Limitations](./limitations.md) - Understand what CuTe DSL cannot do

## Further Reading
- [CuTe C++ Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [CUTLASS Python Documentation](https://github.com/NVIDIA/cutlass/tree/main/python)