---
topic: "tensors"
difficulty: "intermediate"
related_topics: ["layouts", "data_types", "memory_hierarchy"]
---

# CuTe Tensors

## Overview

In CuTe DSL, a **tensor** is a view of memory with an associated **layout** that describes how logical coordinates map to physical memory addresses. Tensors are the primary abstraction for working with data on the GPU.

**Key concept:** A tensor = pointer + layout + metadata

Unlike raw pointers, tensors carry structural information that enables:
- Automatic bounds checking (in debug mode)
- Compile-time address optimization
- Type-safe operations
- Composable transformations

---

## Tensor Structure

Every CuTe tensor contains:

1. **Data pointer**: Memory address (GMEM, SMEM, TMEM, or registers)
2. **Layout**: Shape and stride information
3. **Element type**: Data type (Float16, Float32, Int32, etc.)
4. **Memory space**: Where the data lives (global, shared, register, etc.)

### Tensor Anatomy

tensor = cute.make_tensor(ptr, layout)

# Components:
# - tensor.data_ptr     -> Memory address
# - tensor.layout       -> Shape and stride
# - tensor.shape        -> Just the shape part
# - tensor.stride       -> Just the stride part
# - tensor.element_type -> Data type
# - tensor.memspace     -> Memory space (gmem, smem, etc.)

---

## Creating Tensors

### From Pointer and Layout

# Most common: explicit pointer and layout
ptr = cute.ptr(address, element_type, memspace)
layout = cute.make_layout(shape, stride)
tensor = cute.make_tensor(ptr, layout)

### From PyTorch/Framework Tensors (via DLPack)

import torch
from cutlass.cute.runtime import from_dlpack

# Create PyTorch tensor
torch_tensor = torch.randn(128, 64, device="cuda")

# Convert to CuTe tensor (zero-copy)
cute_tensor = from_dlpack(torch_tensor)

# Now you can use cute_tensor in CuTe DSL functions

**Important:** The CuTe tensor shares memory with the PyTorch tensor. Changes to one affect the other.

### Allocating New Tensors

Within a @cute.jit host function, you can allocate new tensors:

@cute.jit
def allocate_example():
    # Allocate a new tensor
    shape = cute.make_shape(128, 64)
    tensor = cute.make_tensor(shape, cutlass.Float32)
    
    # Tensor is allocated in appropriate memory space
    return tensor

**Note:** You cannot allocate new tensors inside @cute.kernel functions (GPU kernels cannot do dynamic allocation).

---

## Tensor Operations

### Indexing and Slicing

# 1D indexing
value = tensor[i]

# 2D indexing  
value = tensor[i, j]

# 3D indexing
value = tensor[i, j, k]

# Slicing (creates a view)
subtensor = tensor[start:end]
subtensor = tensor[start:end, :]

# Using coordinates
coord = cute.make_coord(i, j)
value = tensor[coord]

### Shape and Size

# Get shape
shape = tensor.shape  # Returns (dim0, dim1, ...)

# Get size of specific dimension
dim0_size = cute.size(tensor, mode=0)
dim1_size = cute.size(tensor, mode=1)

# Get total number of elements
total_size = cute.size(tensor)

# Get rank (number of dimensions)
rank = cute.rank(tensor)

### Reshaping and Layout Changes

# Reshape (creates new view with different layout)
reshaped = cute.reshape(tensor, new_shape)

# Flatten
flattened = cute.flatten(tensor)

# Transpose
transposed = cute.transpose(tensor, dim0, dim1)

---

## Memory Spaces

Tensors can live in different memory spaces, each with different characteristics.

### Global Memory (GMEM)

**Largest capacity**, **slowest access**, accessible by all threads.

# Tensor in global memory
gmem_tensor = cute.make_tensor(
    cute.ptr(address, dtype, cute.AddressSpace.gmem),
    layout
)

**Typical use:** Input/output data, large intermediate results

### Shared Memory (SMEM)

**Medium capacity** (48-256KB per block), **fast access**, shared within thread block.

# Tensor in shared memory
smem_tensor = cute.make_tensor(
    cute.ptr(address, dtype, cute.AddressSpace.smem),
    layout
)

**Typical use:** Tile data for reuse within a thread block, staging for GMEM ↔ registers

**Important:** Shared memory requires careful layout design to avoid bank conflicts.

### Register Memory (RMEM)

**Smallest capacity**, **fastest access**, private to each thread.

# Allocate register tensor (fragment)
rmem_tensor = cute.make_rmem_tensor(shape, dtype)

**Typical use:** Thread-private accumulation, temporary computation

### Tensor Memory (TMEM) - Blackwell

**Architecture-specific** memory on Blackwell GPUs, very fast for tensor core operations.

# TMEM tensor (Blackwell only)
tmem_ptr = allocate_tmem(...)
tmem_tensor = cute.make_tensor(tmem_ptr, layout)

**Typical use:** Accumulator storage for large MMA operations

---

## Tensor Partitioning

Partitioning divides a tensor among threads or thread blocks.

### Thread-Level Partitioning

# Partition a tensor among threads in a block
tiled_copy = cute.make_tiled_copy(copy_atom, ...)
thread_tensor = tiled_copy.partition_S(source_tensor)

# Each thread gets its own view
thread_data = thread_tensor[thread_id]

### Block-Level Partitioning

# Divide tensor into tiles for thread blocks
tile_shape = cute.make_shape(128, 64)
tiled = cute.zipped_divide(tensor, tile_shape)

# Each block gets one tile
block_tile = tiled[block_idx_m, block_idx_n]

### Example: GEMM Partitioning

# Global tensors
gA = cute.make_tensor(ptr_A, layout_A)  # M×K
gB = cute.make_tensor(ptr_B, layout_B)  # K×N
gC = cute.make_tensor(ptr_C, layout_C)  # M×N

# Partition into tiles for thread blocks
tile_shape = cute.make_shape(128, 128, 32)  # Tile M×N×K
tiled_A = cute.zipped_divide(gA, tile_shape[:2] + (tile_shape[2],))
tiled_B = cute.zipped_divide(gB, (tile_shape[2],) + tile_shape[1:2])
tiled_C = cute.zipped_divide(gC, tile_shape[:2])

# Get this block's tile
bidx, bidy = cute.arch.block_idx()[:2]
block_A = tiled_A[bidx, :]
block_B = tiled_B[:, bidy]
block_C = tiled_C[bidx, bidy]

# Further partition among threads...
thr_mma = tiled_mma.get_slice(thread_idx)
thr_A = thr_mma.partition_A(block_A)
thr_B = thr_mma.partition_B(block_B)
thr_C = thr_mma.partition_C(block_C)

---

## Data Movement Between Memory Spaces

### Copying Data

# Copy from global memory to shared memory
cute.copy(tma_atom, gmem_tensor, smem_tensor)

# Copy from shared memory to registers
cute.copy(copy_atom, smem_tensor, rmem_tensor)

# Copy from registers to global memory
cute.copy(copy_atom, rmem_tensor, gmem_tensor)

### Synchronization

After writing to shared memory, synchronize before reading:

# Write to shared memory
cute.copy(copy_atom, gmem_src, smem_dst)

# Synchronize threads in block
cute.arch.syncthreads()

# Now safe to read from shared memory
cute.copy(copy_atom, smem_dst, rmem_dst)

---

## Tensor Fragments

**Fragments** are register-resident tensors optimized for specific operations.

### Creating Fragments

# Fragment compatible with MMA operation
tiled_mma = cute.make_tiled_mma(mma_atom)
fragment_A = tiled_mma.make_fragment_A(smem_layout_A)
fragment_B = tiled_mma.make_fragment_B(smem_layout_B)
fragment_C = tiled_mma.make_fragment_C(shape_C)

# Generic register fragment
fragment = cute.make_rmem_tensor(shape, dtype)

### Loading and Storing Fragments

# Load from tensor to fragment
values = fragment.load()  # Returns actual values

# Store to fragment
fragment.store(values)

# Element-wise access
fragment[i] = value

---

## Tensor Attributes and Metadata

### Querying Tensor Properties

# Shape and stride
print(f"Shape: {tensor.shape}")
print(f"Stride: {tensor.stride}")

# Element type and size
print(f"Element type: {tensor.element_type}")
print(f"Element size: {cute.size_in_bytes(tensor.element_type, 1)} bytes")

# Memory space
print(f"Memory space: {tensor.memspace}")

# Total size
print(f"Total elements: {cute.size(tensor)}")
print(f"Total bytes: {cute.size_in_bytes(tensor.element_type, cute.size(tensor))}")

# Layout details
print(f"Layout: {tensor.layout}")

---

## Working with Dynamic Tensors

Tensors can have **dynamic layouts** where shape/stride are only known at runtime.

### Static vs Dynamic

# Static tensor (shape known at compile time)
static_tensor = from_dlpack(torch_tensor)  # Shape is (M, N)

# Dynamic tensor (shape known at runtime)
dynamic_tensor = from_dlpack(torch_tensor).mark_layout_dynamic()
# Shape is (?, ?)

**Why use dynamic:** Compile one kernel for multiple shapes, reducing compilation overhead.

**Trade-off:** Dynamic layouts may be slightly slower due to runtime address calculation.

---

## Tensor Alignment

Alignment affects performance and some operations require specific alignment.

### Specifying Alignment

# Assume 128-byte alignment
tensor = from_dlpack(torch_tensor, assumed_align=128)

# Use with cute.assume for dynamic shapes
@cute.jit
def example(tensor: cute.Tensor):
    # Assume size is divisible by 16
    size = cute.assume(cute.size(tensor), divby=16)

**Why alignment matters:**
- Vectorized loads/stores require aligned addresses
- Tensor core operations need aligned data
- TMA (Hopper+) requires specific alignment

---

## Common Patterns

### Pattern 1: Load → Compute → Store

# Load tile from global to shared
cute.copy(tma_atom, gmem_tensor[tile_idx], smem_tensor)

# Synchronize
cute.arch.syncthreads()

# Load from shared to registers
cute.copy(copy_atom, smem_tensor, rmem_tensor)

# Compute
result = process(rmem_tensor.load())

# Store back to shared
rmem_result.store(result)
cute.copy(copy_atom, rmem_result, smem_tensor)

# Synchronize
cute.arch.syncthreads()

# Store shared to global
cute.copy(copy_atom, smem_tensor, gmem_tensor[tile_idx])

### Pattern 2: Double Buffering

# Two shared memory buffers for pipelining
smem_buffers = [smem_tensor0, smem_tensor1]

for k_tile in range(num_k_tiles):
    buffer_idx = k_tile % 2
    
    # Load next tile while computing current tile
    cute.copy(tma_atom, gmem[k_tile + 1], smem_buffers[1 - buffer_idx])
    
    # Compute with current tile
    compute(smem_buffers[buffer_idx])
    
    # Sync before swapping buffers
    cute.arch.syncthreads()

### Pattern 3: Reduction

# Thread-local accumulation
accumulator = cute.make_rmem_tensor(cute.make_shape(1), cutlass.Float32)
accumulator.store(0.0)

# Each thread accumulates its portion
for i in range(thread_elements):
    val = input_tensor[thread_start + i]
    accumulator.store(accumulator.load() + val)

# Block-level reduction (requires shared memory + synchronization)
# ... reduction tree in shared memory ...

---

## Best Practices

### ✅ DO

- **Use from_dlpack** for framework integration (zero-copy)
- **Partition tensors** hierarchically (block level, then thread level)
- **Minimize GMEM access** by staging through SMEM
- **Align tensors** for vectorized operations
- **Use fragments** for MMA operations
- **Match layouts** to hardware requirements (Tensor Cores, TMA, etc.)

### ❌ DON'T

- **Don't allocate in kernels** (no dynamic memory on GPU)
- **Don't ignore alignment** (causes slowdowns or errors)
- **Don't access tensors across thread blocks** (undefined behavior)
- **Don't forget synchronization** after SMEM writes
- **Don't use dynamic layouts unnecessarily** (compile-time is faster)

---

## Debugging Tensors

### Print Tensor Info

@cute.jit
def debug_tensor(tensor: cute.Tensor):
    # Compile-time info (Python print)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor stride: {tensor.stride}")
    print(f"Tensor layout: {tensor.layout}")
    
    # Runtime info (cute.printf in kernel)
    cute.printf("Tensor data pointer: %p\n", tensor.data_ptr)
    cute.printf("Tensor size: %d\n", cute.size(tensor))

### Visualize Layout

# Use CuTe's pretty printing
print(cute.pretty_str(tensor.layout))

# Example output:
# (128,64):(1,128)
# This is 128 rows × 64 cols, column-major (stride 1 in row, 128 in col)

---

## Advanced: Custom Tensor Types

You can create custom tensor-like types for specialized use cases:

class MyCustomTensor:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
    
    def __extract_mlir_values__(self):
        # Required for CuTe DSL integration
        return [self.data, self.metadata]
    
    def __new_from_mlir_values__(self, values):
        return MyCustomTensor(values[0], values[1])

---

## Summary

**Tensors are:**
- Views of memory with structure (layout)
- Composable and transformable
- Type-safe and bounds-checked (debug mode)
- Optimized at compile time

**Key operations:**
- `cute.make_tensor()` - Create tensor from pointer and layout
- `from_dlpack()` - Convert framework tensors
- `cute.size()`, `.shape`, `.stride` - Query structure
- `cute.copy()` - Move data between memory spaces
- Partitioning - Divide among threads/blocks

**Memory spaces:**
- GMEM (global) - large, slow
- SMEM (shared) - medium, fast, block-shared
- RMEM (registers) - small, fastest, thread-private
- TMEM (tensor memory) - Blackwell-specific

---

## Next Steps

- [Data Types](./data_types.md) - Element types for tensors
- [Memory Hierarchy](../03_memory_and_layouts/memory_hierarchy.md) - Deep dive into memory spaces
- [Copy Operations](../04_operations/copy_atoms.md) - Efficient data movement
- [MMA Operations](../04_operations/mma_atoms.md) - Tensor core matrix operations

## Related Documentation
- [Layouts](./layouts.md) - Understanding tensor layouts
- [Static vs Dynamic](../03_memory_and_layouts/static_vs_dynamic.md) - Layout flexibility
- [Framework Integration](../07_framework_integration/dlpack.md) - PyTorch/JAX interop