---
topic: "layouts"
difficulty: "intermediate"
related_topics: ["tensors", "memory_hierarchy", "static_vs_dynamic"]
---

# CuTe Layout Algebra

## Core Concept

CuTe uses a hierarchical layout algebra to describe how data maps to memory. Unlike raw CUDA pointers, layouts are **composable abstractions** that handle address calculation at compile time.

**Key insight:** A layout defines a bijection (one-to-one mapping) between logical coordinates and memory locations.

---

## Why Layouts Matter

In traditional CUDA programming, you manually calculate memory addresses:

# Raw CUDA (for contrast) - manual address calculation
# int idx = blockIdx.x * blockDim.x + threadIdx.x;
# float val = data[row * width + col];  # Error-prone!

In CuTe DSL, layouts handle this automatically:

# CuTe DSL (Python) - layout handles addressing
tensor = cute.make_tensor(ptr, layout)
val = tensor[row, col]  # Layout computes address

**Benefits:**
- ✅ Fewer bugs (no manual address arithmetic)
- ✅ Compile-time optimization
- ✅ Composable abstractions
- ✅ Architecture-portable code

---

## Core Components

A layout consists of two parts: **Shape** and **Stride**.

### Shape

Describes the **logical dimensions** of a tensor.

**Examples:**

# 1D tensor with 128 elements
shape = cute.make_shape(128)

# 2D tensor: 64 rows × 32 columns
shape = cute.make_shape(64, 32)

# 3D tensor: 8 × 16 × 32
shape = cute.make_shape(8, 16, 32)

# Hierarchical shape (nested)
shape = cute.make_shape((64, 32), 16)  # (M, N) × K

**Shape tells you:** "How many elements along each dimension?"

### Stride

Describes **how to traverse memory** - the step size between consecutive elements in each dimension.

**Examples:**

# Row-major stride for 2D tensor (64, 32)
stride = cute.make_stride(32, 1)  # Move 32 to next row, 1 to next col

# Column-major stride for 2D tensor (64, 32)
stride = cute.make_stride(1, 64)  # Move 1 to next row, 64 to next col

**Stride tells you:** "How many memory locations to jump for each dimension?"

### Putting It Together: Layout

A layout combines shape and stride:

# Column-major 64×32 matrix
layout = cute.make_layout(
    cute.make_shape(64, 32),    # 64 rows, 32 columns
    cute.make_stride(1, 64)     # Column-major: stride 1 in rows
)

**Memory layout visualization:**

Column-major (stride 1 in row direction):
Memory: [A00, A10, A20, ..., A60, A01, A11, A21, ...]
         └── same column ──┘  └── next column ──┘

Row-major (stride 1 in column direction):
Memory: [A00, A01, A02, ..., A31, A10, A11, A12, ...]
         └── same row ──┘  └── next row ──┘

---

## Layout Composition

Layouts can be nested and composed to create complex memory patterns.

### Example: Tiled Matrix Layout

# Divide a large matrix into 128×64 tiles
outer_shape = cute.make_shape(num_tiles_m, num_tiles_n)
inner_shape = cute.make_shape(128, 64)

tiled_layout = cute.make_layout(
    cute.make_shape(outer_shape, inner_shape),
    cute.make_stride(...)
)

This creates a hierarchical structure: (Tiles_M, Tiles_N) × (Tile_M, Tile_N)

### Example: Thread-to-Data Mapping

# Map 256 threads (32×8) to a 128×64 tile
thread_layout = cute.make_layout(
    cute.make_shape(32, 8),     # 32 threads in M, 8 in N
    cute.make_stride(1, 32)     # Thread arrangement
)

---

## Common Layout Patterns

### Pattern 1: Row-Major (C/C++ default)

layout = cute.make_layout(
    cute.make_shape(M, N),
    cute.make_stride(N, 1)  # Stride N in row, 1 in column
)

**Memory order:** Row 0 entirely, then row 1, then row 2, ...

**Good for:** Sequential row access

### Pattern 2: Column-Major (Fortran/BLAS default)

layout = cute.make_layout(
    cute.make_shape(M, N),
    cute.make_stride(1, M)  # Stride 1 in row, M in column
)

**Memory order:** Column 0 entirely, then column 1, then column 2, ...

**Good for:** Sequential column access, BLAS compatibility

### Pattern 3: Swizzled Layout (for shared memory)

Swizzled layouts avoid bank conflicts in shared memory.

# Swizzled layout to avoid bank conflicts (conceptual)
# In Python CuTe, prefer composing/swizzling via supported helpers such as
# cute.make_composed_layout or recast_layout/recast_tensor where applicable.
# Example (pseudo-API):
# layout_inner = cute.make_layout(cute.make_shape(128, 64), cute.make_stride(1, 128))
# layout = cute.make_composed_layout(inner=layout_inner, outer=..., offset=...)

### Pattern 4: Blocked/Tiled Layout

# Block a matrix into 16×16 tiles
tile_size = (16, 16)
outer_layout = cute.make_layout(
    cute.make_shape(M // 16, N // 16),  # Number of tiles
    ...
)
inner_layout = cute.make_layout(
    cute.make_shape(16, 16),  # Tile size
    ...
)

**Purpose:** Locality for cache efficiency, GPU thread block mapping.

---

## Why This Matters for Kernel Performance

### 1. Compile-Time Optimization

Layouts enable the compiler to:
- **Eliminate address calculations** that can be computed at compile time
- **Strength reduction** (multiply → shift for powers of 2)
- **Constant propagation** through address arithmetic

### 2. Bank Conflict Avoidance

Proper stride selection prevents shared memory bank conflicts:

# ❌ BAD: All threads access same bank
stride = cute.make_stride(32, 1)  # Stride 32 = bank conflict!

# ✅ GOOD: Threads access different banks  
stride = cute.make_stride(1, 33)  # Padding avoids conflicts

**Bank conflicts** occur when multiple threads in a warp access different addresses in the same memory bank, serializing the accesses.

### 3. Memory Coalescing

Layouts encode memory access patterns that enable coalesced global memory access:

# ✅ Coalesced: Adjacent threads access adjacent memory
thread_layout = cute.make_layout(
    cute.make_shape(32),
    cute.make_stride(1)
)

# ❌ Uncoalesced: Adjacent threads access strided memory
thread_layout = cute.make_layout(
    cute.make_shape(32),
    cute.make_stride(32)  # 32-way stride = poor coalescing
)

**Coalescing** means combining multiple memory accesses from a warp into fewer, wider transactions - critical for bandwidth.

---

## Working with Layouts in Practice

### Creating a Tensor with a Layout

# Define layout
layout = cute.make_layout(
    cute.make_shape(128, 64),
    cute.make_stride(1, 128)  # Column-major
)

# Create tensor from pointer and layout
tensor = cute.make_tensor(ptr, layout)

# Access elements (layout handles addressing)
value = tensor[row, col]

### Partitioning Layouts for Thread Blocks

# Global tensor layout
global_layout = cute.make_layout(
    cute.make_shape(M, N),
    cute.make_stride(1, M)
)

# Tile it for thread blocks
tile_shape = cute.make_shape(128, 64)
tiled = cute.zipped_divide(global_layout, tile_shape)

# Now tiled has structure: ((Tile_M, Tile_N), (Block_M, Block_N))

### Slicing and Indexing

# Get a specific tile
tile = tiled[block_idx_m, block_idx_n]

# Get a subset
subset = tensor[cute.make_coord(start_row, start_col), 
                cute.make_coord(end_row, end_col)]

---

## Architecture-Specific Considerations

### Ampere (SM80)

- 128KB shared memory per block
- 32-way bank conflicts to avoid
- 16x8x16 MMA atoms require specific layouts for A/B matrices

**Typical layout for MMA:**
MMA operations expect data laid out to match tensor core requirements.

### Hopper (SM90)

- 256KB shared memory (configurable)
- TMA (Tensor Memory Accelerator) requires aligned layouts
- Specific stride requirements for TMA descriptors

**TMA-compatible layout:**
Must be contiguous in the dimension being transferred by TMA.

### Blackwell (SM100)

- Even larger shared memory options
- 2-CTA instructions may require specific layouts
- TMEM (Tensor Memory) has its own layout requirements

---

## Common Pitfalls

### ❌ Pitfall 1: Wrong Stride Causes Bank Conflicts

# BAD: Every thread accesses same bank
layout = cute.make_layout(
    cute.make_shape(128, 32),
    cute.make_stride(32, 1)  # Stride 32 = bank conflict!
)

# GOOD: Proper stride avoids conflicts
layout = cute.make_layout(
    cute.make_shape(128, 32),
    cute.make_stride(1, 128)  # Or add padding: stride(1, 129)
)

### ❌ Pitfall 2: Mismatched Layout and Memory Order

# Python/PyTorch tensors are row-major by default
torch_tensor = torch.randn(64, 32)  # Row-major

# ❌ BAD: Assuming column-major
layout = cute.make_layout(
    cute.make_shape(64, 32),
    cute.make_stride(1, 64)  # Column-major - WRONG!
)

# ✅ GOOD: Match PyTorch's row-major layout
layout = cute.make_layout(
    cute.make_shape(64, 32),
    cute.make_stride(32, 1)  # Row-major - CORRECT!
)

### ❌ Pitfall 3: Not Considering Alignment

# Tensor cores require specific alignment (often 128-bit)
# Ensure your layout strides respect alignment requirements

---

## Advanced: Layout Algebra Operations

CuTe provides operations on layouts:

### Composition

# Compose two layouts
composed = cute.composition(layout1, layout2)

### Complement

# Find the complement of a layout (unused coordinates)
complement = cute.complement(layout, shape)

### Logical Division

# Divide a layout into tiles
tiled = cute.logical_divide(layout, tile_shape)

### Zipped Division

# Divide and zip (interleave) the coordinates
zipped = cute.zipped_divide(layout, tile_shape)

---

## Practical Example: GEMM Tiling

# Problem: Multiply A (M×K) and B (K×N) to get C (M×N)

# Step 1: Define tile sizes
tile_M, tile_N, tile_K = 128, 128, 32

# Step 2: Create tiled layouts for A, B, C
# A is M×K, tiled as (tile_M × tile_K)
layout_A = cute.make_layout(
    cute.make_shape(M, K),
    cute.make_stride(1, M)  # Column-major
)
tiled_A = cute.zipped_divide(
    layout_A, 
    cute.make_shape(tile_M, tile_K)
)

# B is K×N, tiled as (tile_K × tile_N)
layout_B = cute.make_layout(
    cute.make_shape(K, N),
    cute.make_stride(1, K)  # Column-major
)
tiled_B = cute.zipped_divide(
    layout_B,
    cute.make_shape(tile_K, tile_N)
)

# C is M×N, tiled as (tile_M × tile_N)
layout_C = cute.make_layout(
    cute.make_shape(M, N),
    cute.make_stride(1, M)  # Column-major
)
tiled_C = cute.zipped_divide(
    layout_C,
    cute.make_shape(tile_M, tile_N)
)

# Step 3: Each thread block processes one tile
block_idx_m, block_idx_n = cute.arch.block_idx()[0:2]
tile_A = tiled_A[(None, block_idx_m), :]
tile_B = tiled_B[:, (None, block_idx_n)]
tile_C = tiled_C[(None, block_idx_m), (None, block_idx_n)]

---

## Summary

**Layouts provide:**
- ✅ Automatic address calculation
- ✅ Compile-time optimization
- ✅ Bank conflict avoidance
- ✅ Memory coalescing control
- ✅ Composable abstractions
- ✅ Architecture portability

**Key operations:**
- `cute.make_shape()` - Define logical dimensions
- `cute.make_stride()` - Define memory traversal
- `cute.make_layout()` - Combine shape and stride
- `cute.zipped_divide()` - Tile layouts
- `cute.make_tensor()` - Create tensors from layouts

---

## Next Steps

- [Tensors](./tensors.md) - Using layouts to create and manipulate tensors
- [Memory Hierarchy](../03_memory_and_layouts/memory_hierarchy.md) - Layouts for different memory spaces
- [Static vs Dynamic Layouts](../03_memory_and_layouts/static_vs_dynamic.md) - When layouts are known at compile time
- [Tiling Patterns](../08_optimization/tiling_patterns.md) - Advanced tiling strategies

---

## Related Sections

- [MMA Atoms](../04_operations/mma_atoms.md) - Layouts required for tensor core operations
- [Copy Atoms](../04_operations/copy_atoms.md) - Layouts for efficient data movement
- [Swizzling](../03_memory_and_layouts/memory_hierarchy.md) - Advanced shared memory layouts