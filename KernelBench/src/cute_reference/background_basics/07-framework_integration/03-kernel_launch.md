---
topic: "kernel_launch"
difficulty: "beginner"
related_topics: ["pytorch_integration", "jit_compilation", "thread_layouts"]
---

# Kernel Launch Configuration

## Overview

Launching CuTe DSL kernels requires specifying the execution configuration: how many thread blocks to use, how many threads per block, and other launch parameters. Understanding launch configuration is essential for achieving good performance.

**Key concepts:**
- Grid and block dimensions
- Launch syntax
- Thread/block indexing
- Occupancy considerations
- Launch parameters
- Dynamic vs static configuration

---

## Launch Basics

### The launch() Method

**Every CuTe kernel has a `launch()` method:**
```python
@cute.kernel
def my_kernel(data: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    data[tid] = 0.0

# Launch with grid and block dimensions
my_kernel.launch(grid=[4, 1, 1], block=[256, 1, 1])(data)
```

**Syntax breakdown:**
```python
kernel.launch(grid=..., block=...)(arguments)
#      ↑                            ↑
#      Launch config               Kernel args
```

### Grid and Block Dimensions

**3D configuration:**
```python
# Grid: (x, y, z) number of blocks
grid = [gridDim_x, gridDim_y, gridDim_z]

# Block: (x, y, z) threads per block
block = [blockDim_x, blockDim_y, blockDim_z]

kernel.launch(grid=grid, block=block)(args)
```

**Common patterns:**
```python
# 1D launch
grid = [num_blocks, 1, 1]
block = [threads_per_block, 1, 1]

# 2D launch
grid = [blocks_x, blocks_y, 1]
block = [threads_x, threads_y, 1]

# 3D launch
grid = [blocks_x, blocks_y, blocks_z]
block = [threads_x, threads_y, threads_z]
```

---

## 1D Launch Configuration

### Basic 1D Kernel

**Process 1D array:**
```python
import torch
import cutlass.cute as cute

@cute.kernel
def scale_1d(data: cute.Tensor, scale: cutlass.Float32):
    # Thread index
    tid = cute.arch.thread_idx()[0]
    
    # Block index
    bid = cute.arch.block_idx()[0]
    
    # Block size
    block_size = cute.arch.block_dim_x()
    
    # Global index
    idx = bid * block_size + tid
    
    # Bounds check
    if idx < cute.size(data):
        data[idx] = data[idx] * scale

# Launch configuration
n = 10000  # Array size
threads_per_block = 256
num_blocks = (n + threads_per_block - 1) // threads_per_block

grid = [num_blocks, 1, 1]
block = [threads_per_block, 1, 1]

# Create data
data = torch.randn(n, device='cuda')
cute_data = cute.from_dlpack(data)

# Launch
scale_1d.launch(grid=grid, block=block)(cute_data, scale=2.0)
torch.cuda.synchronize()
```

### Calculating Grid Size

**Common pattern:**
```python
def calculate_grid_1d(n: int, block_size: int) -> tuple:
    """Calculate 1D grid size for n elements"""
    num_blocks = (n + block_size - 1) // block_size
    return (num_blocks, 1, 1)

# Usage
n = 10000
block = (256, 1, 1)
grid = calculate_grid_1d(n, block[0])

kernel.launch(grid=grid, block=block)(data)
```

---

## 2D Launch Configuration

### Basic 2D Kernel

**Process 2D matrix:**
```python
@cute.kernel
def scale_2d(matrix: cute.Tensor, scale: cutlass.Float32):
    # Thread indices
    tx = cute.arch.thread_idx()[0]
    ty = cute.arch.thread_idx()[1]
    
    # Block indices
    bx = cute.arch.block_idx()[0]
    by = cute.arch.block_idx()[1]
    
    # Block dimensions
    block_width = cute.arch.block_dim_x()
    block_height = cute.arch.block_dim_y()
    
    # Global indices
    i = bx * block_width + tx   # Row
    j = by * block_height + ty  # Column
    
    # Matrix dimensions
    M, N = cute.shape(matrix)
    
    # Bounds check
    if i < M and j < N:
        matrix[i, j] = matrix[i, j] * scale

# Launch configuration
M, N = 1024, 2048  # Matrix size
tile_m, tile_n = 16, 16  # Block size

grid = [
    (M + tile_m - 1) // tile_m,
    (N + tile_n - 1) // tile_n,
    1
]
block = [tile_m, tile_n, 1]

# Create matrix
matrix = torch.randn(M, N, device='cuda')
cute_matrix = cute.from_dlpack(matrix)

# Launch
scale_2d.launch(grid=grid, block=block)(cute_matrix, scale=2.0)
```

### 2D Grid Calculation
```python
def calculate_grid_2d(
    m: int, n: int,
    tile_m: int, tile_n: int
) -> tuple:
    """Calculate 2D grid size for m×n matrix"""
    blocks_m = (m + tile_m - 1) // tile_m
    blocks_n = (n + tile_n - 1) // tile_n
    return (blocks_m, blocks_n, 1)

# Usage
M, N = 1024, 2048
tile_m, tile_n = 16, 16

grid = calculate_grid_2d(M, N, tile_m, tile_n)
block = (tile_m, tile_n, 1)

kernel.launch(grid=grid, block=block)(matrix)
```

---

## Thread Block Sizes

### Common Block Sizes

**Typical configurations:**
```python
# 1D blocks
block_64 = [64, 1, 1]      # Small
block_128 = [128, 1, 1]    # Medium
block_256 = [256, 1, 1]    # Common
block_512 = [512, 1, 1]    # Large
block_1024 = [1024, 1, 1]  # Maximum

# 2D blocks
block_8x8 = [8, 8, 1]      # 64 threads
block_16x16 = [16, 16, 1]  # 256 threads (very common)
block_32x32 = [32, 32, 1]  # 1024 threads (maximum)

# 3D blocks
block_8x8x4 = [8, 8, 4]    # 256 threads
block_16x8x8 = [16, 8, 8]  # 1024 threads
```

### Maximum Limits

**GPU hardware limits:**
```python
# Maximum threads per block: 1024 (most modern GPUs)
# Maximum block dimensions:
#   - x: 1024
#   - y: 1024
#   - z: 64

# Maximum grid dimensions:
#   - x: 2^31 - 1
#   - y: 65535
#   - z: 65535
```

**Valid configurations:**
```python
# ✓ VALID
block = [1024, 1, 1]     # 1024 threads (max)
block = [32, 32, 1]      # 1024 threads (max)
block = [16, 16, 4]      # 1024 threads (max)
block = [256, 1, 1]      # 256 threads (common)

# ✗ INVALID
block = [2048, 1, 1]     # > 1024 threads
block = [64, 64, 1]      # 4096 > 1024 threads
block = [1024, 2, 1]     # 2048 > 1024 threads
```

### Choosing Block Size

**Guidelines:**
```python
# Rule of thumb: Multiple of warp size (32)
block_size = 32 * n  # where n = 1, 2, 4, 8, ..., 32

# Common choices:
# - 128: Good for memory-bound kernels
# - 256: General purpose (very common)
# - 512: Compute-intensive kernels

def choose_block_size(kernel_type: str) -> int:
    """Choose block size based on kernel type"""
    if kernel_type == "memory_bound":
        return 128  # Lower occupancy OK
    elif kernel_type == "compute_bound":
        return 256  # Balance occupancy and resources
    elif kernel_type == "bandwidth_intensive":
        return 512  # Higher occupancy
    else:
        return 256  # Default
```

---

## Thread Indexing

### Accessing Thread Indices

**Get current thread position:**
```python
@cute.kernel
def indexing_example(data: cute.Tensor):
    # Thread index within block (0 to blockDim-1)
    tx = cute.arch.thread_idx()[0]
    ty = cute.arch.thread_idx()[1]
    tz = cute.arch.thread_idx()[2]
    
    # Block index within grid (0 to gridDim-1)
    bx = cute.arch.block_idx()[0]
    by = cute.arch.block_idx()[1]
    bz = cute.arch.block_idx()[2]
    
    # Block dimensions
    block_x = cute.arch.block_dim_x()
    block_y = cute.arch.block_dim_y()
    block_z = cute.arch.block_dim_z()
    
    # Grid dimensions
    grid_x = cute.arch.grid_dim_x()
    grid_y = cute.arch.grid_dim_y()
    grid_z = cute.arch.grid_dim_z()
    
    # Global index
    global_x = bx * block_x + tx
    global_y = by * block_y + ty
    global_z = bz * block_z + tz
```

### Linear Thread ID

**Convert 3D to 1D:**
```python
@cute.kernel
def linear_tid(data: cute.Tensor):
    tx = cute.arch.thread_idx()[0]
    ty = cute.arch.thread_idx()[1]
    tz = cute.arch.thread_idx()[2]
    
    block_x = cute.arch.block_dim_x()
    block_y = cute.arch.block_dim_y()
    
    # Linear thread ID within block
    linear_tid = tz * (block_x * block_y) + ty * block_x + tx
    
    # Use linear_tid...
```

---

## Launch Parameters

### Basic Parameters

**Grid and block are required:**
```python
# Minimal launch
kernel.launch(
    grid=[num_blocks, 1, 1],
    block=[threads_per_block, 1, 1]
)(args)
```

### Shared Memory (if needed)

**Allocate dynamic shared memory:**
```python
@cute.kernel
def with_shared_mem(data: cute.Tensor):
    # Dynamic shared memory allocated at launch
    smem = cute.make_smem_tensor(
        cute.make_shape(256),
        cutlass.Float32
    )
    
    tid = cute.arch.thread_idx()[0]
    smem[tid] = data[tid]
    
    cute.arch.syncthreads()
    
    data[tid] = smem[tid]

# Launch with shared memory
shared_mem_bytes = 256 * 4  # 256 floats × 4 bytes
kernel.launch(
    grid=[4, 1, 1],
    block=[256, 1, 1],
    shared_mem=shared_mem_bytes  # Dynamic shared memory
)(data)
```

### Stream (if needed)

**Launch on specific CUDA stream:**
```python
import torch

# Create CUDA stream
stream = torch.cuda.Stream()

# Launch on stream
with torch.cuda.stream(stream):
    kernel.launch(grid=grid, block=block)(data)

# Or explicitly
kernel.launch(
    grid=grid,
    block=block,
    stream=stream.cuda_stream
)(data)
```

---

## Occupancy Considerations

### What is Occupancy?

**Occupancy = Active warps / Maximum possible warps**
```python
# Each SM has limited resources:
# - Threads
# - Registers
# - Shared memory

# Higher occupancy = Better latency hiding
# But: May not always mean better performance
```

### Computing Theoretical Occupancy
```python
def compute_occupancy(
    threads_per_block: int,
    registers_per_thread: int,
    shared_mem_per_block: int,
    max_threads_per_sm: int = 2048,  # Ampere/Hopper
    max_registers_per_sm: int = 65536,  # Ampere
    max_shared_mem_per_sm: int = 102400  # 100KB, Ampere
) -> float:
    """Compute theoretical occupancy"""
    
    # Blocks limited by threads
    blocks_by_threads = max_threads_per_sm // threads_per_block
    
    # Blocks limited by registers
    registers_per_block = threads_per_block * registers_per_thread
    blocks_by_registers = max_registers_per_sm // registers_per_block
    
    # Blocks limited by shared memory
    if shared_mem_per_block > 0:
        blocks_by_smem = max_shared_mem_per_sm // shared_mem_per_block
    else:
        blocks_by_smem = float('inf')
    
    # Actual blocks
    blocks_per_sm = min(blocks_by_threads, blocks_by_registers, blocks_by_smem)
    
    # Occupancy
    active_threads = blocks_per_sm * threads_per_block
    occupancy = active_threads / max_threads_per_sm
    
    return occupancy

# Example
occ = compute_occupancy(
    threads_per_block=256,
    registers_per_thread=32,
    shared_mem_per_block=16384
)
print(f"Occupancy: {occ*100:.1f}%")
```

### Balancing Occupancy
```python
# High occupancy configuration
grid = [many_blocks, 1, 1]
block = [128, 1, 1]  # Smaller blocks = more blocks per SM

# Lower occupancy but more resources per thread
grid = [fewer_blocks, 1, 1]
block = [256, 1, 1]  # Larger blocks = more registers available
```

---

## Dynamic Launch Configuration

### Runtime Calculation

**Calculate at runtime based on problem size:**
```python
@cute.jit
def dynamic_launch(data: torch.Tensor, scale: float):
    """Launch with runtime-calculated configuration"""
    
    n = data.numel()
    
    # Choose block size
    threads_per_block = 256
    
    # Calculate grid size
    num_blocks = (n + threads_per_block - 1) // threads_per_block
    
    # Launch
    cute_data = cute.from_dlpack(data)
    scale_kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[threads_per_block, 1, 1]
    )(cute_data, scale=scale)

# Usage with different sizes
data_small = torch.randn(1000, device='cuda')
dynamic_launch(data_small, 2.0)  # Few blocks

data_large = torch.randn(1000000, device='cuda')
dynamic_launch(data_large, 2.0)  # Many blocks
```

### Adaptive Configuration

**Adjust based on GPU:**
```python
@cute.jit
def adaptive_launch(data: torch.Tensor):
    """Adapt configuration to GPU capabilities"""
    
    # Get device properties
    device = data.device
    props = torch.cuda.get_device_properties(device)
    
    # Max threads per block
    max_threads = props.max_threads_per_block
    
    # Choose block size (80% of max)
    threads_per_block = int(max_threads * 0.8)
    threads_per_block = (threads_per_block // 32) * 32  # Round to warp size
    
    # Calculate grid
    n = data.numel()
    num_blocks = (n + threads_per_block - 1) // threads_per_block
    
    print(f"GPU: {props.name}")
    print(f"Max threads: {max_threads}")
    print(f"Using: {threads_per_block} threads/block")
    print(f"Grid: {num_blocks} blocks")
    
    # Launch
    cute_data = cute.from_dlpack(data)
    kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[threads_per_block, 1, 1]
    )(cute_data)
```

---

## Multi-Dimensional Problems

### GEMM Launch Configuration

**Matrix multiplication:**
```python
@cute.jit
def gemm_launch(
    A: torch.Tensor,  # M × K
    B: torch.Tensor,  # K × N
    C: torch.Tensor   # M × N
):
    """Launch GEMM kernel"""
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    # Tile sizes
    tile_m = 128
    tile_n = 128
    
    # Grid covers output matrix
    grid = [
        (M + tile_m - 1) // tile_m,
        (N + tile_n - 1) // tile_n,
        1
    ]
    
    # Block processes one tile
    block = [256, 1, 1]  # Total threads per block
    
    cute_A = cute.from_dlpack(A)
    cute_B = cute.from_dlpack(B)
    cute_C = cute.from_dlpack(C)
    
    gemm_kernel.launch(grid=grid, block=block)(
        cute_A, cute_B, cute_C
    )
```

### Batch Processing

**Process multiple matrices:**
```python
@cute.jit
def batch_launch(batch: torch.Tensor):
    """
    Launch kernel for batch of matrices
    batch shape: [B, M, N]
    """
    
    B, M, N = batch.shape
    
    # Tile sizes
    tile_m = 16
    tile_n = 16
    
    # Grid: one block per tile per batch
    grid = [
        (M + tile_m - 1) // tile_m,
        (N + tile_n - 1) // tile_n,
        B  # Z dimension for batch
    ]
    
    block = [tile_m, tile_n, 1]
    
    cute_batch = cute.from_dlpack(batch)
    
    batch_kernel.launch(grid=grid, block=block)(cute_batch)

@cute.kernel
def batch_kernel(batch: cute.Tensor):
    """Process batch element"""
    bx = cute.arch.block_idx()[0]
    by = cute.arch.block_idx()[1]
    bz = cute.arch.block_idx()[2]  # Batch index
    
    tx = cute.arch.thread_idx()[0]
    ty = cute.arch.thread_idx()[1]
    
    # Global position
    i = bx * 16 + tx
    j = by * 16 + ty
    
    # Process batch[bz, i, j]
    if i < cute.shape(batch)[1] and j < cute.shape(batch)[2]:
        batch[bz, i, j] = batch[bz, i, j] * 2.0
```

---

## Performance Tips

### Grid-Stride Loop Pattern

**Handle variable sizes efficiently:**
```python
@cute.kernel
def grid_stride_loop(data: cute.Tensor):
    """Grid-stride loop for any problem size"""
    
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    block_size = cute.arch.block_dim_x()
    grid_size = cute.arch.grid_dim_x()
    
    # Stride by grid size
    stride = block_size * grid_size
    
    # Each thread processes multiple elements
    idx = bid * block_size + tid
    while idx < cute.size(data):
        data[idx] = data[idx] * 2.0
        idx += stride

# Launch with fixed configuration
# Works for any data size!
grid = [128, 1, 1]
block = [256, 1, 1]

small_data = torch.randn(1000, device='cuda')
grid_stride_loop.launch(grid=grid, block=block)(cute.from_dlpack(small_data))

large_data = torch.randn(1000000, device='cuda')
grid_stride_loop.launch(grid=grid, block=block)(cute.from_dlpack(large_data))
```

### Persistent Kernels

**Keep blocks resident:**
```python
@cute.kernel
def persistent_kernel(
    work_queue: cute.Tensor,
    num_items: cutlass.Int32
):
    """Persistent kernel processes multiple work items"""
    
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    # Each block processes multiple items
    for item_idx in range(bid, num_items, cute.arch.grid_dim_x()):
        # Process item
        process_item(work_queue, item_idx, tid)
        
        # Synchronize before next item
        cute.arch.syncthreads()

# Launch with fewer blocks than items
# Blocks stay resident and process multiple items
num_items = 1000
grid = [64, 1, 1]  # Fewer blocks than items
block = [256, 1, 1]

persistent_kernel.launch(grid=grid, block=block)(
    work_queue, num_items=num_items
)
```

---

## Common Patterns

### Pattern 1: Element-wise Operation
```python
def elementwise_launch(n: int, kernel):
    """Standard element-wise launch config"""
    threads_per_block = 256
    num_blocks = (n + threads_per_block - 1) // threads_per_block
    
    return (
        [num_blocks, 1, 1],  # grid
        [threads_per_block, 1, 1]  # block
    )

# Usage
n = data.numel()
grid, block = elementwise_launch(n, my_kernel)
my_kernel.launch(grid=grid, block=block)(data)
```

### Pattern 2: 2D Matrix Operation
```python
def matrix_launch(M: int, N: int, tile_m: int = 16, tile_n: int = 16):
    """Standard 2D matrix launch config"""
    grid = [
        (M + tile_m - 1) // tile_m,
        (N + tile_n - 1) // tile_n,
        1
    ]
    block = [tile_m, tile_n, 1]
    return grid, block

# Usage
M, N = matrix.shape
grid, block = matrix_launch(M, N)
matrix_kernel.launch(grid=grid, block=block)(matrix)
```

### Pattern 3: Reduction
```python
def reduction_launch(n: int):
    """Launch config for reduction"""
    # Phase 1: Reduce to num_blocks partial results
    threads_per_block = 256
    num_blocks = (n + threads_per_block - 1) // threads_per_block
    
    grid1 = [num_blocks, 1, 1]
    block1 = [threads_per_block, 1, 1]
    
    # Phase 2: Final reduction
    grid2 = [1, 1, 1]
    block2 = [threads_per_block, 1, 1]
    
    return (grid1, block1), (grid2, block2)

# Usage
(grid1, block1), (grid2, block2) = reduction_launch(n)

# Phase 1: partial reductions
partial = torch.zeros(grid1[0], device='cuda')
reduce_kernel.launch(grid=grid1, block=block1)(data, partial)

# Phase 2: final reduction
result = torch.zeros(1, device='cuda')
reduce_kernel.launch(grid=grid2, block=block2)(partial, result)
```

---

## Debugging Launch Configuration

### Validating Configuration
```python
def validate_launch_config(grid: tuple, block: tuple):
    """Validate launch configuration"""
    
    # Check dimensions
    assert len(grid) == 3, "Grid must be 3D"
    assert len(block) == 3, "Block must be 3D"
    
    # Check block size
    threads_per_block = block[0] * block[1] * block[2]
    assert threads_per_block <= 1024, f"Too many threads per block: {threads_per_block}"
    assert threads_per_block % 32 == 0, f"Block size should be multiple of 32: {threads_per_block}"
    
    # Check block dimensions
    assert block[0] <= 1024, f"Block X too large: {block[0]}"
    assert block[1] <= 1024, f"Block Y too large: {block[1]}"
    assert block[2] <= 64, f"Block Z too large: {block[2]}"
    
    # Check grid dimensions
    assert grid[0] <= 2**31 - 1, f"Grid X too large: {grid[0]}"
    assert grid[1] <= 65535, f"Grid Y too large: {grid[1]}"
    assert grid[2] <= 65535, f"Grid Z too large: {grid[2]}"
    
    print(f"✓ Valid configuration")
    print(f"  Grid: {grid}")
    print(f"  Block: {block}")
    print(f"  Total threads: {grid[0] * grid[1] * grid[2] * threads_per_block}")
```

### Print Launch Info
```python
@cute.kernel
def debug_launch_info(data: cute.Tensor):
    """Print launch configuration from kernel"""
    
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    
    # Only thread 0 of block 0 prints
    if tid == 0 and bid == 0:
        cute.printf("Grid dim: (%d, %d, %d)\n",
                   cute.arch.grid_dim_x(),
                   cute.arch.grid_dim_y(),
                   cute.arch.grid_dim_z())
        
        cute.printf("Block dim: (%d, %d, %d)\n",
                   cute.arch.block_dim_x(),
                   cute.arch.block_dim_y(),
                   cute.arch.block_dim_z())
        
        total_threads = (cute.arch.grid_dim_x() * 
                        cute.arch.grid_dim_y() * 
                        cute.arch.grid_dim_z() *
                        cute.arch.block_dim_x() * 
                        cute.arch.block_dim_y() * 
                        cute.arch.block_dim_z())
        
        cute.printf("Total threads: %d\n", total_threads)

# Launch
debug_launch_info.launch(grid=[4,1,1], block=[256,1,1])(data)
```

---

## Best Practices

### ✅ DO

**Use multiples of warp size:**
```python
block = [256, 1, 1]  # 256 = 32 * 8 ✓
```

**Calculate grid to cover problem:**
```python
num_blocks = (n + block_size - 1) // block_size
```

**Add bounds checking:**
```python
if idx < n:
    process(idx)
```

**Validate configuration:**
```python
validate_launch_config(grid, block)
```

### ❌ DON'T

**Don't use non-warp-multiple sizes:**
```python
block = [250, 1, 1]  # Not multiple of 32 ✗
```

**Don't exceed limits:**
```python
block = [2048, 1, 1]  # > 1024 threads ✗
```

**Don't assume coverage:**
```python
# ✗ BAD: May not cover all elements
grid = [n // 256, 1, 1]

# ✓ GOOD: Always covers all elements
grid = [(n + 255) // 256, 1, 1]
```

---

## Summary

**Kernel launch:**
- `kernel.launch(grid=..., block=...)(args)`
- Grid: Number of blocks (3D)
- Block: Threads per block (3D)
- Maximum 1024 threads per block

**Common configurations:**
- 1D: grid=[num_blocks, 1, 1], block=[256, 1, 1]
- 2D: grid=[blocks_x, blocks_y, 1], block=[16, 16, 1]
- Always use multiples of 32 (warp size)

**Thread indexing:**
- `thread_idx()`: Position in block
- `block_idx()`: Block position in grid
- Global index: `block_idx * block_dim + thread_idx`

**Best practices:**
- Cover entire problem: `(n + block_size - 1) // block_size`
- Add bounds checking: `if idx < n`
- Validate configuration
- Use 256 threads/block as default

---

## Next Steps

- [Thread Layouts](../08_optimization/thread_layouts.md) - Advanced thread organization
- [Occupancy](../08_optimization/performance_tips.md) - Occupancy tuning
- [Profiling](../08_optimization/profiling.md) - Measuring performance

---

## Further Reading

- [CUDA Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)
- [Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)
- [CUDA Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)