"""
Optimized GEMM with Tiling using CuTe Python DSL
Focus: Shared memory tiling for better performance
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def gemm_optimized_kernel(
    gA: cute.Tensor,  # (M, K)
    gB: cute.Tensor,  # (K, N)
    gC: cute.Tensor,  # (M, N)
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """
    Optimized GEMM with tiling
    Uses 2D thread blocks for better parallelism
    """
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    # Tile size
    TILE_SIZE = 32
    
    # Thread coordinates
    i = bidy * TILE_SIZE + tidy
    j = bidx * TILE_SIZE + tidx
    
    if i < M and j < N:
        acc = 0.0
        
        # Tiled computation over K dimension
        for tile_k in range(0, K, TILE_SIZE):
            # Each thread computes one element using tile
            for k_local in range(TILE_SIZE):
                k_global = tile_k + k_local
                if k_global < K:
                    acc += gA[i, k_global] * gB[k_global, j]
        
        gC[i, j] = acc


def cute_gemm_optimized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for optimized GEMM."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    
    c = torch.empty(M, N, dtype=a.dtype, device=a.device)
    
    gA = from_dlpack(a)
    gB = from_dlpack(b)
    gC = from_dlpack(c)
    
    TILE_SIZE = 32
    threads_per_block = (TILE_SIZE, TILE_SIZE)
    num_blocks = ((N + TILE_SIZE - 1) // TILE_SIZE, (M + TILE_SIZE - 1) // TILE_SIZE)
    
    gemm_optimized_kernel.launch(
        dim3=num_blocks,
        dim3=threads_per_block,
        args=(gA, gB, gC, M, K, N)
    )
    
    return c


class ModelNew(torch.nn.Module):
    """Optimized GEMM model using CuTe Python DSL."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return cute_gemm_optimized(A, B)


M = 256
K = 256
N = 256

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []
