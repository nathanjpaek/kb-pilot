"""
Level 2 CuTe Kernel: GEMM + Multiply + LeakyReLU
Operation: C = LeakyReLU(scale * (A @ B))
Focus: Multiply fusion + LeakyReLU
Corresponds to: KernelBench Level 2 Problem 12
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def gemm_multiply_leakyrelu_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    scale: cutlass.Float32,
    negative_slope: cutlass.Float32,
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """Fused GEMM + Multiply + LeakyReLU"""
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    TILE_SIZE = 16
    i = bidy * TILE_SIZE + tidy
    j = bidx * TILE_SIZE + tidx
    
    if i < M and j < N:
        # GEMM
        acc = 0.0
        for k in range(K):
            acc += gA[i, k] * gB[k, j]
        
        # Epilogue 1: Multiply by scale
        scaled = scale * acc
        
        # Epilogue 2: LeakyReLU
        # LeakyReLU(x) = x if x > 0 else negative_slope * x
        if scaled > 0.0:
            result = scaled
        else:
            result = negative_slope * scaled
        
        gC[i, j] = result


def gemm_multiply_leakyrelu(A: torch.Tensor, B: torch.Tensor, scale: float = 1.0, negative_slope: float = 0.01) -> torch.Tensor:
    """Launch GEMM + Multiply + LeakyReLU kernel."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    A = A.cuda().half().contiguous()
    B = B.cuda().half().contiguous()
    
    C = torch.empty(M, N, dtype=torch.float16, device='cuda')
    
    gA = from_dlpack(A)
    gB = from_dlpack(B)
    gC = from_dlpack(C)
    
    TILE_SIZE = 16
    grid_dim = ((N + TILE_SIZE - 1) // TILE_SIZE, (M + TILE_SIZE - 1) // TILE_SIZE, 1)
    block_dim = (TILE_SIZE, TILE_SIZE, 1)
    
    gemm_multiply_leakyrelu_kernel[grid_dim, block_dim](gA, gB, gC, scale, negative_slope, M, K, N)
    
    return C


class ModelNew(torch.nn.Module):
    """GEMM + Multiply + LeakyReLU model using CuTe Python DSL."""
    
    def __init__(self, scale: float = 1.0, negative_slope: float = 0.01):
        super().__init__()
        self.scale = scale
        self.negative_slope = negative_slope
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return gemm_multiply_leakyrelu(A, B, self.scale, self.negative_slope)


M = 512
K = 512
N = 512

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return [1.0, 0.01]
