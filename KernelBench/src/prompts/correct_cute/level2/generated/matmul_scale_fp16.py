"""
Level 2 CuTe Kernel: Matrix Multiplication + Scaling
Operation: C = scale * (A @ B)
Focus: Scalar fusion in epilogue
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def matmul_scale_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    scale: cutlass.Float32,
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """Fused MatMul + Scale: C = scale * (A @ B)"""
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    TILE_SIZE = 16
    i = bidy * TILE_SIZE + tidy
    j = bidx * TILE_SIZE + tidx
    
    if i < M and j < N:
        acc = 0.0
        for k in range(K):
            acc += gA[i, k] * gB[k, j]
        
        # Epilogue: Scale
        result = scale * acc
        gC[i, j] = result


def matmul_scale(A: torch.Tensor, B: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Launch MatMul + Scale kernel."""
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
    
    matmul_scale_kernel[grid_dim, block_dim](gA, gB, gC, scale, M, K, N)
    
    return C


class ModelNew(torch.nn.Module):
    """MatMul + Scale model using CuTe Python DSL."""
    
    def __init__(self, scale: float = 0.5):
        super().__init__()
        self.scale = scale
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul_scale(A, B, self.scale)


M = 256
K = 256
N = 256

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return [0.5]
