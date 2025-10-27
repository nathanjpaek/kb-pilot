"""
Level 2 CuTe Kernel: GEMM + Scale + Sigmoid
Operation: C = Sigmoid(scale * (A @ B))
Focus: Scalar and activation fusion
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def gemm_scale_sigmoid_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    scale: cutlass.Float32,
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """Fused GEMM + Scale + Sigmoid: C = Sigmoid(scale * (A @ B))"""
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    TILE_SIZE = 16
    i = bidy * TILE_SIZE + tidy
    j = bidx * TILE_SIZE + tidx
    
    if i < M and j < N:
        acc = 0.0
        for k in range(K):
            acc += gA[i, k] * gB[k, j]
        
        # Epilogue 1: Scale
        scaled = scale * acc
        
        # Epilogue 2: Sigmoid (1 / (1 + exp(-x)))
        # Clamp for numerical stability
        clamped = cutlass.maximum(cutlass.minimum(scaled, 88.0), -88.0)
        result = 1.0 / (1.0 + cutlass.exp(-clamped))
        
        gC[i, j] = result


def gemm_scale_sigmoid(A: torch.Tensor, B: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Launch GEMM + Scale + Sigmoid kernel."""
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
    
    gemm_scale_sigmoid_kernel[grid_dim, block_dim](gA, gB, gC, scale, M, K, N)
    
    return C


class ModelNew(torch.nn.Module):
    """GEMM + Scale + Sigmoid model using CuTe Python DSL."""
    
    def __init__(self, scale: float = 0.5):
        super().__init__()
        self.scale = scale
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return gemm_scale_sigmoid(A, B, self.scale)


M = 512
K = 512
N = 512

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return [0.5]
