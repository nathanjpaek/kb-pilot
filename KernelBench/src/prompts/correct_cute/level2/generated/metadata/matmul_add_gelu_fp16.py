"""
Level 2 CuTe Kernel: MatMul + Add + GELU
Operation: C = GELU(A @ B + residual)
Focus: GELU activation, residual connection fusion
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def matmul_add_gelu_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gResidual: cute.Tensor,  # Residual tensor (M, N)
    gC: cute.Tensor,
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """Fused MatMul + Add + GELU"""
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    TILE_SIZE = 16
    i = bidy * TILE_SIZE + tidy
    j = bidx * TILE_SIZE + tidx
    
    if i < M and j < N:
        # MatMul
        acc = 0.0
        for k in range(K):
            acc += gA[i, k] * gB[k, j]
        
        # Epilogue 1: Add residual
        residual_val = gResidual[i, j]
        summed = acc + residual_val
        
        # Epilogue 2: GELU approximation
        # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        x_cubed = summed * summed * summed
        inner = 0.79788456 * (summed + 0.044715 * x_cubed)
        tanh_val = cutlass.tanh(inner)
        result = 0.5 * summed * (1.0 + tanh_val)
        
        gC[i, j] = result


def matmul_add_gelu(A: torch.Tensor, B: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Launch MatMul + Add + GELU kernel."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2 and residual.shape == (M, N)
    
    A = A.cuda().half().contiguous()
    B = B.cuda().half().contiguous()
    residual = residual.cuda().half().contiguous()
    
    C = torch.empty(M, N, dtype=torch.float16, device='cuda')
    
    gA = from_dlpack(A)
    gB = from_dlpack(B)
    gResidual = from_dlpack(residual)
    gC = from_dlpack(C)
    
    TILE_SIZE = 16
    grid_dim = ((N + TILE_SIZE - 1) // TILE_SIZE, (M + TILE_SIZE - 1) // TILE_SIZE, 1)
    block_dim = (TILE_SIZE, TILE_SIZE, 1)
    
    matmul_add_gelu_kernel[grid_dim, block_dim](gA, gB, gResidual, gC, M, K, N)
    
    return C


class ModelNew(torch.nn.Module):
    """MatMul + Add + GELU model using CuTe Python DSL."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return matmul_add_gelu(A, B, residual)


M = 512
K = 512
N = 512

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    residual = torch.randn(M, N)
    return [A, B, residual]

def get_init_inputs():
    return []
