"""
Level 2 CuTe Kernel: Matrix Multiplication + Bias Addition
Operation: C = A @ B + bias
Focus: Basic fusion pattern, epilogue operations
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def matmul_bias_kernel(
    gA: cute.Tensor,     # (M, K)
    gB: cute.Tensor,     # (K, N)
    gBias: cute.Tensor,  # (N,) - bias vector
    gC: cute.Tensor,     # (M, N) - output
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """
    Fused MatMul + Bias: C = A @ B + bias
    Uses basic thread-level parallelism with epilogue fusion
    """
    # Thread indices
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    # Tile size
    TILE_SIZE = 16
    
    # Global indices for this thread
    i = bidy * TILE_SIZE + tidy
    j = bidx * TILE_SIZE + tidx
    
    if i < M and j < N:
        # Compute matrix multiplication
        acc = 0.0
        for k in range(K):
            acc += gA[i, k] * gB[k, j]
        
        # Epilogue: Add bias
        bias_val = gBias[j]
        result = acc + bias_val
        
        # Store result
        gC[i, j] = result


def matmul_bias(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Launch MatMul + Bias kernel."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    assert bias.shape[0] == N, "Bias must have N elements"
    
    A = A.cuda().half().contiguous()
    B = B.cuda().half().contiguous()
    bias = bias.cuda().half().contiguous()
    
    C = torch.empty(M, N, dtype=torch.float16, device='cuda')
    
    gA = from_dlpack(A)
    gB = from_dlpack(B)
    gBias = from_dlpack(bias)
    gC = from_dlpack(C)
    
    TILE_SIZE = 16
    grid_dim = ((N + TILE_SIZE - 1) // TILE_SIZE, (M + TILE_SIZE - 1) // TILE_SIZE, 1)
    block_dim = (TILE_SIZE, TILE_SIZE, 1)
    
    matmul_bias_kernel[grid_dim, block_dim](gA, gB, gBias, gC, M, K, N)
    
    return C


class ModelNew(torch.nn.Module):
    """MatMul + Bias model using CuTe Python DSL."""
    
    def __init__(self, K_dim: int, N_dim: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(N_dim, dtype=torch.float16))
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul_bias(A, B, self.bias)


M = 256
K = 256
N = 256

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return [K, N]
