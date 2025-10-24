"""
Level 2 CuTe Kernel: GEMM + Bias + Tanh
Operation: C = Tanh(A @ B + bias)
Focus: Tanh activation fusion
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def gemm_bias_tanh_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gBias: cute.Tensor,
    gC: cute.Tensor,
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """Fused GEMM + Bias + Tanh: C = Tanh(A @ B + bias)"""
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    TILE_SIZE = 16
    i = bidy * TILE_SIZE + tidy
    j = bidx * TILE_SIZE + tidx
    
    if i < M and j < N:
        acc = 0.0
        for k in range(K):
            acc += gA[i, k] * gB[k, j]
        
        # Epilogue 1: Add bias
        biased = acc + gBias[j]
        
        # Epilogue 2: Tanh activation
        # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        clamped = cutlass.maximum(cutlass.minimum(biased, 88.0), -88.0)
        exp_pos = cutlass.exp(clamped)
        exp_neg = cutlass.exp(-clamped)
        result = (exp_pos - exp_neg) / (exp_pos + exp_neg)
        
        gC[i, j] = result


def gemm_bias_tanh(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Launch GEMM + Bias + Tanh kernel."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2 and bias.shape[0] == N
    
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
    
    gemm_bias_tanh_kernel[grid_dim, block_dim](gA, gB, gBias, gC, M, K, N)
    
    return C


class ModelNew(torch.nn.Module):
    """GEMM + Bias + Tanh model using CuTe Python DSL."""
    
    def __init__(self, K_dim: int, N_dim: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(N_dim, dtype=torch.float16))
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return gemm_bias_tanh(A, B, self.bias)


M = 512
K = 512
N = 512

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return [K, N]
