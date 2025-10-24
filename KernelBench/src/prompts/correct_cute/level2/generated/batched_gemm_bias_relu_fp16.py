"""
Level 2 CuTe Kernel: Batched GEMM + Bias + ReLU
Operation: C[b] = ReLU(A[b] @ B[b] + bias)
Focus: Batched operations with fusion
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def batched_gemm_bias_relu_kernel(
    gA: cute.Tensor,     # (B, M, K)
    gB: cute.Tensor,     # (B, K, N)
    gBias: cute.Tensor,  # (N,) - shared across batches
    gC: cute.Tensor,     # (B, M, N)
    B: cutlass.Int32,
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """Batched Fused GEMM + Bias + ReLU"""
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    
    # Batch index from z dimension
    b = bidz
    
    TILE_SIZE = 16
    i = bidy * TILE_SIZE + tidy
    j = bidx * TILE_SIZE + tidx
    
    if b < B and i < M and j < N:
        # GEMM for batch b
        acc = 0.0
        for k in range(K):
            acc += gA[b, i, k] * gB[b, k, j]
        
        # Epilogue 1: Add bias
        biased = acc + gBias[j]
        
        # Epilogue 2: ReLU
        result = cutlass.maximum(biased, 0.0)
        
        gC[b, i, j] = result


def batched_gemm_bias_relu(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Launch Batched GEMM + Bias + ReLU kernel."""
    B_size, M, K = A.shape
    B2, K2, N = B.shape
    assert B_size == B2 and K == K2 and bias.shape[0] == N
    
    A = A.cuda().half().contiguous()
    B = B.cuda().half().contiguous()
    bias = bias.cuda().half().contiguous()
    
    C = torch.empty(B_size, M, N, dtype=torch.float16, device='cuda')
    
    gA = from_dlpack(A)
    gB = from_dlpack(B)
    gBias = from_dlpack(bias)
    gC = from_dlpack(C)
    
    TILE_SIZE = 16
    grid_dim = ((N + TILE_SIZE - 1) // TILE_SIZE, (M + TILE_SIZE - 1) // TILE_SIZE, B_size)
    block_dim = (TILE_SIZE, TILE_SIZE, 1)
    
    batched_gemm_bias_relu_kernel[grid_dim, block_dim](gA, gB, gBias, gC, B_size, M, K, N)
    
    return C


class ModelNew(torch.nn.Module):
    """Batched GEMM + Bias + ReLU model using CuTe Python DSL."""
    
    def __init__(self, N_dim: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(N_dim, dtype=torch.float16))
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return batched_gemm_bias_relu(A, B, self.bias)


B = 16
M = 256
K = 256
N = 256

def get_inputs():
    A = torch.randn(B, M, K)
    B_mat = torch.randn(B, K, N)
    return [A, B_mat]

def get_init_inputs():
    return [N]
