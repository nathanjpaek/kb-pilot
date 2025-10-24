"""
Fused GEMM + Bias + ReLU using CuTe Python DSL
Focus: Multiple operation fusion, activation functions
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def fused_gemm_bias_relu_kernel(
    gA: cute.Tensor,    # (M, K)
    gB: cute.Tensor,    # (K, N)
    gBias: cute.Tensor, # (N,)
    gC: cute.Tensor,    # (M, N)
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """
    Fused operation: C = ReLU((A @ B) + Bias)
    Multiple epilogues: bias + ReLU activation
    """
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    i = bidy * bdimy + tidy
    j = bidx * bdimx + tidx
    
    if i < M and j < N:
        # GEMM computation
        acc = 0.0
        for k in range(K):
            acc += gA[i, k] * gB[k, j]
        
        # Epilogue 1: Add bias
        bias_val = gBias[j]
        result = acc + bias_val
        
        # Epilogue 2: ReLU activation
        result = cute.max(result, 0.0)
        
        gC[i, j] = result


def cute_fused_gemm_bias_relu(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Wrapper for fused GEMM + Bias + ReLU."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and bias.shape[0] == N
    
    c = torch.empty(M, N, dtype=a.dtype, device=a.device)
    
    gA = from_dlpack(a)
    gB = from_dlpack(b)
    gBias = from_dlpack(bias)
    gC = from_dlpack(c)
    
    threads_per_block = (16, 16)
    num_blocks = ((N + 15) // 16, (M + 15) // 16)
    
    fused_gemm_bias_relu_kernel.launch(
        dim3=num_blocks,
        dim3=threads_per_block,
        args=(gA, gB, gBias, gC, M, K, N)
    )
    
    return c


class ModelNew(torch.nn.Module):
    """Fused GEMM + Bias + ReLU model using CuTe Python DSL."""
    
    def __init__(self, K_dim: int, N_dim: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(N_dim))
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return cute_fused_gemm_bias_relu(A, B, self.bias)


M = 512
K = 512
N = 512

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return [K, N]
