"""
Small Matrix Multiplication using CuTe Python DSL
Focus: Basic GEMM with accumulation
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def small_gemm_kernel(
    gA: cute.Tensor,  # (M, K)
    gB: cute.Tensor,  # (K, N)
    gC: cute.Tensor,  # (M, N)
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """
    Matrix multiplication: C = A @ B
    Small version with basic thread-level computation
    """
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    i = bidy * bdimy + tidy
    j = bidx * bdimx + tidx
    
    if i < M and j < N:
        acc = 0.0
        for k in range(K):
            acc += gA[i, k] * gB[k, j]
        gC[i, j] = acc


def cute_small_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for small GEMM using CuTe Python DSL."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    
    c = torch.empty(M, N, dtype=a.dtype, device=a.device)
    
    gA = from_dlpack(a)
    gB = from_dlpack(b)
    gC = from_dlpack(c)
    
    threads_per_block = (16, 16)
    num_blocks = ((N + 15) // 16, (M + 15) // 16)
    
    small_gemm_kernel.launch(
        dim3=num_blocks,
        dim3=threads_per_block,
        args=(gA, gB, gC, M, K, N)
    )
    
    return c


class ModelNew(torch.nn.Module):
    """Small GEMM model using CuTe Python DSL."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return cute_small_gemm(A, B)


M = 128
K = 128
N = 128

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []
