"""
Batched Matrix Multiplication using CuTe Python DSL
Focus: 3D tensor operations, batch parallelism
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def batched_gemm_kernel(
    gA: cute.Tensor,  # (B, M, K)
    gB: cute.Tensor,  # (B, K, N)
    gC: cute.Tensor,  # (B, M, N)
    B: cutlass.Int32,
    M: cutlass.Int32,
    K: cutlass.Int32,
    N: cutlass.Int32,
):
    """
    Batched GEMM: C[b] = A[b] @ B[b]
    Each block handles one element across all batches
    """
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    # Batch index from z dimension
    b = bidz
    
    # Matrix indices
    i = bidy * bdimy + tidy
    j = bidx * bdimx + tidx
    
    if b < B and i < M and j < N:
        acc = 0.0
        for k in range(K):
            acc += gA[b, i, k] * gB[b, k, j]
        gC[b, i, j] = acc


def cute_batched_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for batched GEMM."""
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2
    
    c = torch.empty(B, M, N, dtype=a.dtype, device=a.device)
    
    gA = from_dlpack(a)
    gB = from_dlpack(b)
    gC = from_dlpack(c)
    
    threads_per_block = (16, 16, 1)
    num_blocks = ((N + 15) // 16, (M + 15) // 16, B)
    
    batched_gemm_kernel.launch(
        dim3=num_blocks,
        dim3=threads_per_block,
        args=(gA, gB, gC, B, M, K, N)
    )
    
    return c


class ModelNew(torch.nn.Module):
    """Batched GEMM model using CuTe Python DSL."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return cute_batched_gemm(A, B)


B = 16
M = 128
K = 128
N = 128

def get_inputs():
    A = torch.randn(B, M, K)
    B_mat = torch.randn(B, K, N)
    return [A, B_mat]

def get_init_inputs():
    return []
