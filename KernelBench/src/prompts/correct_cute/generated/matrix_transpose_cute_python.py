"""
Matrix Transpose using CuTe Python DSL
Focus: 2D tensor layouts, coalesced memory access
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def matrix_transpose_kernel(
    gA: cute.Tensor,  # Input (M, N)
    gC: cute.Tensor,  # Output (N, M)
    M: cutlass.Int32,
    N: cutlass.Int32,
):
    """
    Matrix transpose: C[j,i] = A[i,j]
    """
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    # Global indices
    i = bidy * bdimy + tidy
    j = bidx * bdimx + tidx
    
    if i < M and j < N:
        # Read from A[i,j], write to C[j,i]
        val = gA[i, j]
        gC[j, i] = val


def cute_transpose(a: torch.Tensor) -> torch.Tensor:
    """Wrapper for matrix transpose using CuTe Python DSL."""
    M, N = a.shape
    c = torch.empty(N, M, dtype=a.dtype, device=a.device)
    
    gA = from_dlpack(a)
    gC = from_dlpack(c)
    
    threads_per_block = (16, 16)  # 256 threads
    num_blocks = ((N + 15) // 16, (M + 15) // 16)
    
    matrix_transpose_kernel.launch(
        dim3=num_blocks,
        dim3=threads_per_block,
        args=(gA, gC, M, N)
    )
    
    return c


class ModelNew(torch.nn.Module):
    """Matrix transpose model using CuTe Python DSL."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor) -> torch.Tensor:
        return cute_transpose(A)


M = 256
N = 256

def get_inputs():
    A = torch.randn(M, N)
    return [A]

def get_init_inputs():
    return []
