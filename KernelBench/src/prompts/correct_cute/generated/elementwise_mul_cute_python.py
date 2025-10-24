"""
Element-wise Multiplication using CuTe Python DSL
Focus: Element-wise operations on 2D tensors
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def elementwise_mul_kernel(
    gA: cute.Tensor,  # Input A (M, N)
    gB: cute.Tensor,  # Input B (M, N)
    gC: cute.Tensor,  # Output (M, N)
    M: cutlass.Int32,
    N: cutlass.Int32,
):
    """
    Element-wise multiplication: C = A * B
    """
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    i = bidy * bdimy + tidy
    j = bidx * bdimx + tidx
    
    if i < M and j < N:
        gC[i, j] = gA[i, j] * gB[i, j]


def cute_elementwise_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for element-wise multiplication."""
    M, N = a.shape
    c = torch.empty_like(a)
    
    gA = from_dlpack(a)
    gB = from_dlpack(b)
    gC = from_dlpack(c)
    
    threads_per_block = (16, 16)
    num_blocks = ((N + 15) // 16, (M + 15) // 16)
    
    elementwise_mul_kernel.launch(
        dim3=num_blocks,
        dim3=threads_per_block,
        args=(gA, gB, gC, M, N)
    )
    
    return c


class ModelNew(torch.nn.Module):
    """Element-wise multiplication model using CuTe Python DSL."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return cute_elementwise_mul(A, B)


M = 512
N = 512

def get_inputs():
    A = torch.randn(M, N)
    B = torch.randn(M, N)
    return [A, B]

def get_init_inputs():
    return []
