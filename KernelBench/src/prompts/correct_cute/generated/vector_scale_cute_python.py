"""
Vector Scaling using CuTe Python DSL
Focus: Broadcasting, scalar multiplication
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def vector_scale_kernel(
    gA: cute.Tensor,     # Input vector
    gC: cute.Tensor,     # Output vector
    scale: cutlass.Float32,
    N: cutlass.Int32,
):
    """
    Vector scaling: C = scale * A
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    
    idx = bidx * bdimx + tidx
    
    if idx < N:
        a_val = gA[idx]
        gC[idx] = scale * a_val


def cute_vector_scale(a: torch.Tensor, scale: float) -> torch.Tensor:
    """Wrapper for vector scaling using CuTe Python DSL."""
    N = a.numel()
    c = torch.empty_like(a)
    
    gA = from_dlpack(a)
    gC = from_dlpack(c)
    
    threads_per_block = 256
    num_blocks = (N + threads_per_block - 1) // threads_per_block
    
    vector_scale_kernel.launch(
        dim3=(num_blocks, 1, 1),
        dim3=(threads_per_block, 1, 1),
        args=(gA, gC, scale, N)
    )
    
    return c


class ModelNew(torch.nn.Module):
    """Vector scaling model using CuTe Python DSL."""
    
    def __init__(self, scale: float = 2.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, A: torch.Tensor) -> torch.Tensor:
        return cute_vector_scale(A, self.scale)


M = 1024
N = 1024

def get_inputs():
    A = torch.randn(M, N)
    return [A]

def get_init_inputs():
    return []
