"""
Problem Name: 47_Sum_reduction_over_a_dimension
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0348 runtime_stats={'mean': 0.0348, 'std': 0.00112, 'min': 0.0329, 'max': 0.0402, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0194, 'std': 0.00126, 'min': 0.0182, 'max': 0.0267, 'num_trials': 100}, 'speedup_ratio': 0.557}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _reduce_sum_axis1(B: int, D1: int, D2: int, block_N: int = 128, dtype: str = "float16", accum_dtype: str = "float32"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((B, D1, D2), dtype),
        Y: T.Tensor((B, 1, D2), dtype),
    ):
        # Grid: bx -> batch dimension, by -> column blocks of the last dimension
        with T.Kernel(B, T.ceildiv(D2, block_N), threads=block_N) as (bx, by):
            partial = T.alloc_fragment((block_N,), accum_dtype)
            T.clear(partial)

            # Accumulate along the reduction dimension (dim=1)
            for r in range(D1):
                for j in T.Parallel(block_N):
                    col = by * block_N + j
                    val = T.if_then_else(col < D2, X[bx, r, col], T.Cast(dtype, 0))
                    partial[j] += T.Cast(accum_dtype, val)

            # Write results back to global memory
            for j in T.Parallel(block_N):
                col = by * block_N + j
                if col < D2:
                    Y[bx, 0, col] = T.Cast(dtype, partial[j])

    return main


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using TileLang.
    Currently supports reduction over dim=1 for 3-D inputs.
    """

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self._cached_kernels = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fall back to PyTorch if the requested reduction is not along dim=1
        # or the tensor is not 3-D (this implementation supports the common use-case only).
        if self.dim != 1 or x.dim() != 3:
            return torch.sum(x, dim=self.dim, keepdim=True)

        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        B, D1, D2 = x_fp16.shape
        cache_key = (B, D1, D2, x_fp16.dtype)

        # Compile & cache kernel for the current dynamic shape
        if cache_key not in self._cached_kernels:
            self._cached_kernels[cache_key] = _reduce_sum_axis1(B, D1, D2)

        kernel = self._cached_kernels[cache_key]
        y = kernel(x_fp16)
        return y.to(x.dtype)