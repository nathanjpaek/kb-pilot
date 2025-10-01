"""
Problem Name: 48_Mean_reduction_over_a_dimension
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0349 runtime_stats={'mean': 0.0349, 'std': 0.0018, 'min': 0.0329, 'max': 0.0486, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0206, 'std': 0.00927, 'min': 0.0176, 'max': 0.112, 'num_trials': 100}, 'speedup_ratio': 0.59}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_mean_axis1_kernel(B, D1, D2, block_N: int = 128,
                             dtype: str = "float16", accum_dtype: str = "float32"):
    """
    Create a specialised kernel that computes mean over axis-1
    for tensors of shape (B, D1, D2).
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((B, D1, D2), dtype),
        Y: T.Tensor((B, D2), dtype),   # axis-1 removed
    ):
        with T.Kernel(B, T.ceildiv(D2, block_N), threads=block_N) as (bx, by):
            acc = T.alloc_fragment((block_N,), accum_dtype)
            T.clear(acc)

            # accumulate sums along dim=1
            for r in range(D1):
                for j in T.Parallel(block_N):
                    col = by * block_N + j
                    val = T.if_then_else(
                        col < D2,
                        T.Cast(accum_dtype, X[bx, r, col]),
                        T.Cast(accum_dtype, 0),
                    )
                    acc[j] += val

            # convert to mean and store
            denom = T.Cast(accum_dtype, D1)
            for j in T.Parallel(block_N):
                col = by * block_N + j
                if col < D2:
                    mean_val = acc[j] / denom
                    Y[bx, col] = T.Cast(dtype, mean_val)

    return kernel


class ModelNew(nn.Module):
    """
    TileLang-accelerated mean reduction (dim == 1) for 3-D tensors.
    Falls back to PyTorch otherwise.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self._kernel_cache = {}   # keyed by (B,D1,D2,dtype)

    def _get_kernel(self, B, D1, D2, dtype="float16"):
        key = (B, D1, D2, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_mean_axis1_kernel(B, D1, D2, dtype=dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # support only dim==1 & 3-D; else fallback
        if x.dim() != 3 or self.dim != 1:
            return torch.mean(x, dim=self.dim)

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        B, D1, D2 = x_fp16.shape
        kernel = self._get_kernel(B, D1, D2, "float16")

        y_fp16 = kernel(x_fp16)
        return y_fp16.to(x.dtype)