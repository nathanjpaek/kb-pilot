"""
Problem Name: 91_cumsum_reverse
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.11 runtime_stats={'mean': 0.11, 'std': 0.0169, 'min': 0.105, 'max': 0.277, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0394, 'std': 0.000856, 'min': 0.0385, 'max': 0.0456, 'num_trials': 100}, 'speedup_ratio': 0.358}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_rev_cumsum_kernel(B: int, N: int, dtype: str = "float32", threads: int = 1):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def rev_cumsum(
        X: T.Tensor((B, N), dtype),
        Y: T.Tensor((B, N), dtype),
    ):
        with T.Kernel(B, threads=threads) as bx:
            running = T.alloc_local((1,), dtype)
            running[0] = T.Cast(dtype, 0)
            for jj in T.serial(N):
                j = N - 1 - jj
                running[0] += X[bx, j]
                Y[bx, j] = running[0]

    return rev_cumsum


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim == 1 or dim == -1, "Current implementation only supports dim=1"
        self.dim = 1 if dim == -1 else dim
        self._kernel_cache = {}

    def _get_kernel(self, B: int, N: int, dtype: str = "float32"):
        key = (B, N, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_rev_cumsum_kernel(B, N, dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and self.dim in (1, -1), "Input must be 2-D, cumulative dim=1"
        x_cuda = x.to(device="cuda", dtype=torch.float32).contiguous()
        B, N = x_cuda.shape
        kernel = self._get_kernel(B, N, "float32")
        y = kernel(x_cuda)
        return y.to(x.dtype)