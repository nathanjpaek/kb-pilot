"""
Problem Name: 93_masked_cumsum
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.158 runtime_stats={'mean': 0.158, 'std': 0.018, 'min': 0.152, 'max': 0.307, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0436, 'std': 0.0103, 'min': 0.04, 'max': 0.112, 'num_trials': 100}, 'speedup_ratio': 0.276}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def masked_cumsum(M, N, in_dtype="float16"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((M, N), in_dtype),
        Mask: T.Tensor((M, N), "bool"),
        Out: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(M, threads=1) as by:
            acc = T.alloc_local((1,), in_dtype)
            acc[0] = 0
            for i in range(N):
                add_val = T.if_then_else(Mask[by, i], X[by, i], 0)
                acc[0] += add_val
                Out[by, i] = acc[0]

    return main


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self._cached_kernels = {}

    def _get_kernel(self, batch_size, length, dtype_str):
        key = (batch_size, length, dtype_str)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = masked_cumsum(batch_size, length, dtype_str)
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert self.dim == 1, "Current implementation supports dim==1 only."
        orig_dtype = x.dtype

        x = x.to(device="cuda", dtype=torch.float16)
        mask = mask.to(device="cuda", dtype=torch.bool)

        batch_size, length = x.shape
        kernel = self._get_kernel(batch_size, length, "float16")
        out = kernel(x, mask)

        return out.to(orig_dtype)