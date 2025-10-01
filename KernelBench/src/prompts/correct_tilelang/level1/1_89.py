"""
Problem Name: 89_cumsum
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.834 runtime_stats={'mean': 0.834, 'std': 0.0306, 'min': 0.771, 'max': 0.893, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0406, 'std': 0.019, 'min': 0.0269, 'max': 0.0936, 'num_trials': 100}, 'speedup_ratio': 0.0487}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_cumsum_kernel(M, N, block_M=128, dtype="float16", accum_dtype="float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def cumsum_kernel(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=block_M) as bx:
            acc = T.alloc_fragment((block_M,), accum_dtype)
            T.clear(acc)
            for j in T.serial(N):
                for i in T.Parallel(block_M):
                    row = bx * block_M + i
                    if row < M:
                        val = T.Cast(accum_dtype, X[row, j])
                        acc[i] += val
                        Y[row, j] = T.Cast(dtype, acc[i])

    return cumsum_kernel


class ModelNew(nn.Module):
    def __init__(self, dim, block_M=128):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.block_M = block_M
        self._cached_kernels = {}

    def _get_kernel(self, M, N, dtype):
        key = (M, N, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = build_cumsum_kernel(
                M, N, block_M=self.block_M, dtype=dtype, accum_dtype="float"
            )
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor):
        if self.dim not in (1, -1):
            return torch.cumsum(x, dim=self.dim)

        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        M, N = x_fp16.shape
        kernel = self._get_kernel(M, N, "float16")
        y_fp16 = kernel(x_fp16)
        return y_fp16.to(x.dtype)