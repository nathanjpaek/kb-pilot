"""
Problem Name: 24_LogSoftmax
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.61 runtime_stats={'mean': 0.61, 'std': 0.0508, 'min': 0.556, 'max': 0.708, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0256, 'std': 0.0235, 'min': 0.0133, 'max': 0.134, 'num_trials': 100}, 'speedup_ratio': 0.042}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_logsoftmax_kernel(B: int, N: int, dtype: str = "float32"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((B, N), dtype),
        Y: T.Tensor((B, N), dtype),
    ):
        with T.Kernel(B, 1, threads=1) as (bx, _):
            max_val = T.alloc_local((1,), dtype)
            sum_val = T.alloc_local((1,), dtype)

            # 1) find max for numerical stability
            max_val[0] = X[bx, 0]
            for j in range(1, N):
                max_val[0] = T.max(max_val[0], X[bx, j])

            # 2) accumulate exp and store (x - max) temporarily in Y
            sum_val[0] = T.Cast(dtype, 0)
            for j in range(N):
                diff = X[bx, j] - max_val[0]
                Y[bx, j] = diff
                sum_val[0] += T.exp(diff)

            # 3) compute log(sum) and finalize logsoftmax
            log_sum = T.log(sum_val[0])
            for j in range(N):
                Y[bx, j] = Y[bx, j] - log_sum

    return kernel


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        if dim != 1:
            raise NotImplementedError("ModelNew supports dim=1 only.")
        self.dim = dim
        self._kernel_cache = {}

    def _get_kernel(self, B: int, N: int, dtype: str):
        key = (B, N, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_logsoftmax_kernel(B, N, dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and self.dim == 1, "Input must be 2D with dim=1"
        x_fp32 = x.to(device="cuda", dtype=torch.float32).contiguous()
        B, N = x_fp32.shape

        kernel = self._get_kernel(B, N, "float32")
        y_fp32 = kernel(x_fp32)

        return y_fp32.to(x.dtype)