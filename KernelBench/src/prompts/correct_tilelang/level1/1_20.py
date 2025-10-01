"""
Problem Name: 20_LeakyReLU
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0327 runtime_stats={'mean': 0.0327, 'std': 0.00106, 'min': 0.0308, 'max': 0.0356, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0177, 'std': 0.00106, 'min': 0.0167, 'max': 0.0245, 'num_trials': 100}, 'speedup_ratio': 0.541}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _leaky_relu_kernel(M: int, N: int, negative_slope: float, dtype: str = "float16", threads: int = 256):
    MN = M * N

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(MN, threads), threads=threads) as bx:
            X_flat = T.Tensor((MN,), dtype, X.data)
            Y_flat = T.Tensor((MN,), dtype, Y.data)
            for tx in T.Parallel(threads):
                idx: T.int32 = bx * threads + tx
                if idx < MN:
                    val = X_flat[idx]
                    Y_flat[idx] = T.if_then_else(val >= 0, val, val * negative_slope)

    return main


class ModelNew(nn.Module):
    """
    Optimized LeakyReLU model using TileLang.
    """

    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = float(negative_slope)
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, dtype_str: str):
        key = (M, N, dtype_str)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _leaky_relu_kernel(
                M, N, self.negative_slope, dtype=dtype_str
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TileLang kernels currently support float16 on CUDA for best performance
        x_in = x.to(device="cuda", dtype=torch.float16).contiguous()
        M, N = x_in.shape
        kernel = self._get_kernel(M, N, "float16")
        y = kernel(x_in)
        return y