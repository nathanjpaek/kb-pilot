"""
Problem Name: 21_Sigmoid
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0433 runtime_stats={'mean': 0.0433, 'std': 0.00295, 'min': 0.0397, 'max': 0.058, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0226, 'std': 0.00863, 'min': 0.018, 'max': 0.0898, 'num_trials': 100}, 'speedup_ratio': 0.522}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_sigmoid_kernel(numel: int, dtype: str = "float16"):
    block_size = 256

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def sigmoid_kernel(
        X: T.Tensor((numel,), dtype),
        Y: T.Tensor((numel,), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                val = X[idx]
                Y[idx] = 1 / (1 + T.exp(-val))

    return sigmoid_kernel


class ModelNew(nn.Module):
    """
    Optimized model using TileLang for Sigmoid activation
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._cached_kernels = {}

    def _get_kernel(self, numel: int, dtype: str = "float16"):
        key = (numel, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = _build_sigmoid_kernel(numel, dtype)
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        numel = x_fp16.numel()
        kernel = self._get_kernel(numel, "float16")
        y_fp16 = kernel(x_fp16.flatten()).view_as(x_fp16)
        return y_fp16