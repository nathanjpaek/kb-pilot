"""
Problem Name: 25_Swish
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0372 runtime_stats={'mean': 0.0372, 'std': 0.00152, 'min': 0.0348, 'max': 0.0445, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0243, 'std': 0.00141, 'min': 0.0226, 'max': 0.0316, 'num_trials': 100}, 'speedup_ratio': 0.653}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_swish_kernel(numel: int, block_size: int = 256, dtype: str = "float16"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def swish_kernel(
        inp: T.Tensor((numel,), dtype),
        out: T.Tensor((numel,), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            for tx in T.Parallel(block_size):
                idx = bx * block_size + tx
                if idx < numel:
                    val = inp[idx]
                    out[idx] = val * T.sigmoid(val)

    return swish_kernel


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._cached_kernels = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.to(device="cuda", dtype=torch.float16).contiguous()
        x_flat = x.view(-1)
        numel = x_flat.numel()

        key = (numel, x.dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = build_swish_kernel(numel)

        swish_kernel = self._cached_kernels[key]
        y_flat = swish_kernel(x_flat)
        y = y_flat.view(orig_shape)
        return y