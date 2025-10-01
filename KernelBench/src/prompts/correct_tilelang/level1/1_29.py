"""
Problem Name: 29_Softplus
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0355 runtime_stats={'mean': 0.0355, 'std': 0.00189, 'min': 0.0334, 'max': 0.0466, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0171, 'std': 0.000838, 'min': 0.0162, 'max': 0.0226, 'num_trials': 100}, 'speedup_ratio': 0.482}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _softplus_kernel(numel: int, block_size: int = 256, in_dtype="float16", accum_dtype="float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def softplus(
        X: T.Tensor((numel,), in_dtype),
        Y: T.Tensor((numel,), in_dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            for tx in T.Parallel(block_size):
                idx: T.int32 = bx * block_size + tx
                if idx < numel:
                    val_f32 = T.Cast(accum_dtype, X[idx])
                    exp_val = T.exp(val_f32)
                    out_f32 = T.log(exp_val + 1.0)
                    Y[idx] = T.Cast(in_dtype, out_f32)

    return softplus


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._cached_kernels = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_half = x.to(device="cuda", dtype=torch.float16)
        numel = x_half.numel()
        key = (numel, x_half.dtype)

        if key not in self._cached_kernels:
            self._cached_kernels[key] = _softplus_kernel(numel)

        kernel = self._cached_kernels[key]
        y_half = kernel(x_half.reshape(-1))
        return y_half.reshape_as(x_half)