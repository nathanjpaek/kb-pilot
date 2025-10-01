"""
Problem Name: 37_FrobeniusNorm_
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=118.0 runtime_stats={'mean': 118.0, 'std': 0.0543, 'min': 118.0, 'max': 119.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.241, 'std': 0.0296, 'min': 0.217, 'max': 0.348, 'num_trials': 100}, 'speedup_ratio': 0.00204}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_sumsq_kernel(numel: int, block_size: int = 256,
                       dtype: str = "float32", accum_dtype: str = "float32"):
    @tilelang.jit()        # no auto-allocated output – user supplies the scalar tensor
    @T.prim_func
    def kernel(
        X: T.Tensor((numel,), dtype),
        S: T.Tensor((1,),     accum_dtype),   # running sum of squares
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx

            partial = T.alloc_local((1,), accum_dtype)
            partial[0] = T.Cast(accum_dtype, 0)

            if idx < numel:
                v = T.Cast(accum_dtype, X[idx])
                partial[0] = v * v

            T.atomic_add(S[0], partial[0])

    return kernel


def build_scale_kernel(numel: int, block_size: int = 256,
                       dtype: str = "float32"):
    @tilelang.jit(out_idx=-1)   # Y is auto-allocated
    @T.prim_func
    def kernel(
        X:        T.Tensor((numel,), dtype),
        inv_norm: T.Tensor((1,),     dtype),
        Y:        T.Tensor((numel,), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                Y[idx] = X[idx] * inv_norm[0]

    return kernel


class ModelNew(nn.Module):
    """
    Optimized Frobenius-norm normalization using TileLang.
    """

    def __init__(self):
        super().__init__()
        self._sumsq_cache = {}
        self._scale_cache = {}

    def _get_sumsq_kernel(self, numel: int, dtype: str):
        key = (numel, dtype)
        if key not in self._sumsq_cache:
            self._sumsq_cache[key] = build_sumsq_kernel(numel, dtype=dtype)
        return self._sumsq_cache[key]

    def _get_scale_kernel(self, numel: int, dtype: str):
        key = (numel, dtype)
        if key not in self._scale_cache:
            self._scale_cache[key] = build_scale_kernel(numel, dtype=dtype)
        return self._scale_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(device="cuda", dtype=torch.float32).contiguous()
        numel = x_fp32.numel()

        # 1. Sum of squares
        sumsq = torch.zeros(1, dtype=torch.float32, device="cuda")
        sumsq_kernel = self._get_sumsq_kernel(numel, "float32")
        sumsq_kernel(x_fp32.view(-1), sumsq)

        # 2. Compute inverse norm
        norm_val = torch.sqrt(sumsq[0])
        inv_norm = torch.tensor([1.0 / norm_val], dtype=torch.float32, device="cuda")

        # 3. Scale tensor
        scale_kernel = self._get_scale_kernel(numel, "float32")
        y_fp32 = scale_kernel(x_fp32.view(-1), inv_norm).view_as(x_fp32)

        return y_fp32.to(orig_dtype)