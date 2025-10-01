"""
Problem Name: 14_Conv3d_LeakyReLU_Sum_Clamp_GELU
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.529 runtime_stats={'mean': 0.529, 'std': 0.00822, 'min': 0.522, 'max': 0.6, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.628, 'std': 0.00842, 'min': 0.62, 'max': 0.701, 'num_trials': 100}, 'speedup_ratio': 1.19}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : fused LeakyReLU(+0.2) + bias_add + clamp + GELU   #
# --------------------------------------------------------------------------- #
def _build_fused_elem_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    neg_slope: float = 0.2,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total = N * C * D * H * W
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        B: T.Tensor((C,), dtype),                       # (C,1,1,1) flattened
        Out: T.Tensor((N, C, D, H, W), dtype),
    ):
        half_f       = T.Cast(accum_dtype, 0.5)
        neg_slope_f  = T.Cast(accum_dtype, float(neg_slope))
        clamp_min_f  = T.Cast(accum_dtype, float(clamp_min))
        clamp_max_f  = T.Cast(accum_dtype, float(clamp_max))
        inv_sqrt2_f  = T.Cast(accum_dtype, inv_sqrt2)

        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total:
                w   = idx % W
                t1  = idx // W
                h   = t1 % H
                t2  = t1 // H
                d_  = t2 % D
                t3  = t2 // D
                c   = t3 % C
                n   = t3 // C

                val = T.Cast(accum_dtype, X[n, c, d_, h, w])

                # LeakyReLU
                val = T.if_then_else(val > T.Cast(accum_dtype, 0), val, val * neg_slope_f)

                # Bias add (broadcast)
                val += T.Cast(accum_dtype, B[c])

                # Clamp [-1,1]
                val = T.max(val, clamp_min_f)
                val = -T.max(-val, -clamp_max_f)

                # GELU (erf formulation)
                val = half_f * val * (T.Cast(accum_dtype, 1.0) + T.erf(val * inv_sqrt2_f))

                Out[n, c, d_, h, w] = T.Cast(dtype, val)

    return fused_kernel


# --------------------------------------------------------------------------- #
# Optimised PyTorch module                                                    #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d → fused(LeakyReLU + bias_add + clamp + GELU) implemented in TileLang.
    """

    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        # Conv3d with identical default initialisation
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

        # Learnable bias tensor (same init as original)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

        # Kernel cache : key = (N,C,D,H,W,dtype)
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # --------------------------------------------------------------------- #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_elem_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard Conv3d first
        x = self.conv(x)

        # Move to CUDA/fp16 for fused kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_fp16.shape

        bias_fp16 = (
            self.sum_tensor.view(-1)
            .to(device="cuda", dtype=torch.float16)
            .contiguous()
        )

        kernel = self._get_kernel(N, C, D, H, W, "float16")
        y_fp16 = kernel(x_fp16, bias_fp16)

        return y_fp16.to(x.dtype)