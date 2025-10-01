"""
Problem Name: 44_Conv3d_Tanh_GELU_Clamp
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.89 runtime_stats={'mean': 1.89, 'std': 0.005, 'min': 1.89, 'max': 1.91, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.94, 'std': 0.0276, 'min': 1.93, 'max': 2.2, 'num_trials': 100}, 'speedup_ratio': 1.03}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory :  Tanh  →  GELU  →  Clamp                                   #
# --------------------------------------------------------------------------- #
def _build_fused_act_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    clamp_min: float,
    clamp_max: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOTAL = N * C * D * H * W
    INV_SQRT2 = 0.7071067811865476  # 1/sqrt(2)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, C, D, H, W), dtype),
    ):
        half_f       = T.Cast(accum_dtype, 0.5)
        one_f        = T.Cast(accum_dtype, 1.0)
        inv_sqrt2_f  = T.Cast(accum_dtype, INV_SQRT2)
        cmin_f       = T.Cast(accum_dtype, float(clamp_min))
        cmax_f       = T.Cast(accum_dtype, float(clamp_max))

        with T.Kernel(T.ceildiv(TOTAL, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOTAL:
                w = idx % W
                t1 = idx // W
                h = t1 % H
                t2 = t1 // H
                d_ = t2 % D
                t3 = t2 // D
                c = t3 % C
                n = t3 // C

                val = T.Cast(accum_dtype, X[n, c, d_, h, w])

                # Tanh
                val = T.tanh(val)

                # GELU (erf formulation)
                val = half_f * val * (one_f + T.erf(val * inv_sqrt2_f))

                # Clamp [clamp_min, clamp_max]
                val = T.max(val, cmin_f)
                val = -T.max(-val, -cmax_f)

                Y[n, c, d_, h, w] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper using TileLang                                              #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d  →  fused( Tanh → GELU → Clamp )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        clamp_min: float,
        clamp_max: float,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # kernel cache : (N,C,D,H,W,dtype) -> compiled kernel
        self._kernels: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, shape, dtype: str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_fused_act_kernel(
                N,
                C,
                D,
                H,
                W,
                self.clamp_min,
                self.clamp_max,
                dtype=dtype,
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch convolution (same params / init as reference)
        x = self.conv(x)

        # Prepare for fused TileLang kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_fp16.shape

        kernel = self._get_kernel((N, C, D, H, W), "float16")
        y_fp16 = kernel(x_fp16)

        return y_fp16.to(x.dtype)