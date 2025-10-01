"""
Problem Name: 35_Conv3d_GELU_Tanh_Clamp
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.27 runtime_stats={'mean': 2.27, 'std': 0.0195, 'min': 2.26, 'max': 2.45, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.41, 'std': 0.0243, 'min': 2.39, 'max': 2.63, 'num_trials': 100}, 'speedup_ratio': 1.06}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                       TileLang kernel  GELU -> Tanh -> Clamp                #
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
    total = N * C * D * H * W
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, D, H, W), dtype),
        Out: T.Tensor((N, C, D, H, W), dtype),
    ):
        half_f       = T.Cast(accum_dtype, 0.5)
        one_f        = T.Cast(accum_dtype, 1.0)
        inv_sqrt2_f  = T.Cast(accum_dtype, inv_sqrt2)
        cmin_f       = T.Cast(accum_dtype, float(clamp_min))
        cmax_f       = T.Cast(accum_dtype, float(clamp_max))

        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total:
                w  = idx % W
                t1 = idx // W
                h  = t1 % H
                t2 = t1 // H
                d  = t2 % D
                t3 = t2 // D
                c  = t3 % C
                n  = t3 // C

                val = T.Cast(accum_dtype, X[n, c, d, h, w])

                # GELU
                val = half_f * val * (one_f + T.erf(val * inv_sqrt2_f))

                # Tanh
                val = T.tanh(val)

                # Clamp
                val = T.max(val, cmin_f)
                val = T.min(val, cmax_f)

                Out[n, c, d, h, w] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
#                               PyTorch wrapper                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d  →  fused( GELU → Tanh → Clamp ) implemented with TileLang.
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

        # Conv3d with default PyTorch initialisation (CRITICAL)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

        # Store clamp bounds
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # Kernel cache : key = (N,C,D,H,W,dtype)
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

        # Move conv params to CUDA / fp16 for fastest cuDNN path
        self.conv.to(device="cuda", dtype=torch.float16)

    # ------------------------------------------------------------------ #
    def _get_kernel(self, shape, dtype: str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_act_kernel(
                N,
                C,
                D,
                H,
                W,
                self.clamp_min,
                self.clamp_max,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Input to CUDA / fp16
        x_fp16 = x.to(device="cuda", dtype=torch.float16)

        # Conv3d (cuDNN)
        y_fp16 = self.conv(x_fp16).contiguous()

        # Fused element-wise kernel
        N, C, D, H, W = y_fp16.shape
        kernel = self._get_kernel((N, C, D, H, W), "float16")
        out_fp16 = kernel(y_fp16)

        return out_fp16.to(orig_dtype)