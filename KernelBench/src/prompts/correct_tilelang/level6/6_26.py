"""
Problem Name: 26_Conv3d_GELU_Clamp_Tanh
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.532 runtime_stats={'mean': 0.532, 'std': 0.0109, 'min': 0.522, 'max': 0.633, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.554, 'std': 0.00767, 'min': 0.546, 'max': 0.624, 'num_trials': 100}, 'speedup_ratio': 1.04}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                   TileLang kernel factory  GELU → clamp → tanh              #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
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
        X:   T.Tensor((N, C, D, H, W), dtype),    # Conv3d output
        Out: T.Tensor((N, C, D, H, W), dtype),    # final result
    ):
        half_f       = T.Cast(accum_dtype, 0.5)
        one_f        = T.Cast(accum_dtype, 1.0)
        inv_sqrt2_f  = T.Cast(accum_dtype, inv_sqrt2)
        clamp_min_f  = T.Cast(accum_dtype, float(clamp_min))
        clamp_max_f  = T.Cast(accum_dtype, float(clamp_max))

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

                # GELU (erf formulation)
                val = half_f * val * (one_f + T.erf(val * inv_sqrt2_f))

                # clamp
                val = T.max(val, clamp_min_f)
                val = T.min(val, clamp_max_f)

                # tanh
                val = T.tanh(val)

                Out[n, c, d, h, w] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
#                              PyTorch wrapper                                #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d → fused(GELU → Clamp → Tanh) implemented with TileLang.
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

        # Conv3d layer (default PyTorch initialisation is correct)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # move weights to GPU / fp16 for performance
        self.conv.to(device="cuda", dtype=torch.float16)

        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # kernel cache : key = (N,C,D,H,W,dtype)
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, shape, dtype: str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
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

        # input to CUDA/fp16
        x_fp16 = x.to(device="cuda", dtype=torch.float16)

        # Conv3d (cuDNN)
        y_fp16 = self.conv(x_fp16).contiguous()

        # fused element-wise kernel
        N, C, D, H, W = y_fp16.shape
        kernel = self._get_kernel((N, C, D, H, W), "float16")
        out_fp16 = kernel(y_fp16)

        return out_fp16.to(orig_dtype)