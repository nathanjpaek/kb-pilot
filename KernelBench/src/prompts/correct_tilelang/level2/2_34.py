"""
Problem Name: 34_ConvTranspose3d_LayerNorm_GELU_Scaling
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=35.4 runtime_stats={'mean': 35.4, 'std': 0.0916, 'min': 35.4, 'max': 35.8, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 35.4, 'std': 0.0322, 'min': 35.4, 'max': 35.6, 'num_trials': 100}, 'speedup_ratio': 1.0}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : GELU + scaling                                    #
# --------------------------------------------------------------------------- #
def _build_gelu_scale_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    scaling_factor: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total = N * C * D * H * W
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gelu_scale(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, C, D, H, W), dtype),
    ):
        half_f       = T.Cast(accum_dtype, 0.5)
        inv_sqrt2_f  = T.Cast(accum_dtype, inv_sqrt2)
        scale_f      = T.Cast(accum_dtype, float(scaling_factor))

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

                x_val_f32 = T.Cast(accum_dtype, X[n, c, d_, h, w])

                gelu = half_f * x_val_f32 * (
                    T.Cast(accum_dtype, 1.0) +
                    T.erf(x_val_f32 * inv_sqrt2_f)
                )

                out_val = gelu * scale_f
                Y[n, c, d_, h, w] = T.Cast(dtype, out_val)

    return gelu_scale


# --------------------------------------------------------------------------- #
# Optimised PyTorch module                                                    #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  LayerNorm  →  GELU * scale  (fused TileLang)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        bias: bool = True,
        eps: float = 1e-5,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        # Keep original ConvTranspose3d (weights/bias initialised by PyTorch)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Same LayerNorm as reference model
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)

        self.scaling_factor = float(scaling_factor)

        # kernel cache : (N,C,D,H,W,dtype) -> compiled kernel
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # --------------------------------------------------------------------- #
    def _get_kernel(self, shape, dtype_str: str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype_str)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_gelu_scale_kernel(
                N,
                C,
                D,
                H,
                W,
                self.scaling_factor,
                dtype=dtype_str,
            )
        return self._kern_cache[key]

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvTranspose3d (FP32 for accuracy)
        x = self.conv_transpose(x)

        # LayerNorm (FP32)
        x = self.layer_norm(x)

        # Move to CUDA / FP16 for fused kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_fp16.shape

        kernel = self._get_kernel((N, C, D, H, W), "float16")
        y_fp16 = kernel(x_fp16)

        # Return in original dtype
        return y_fp16.to(x.dtype)