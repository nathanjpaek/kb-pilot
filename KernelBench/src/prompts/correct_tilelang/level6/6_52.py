"""
Problem Name: 52_ConvTranspose3d_Clamp_Sigmoid_Multiply
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.92 runtime_stats={'mean': 4.92, 'std': 0.0111, 'min': 4.91, 'max': 5.02, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.96, 'std': 0.0184, 'min': 4.95, 'max': 5.07, 'num_trials': 100}, 'speedup_ratio': 1.01}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory    clamp[-1,1] → sigmoid → ×2                                #
# --------------------------------------------------------------------------- #
def _build_post_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    clamp_min: float,
    clamp_max: float,
    scale: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT = N * C * D * H * W
    GRID = (TOT + block_size - 1) // block_size

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, C, D, H, W), dtype),
    ):
        cmin = T.Cast(accum_dtype, clamp_min)
        cmax = T.Cast(accum_dtype, clamp_max)
        one  = T.Cast(accum_dtype, 1.0)
        two  = T.Cast(accum_dtype, scale)

        with T.Kernel(GRID, threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOT:
                w  = idx % W
                tmp = idx // W
                h  = tmp % H
                tmp //= H
                d  = tmp % D
                tmp //= D
                c  = tmp % C
                n  = tmp // C

                val = T.Cast(accum_dtype, X[n, c, d, h, w])
                val = T.max(val, cmin)
                val = T.min(val, cmax)
                sig = one / (one + T.exp(-val))
                out = sig * two
                Y[n, c, d, h, w] = T.Cast(dtype, out)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with TileLang post-processing                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  clamp[-1,1] → sigmoid → ×2   (fused in TileLang)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        # -------- ConvTranspose3d parameters (identical init) ---------------
        w_shape = (
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(w_shape))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size ** 3
        bound  = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Hyper-parameters
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = 0
        self.output_padding = 0

        self.clamp_min = -1.0
        self.clamp_max = 1.0
        self.scale     = 2.0

        # Kernel cache  {(N,C,D,H,W,dtype): kernel}
        self._kern_cache: Dict[Tuple, callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_post_kernel(
                N,
                C,
                D,
                H,
                W,
                self.clamp_min,
                self.clamp_max,
                self.scale,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16)

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        y = F.conv_transpose3d(
            x_fp16,
            w_fp16,
            b_fp16,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=1,
        )

        N, C, D, H, W = y.shape
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        out_fp16 = kernel(y.contiguous())

        return out_fp16.to(orig_dtype)