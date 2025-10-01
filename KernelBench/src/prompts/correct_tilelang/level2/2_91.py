"""
Problem Name: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.132 runtime_stats={'mean': 0.132, 'std': 0.026, 'min': 0.124, 'max': 0.386, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.22, 'std': 0.0146, 'min': 0.212, 'max': 0.357, 'num_trials': 100}, 'speedup_ratio': 1.67}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : softmax(dim=1) + bias + scale + sigmoid
# --------------------------------------------------------------------------- #
def _build_softmax_bias_scale_sigmoid_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    scale_val: float,
    block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    spatial = N * H * W
    grid = (spatial + block - 1) // block
    one_f = 1.0
    scale_f = float(scale_val)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:  T.Tensor((N, C, H, W), dtype),
        B:  T.Tensor((C,),           dtype),   # bias after softmax
        Out: T.Tensor((N, C, H, W), dtype),
    ):
        one_c   = T.Cast(accum_dtype, one_f)
        scale_c = T.Cast(accum_dtype, scale_f)

        with T.Kernel(grid, threads=block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < spatial:
                w  = idx % W
                tmp= idx // W
                h  = tmp % H
                n  = tmp // H

                # ---------------- pass 1 : denominator --------------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0)
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, h, w])
                    sum_exp[0] += T.exp(val)

                inv_sum = one_c / sum_exp[0]

                # ------------- pass 2 : output computation ----------------
                for c in T.serial(C):
                    x_val = T.Cast(accum_dtype, X[n, c, h, w])
                    softmax_val = T.exp(x_val) * inv_sum
                    y = (softmax_val + T.Cast(accum_dtype, B[c])) * scale_c
                    sig = one_c / (one_c + T.exp(-y))
                    Out[n, c, h, w] = T.Cast(dtype, sig)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused TileLang kernel                                 #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d  →  softmax(dim=1) → +bias → *scale → sigmoid
    The first op remains in cuDNN; the rest is a fused TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        bias_shape: tuple,
        scaling_factor: float,
    ):
        super().__init__()

        # ---------------- ConvTranspose2d parameters ----------------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # Additional bias after softmax
        self.bias = nn.Parameter(torch.randn(bias_shape).view(-1))

        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)
        self.scale = float(scaling_factor)

        # Kernel cache : {(N,C,H,W,dtype) : kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_softmax_bias_scale_sigmoid_kernel(
                N, C, H, W, self.scale, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---------------- ConvTranspose2d (cuDNN) ------------------------
        w = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_conv = self.conv_bias.to(device="cuda", dtype=torch.float16).contiguous()
        x = x.to(device="cuda", dtype=torch.float16)
        y = F.conv_transpose2d(
            x, w, b_conv,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )

        N, C, H, W = y.shape
        kernel = self._get_kernel(N, C, H, W, "float16")

        bias_post = self.bias.to(device="cuda", dtype=torch.float16).contiguous().view(-1)

        out_fp16 = kernel(y, bias_post)
        return out_fp16.to(orig_dtype)