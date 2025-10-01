"""
Problem Name: 45_ConvTranspose3d_Swish_Clamp_Softmax_BiasAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=5.14 runtime_stats={'mean': 5.14, 'std': 0.0401, 'min': 5.12, 'max': 5.5, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 6.04, 'std': 0.0534, 'min': 5.99, 'max': 6.37, 'num_trials': 100}, 'speedup_ratio': 1.18}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :  Swish → clamp → softmax(dim=1) → +bias          #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    spatial = N * D * H * W
    grid = (spatial + threads_per_block - 1) // threads_per_block
    clamp_lo = -1.0
    clamp_hi = 1.0
    one_f = 1.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:    T.Tensor((N, C, D, H, W), dtype),   # input from ConvT3d
        Bias: T.Tensor((C,), dtype),              # (C,)  – broadcast
        Out:  T.Tensor((N, C, D, H, W), dtype),   # output
    ):
        lo  = T.Cast(accum_dtype, clamp_lo)
        hi  = T.Cast(accum_dtype, clamp_hi)
        one = T.Cast(accum_dtype, one_f)

        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < spatial:
                w  = idx % W
                tmp = idx // W
                h  = tmp % H
                tmp //= H
                d  = tmp % D
                n  = tmp // D

                # ---------------- pass-1 : accumulate sum_exp --------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0)
                for c in T.serial(C):
                    v = T.Cast(accum_dtype, X[n, c, d, h, w])
                    sig = one / (one + T.exp(-v))           # sigmoid
                    v = v * sig                             # swish
                    v = T.max(v, lo)
                    v = T.min(v, hi)                        # clamp
                    sum_exp[0] += T.exp(v)

                inv_sum = one / sum_exp[0]

                # ---------------- pass-2 : write results -------------------
                for c in T.serial(C):
                    v = T.Cast(accum_dtype, X[n, c, d, h, w])
                    sig = one / (one + T.exp(-v))
                    v = v * sig
                    v = T.max(v, lo)
                    v = T.min(v, hi)
                    prob = T.exp(v) * inv_sum               # softmax prob
                    out_val = prob + T.Cast(accum_dtype, Bias[c])
                    Out[n, c, d, h, w] = T.Cast(dtype, out_val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper                                                             #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  fused TileLang kernel:
        Swish → clamp[-1,1] → softmax(dim=1) → +bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias_shape: tuple,
    ):
        super().__init__()

        # ----------- ConvTranspose3d learnable parameters -----------------
        wt_shape = (in_channels,
                    out_channels,
                    kernel_size,
                    kernel_size,
                    kernel_size)
        self.weight = nn.Parameter(torch.empty(wt_shape))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size ** 3
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # Bias added after softmax
        self.bias = nn.Parameter(torch.randn(bias_shape).view(-1))  # (C,)

        # Hyper-params (defaults identical to nn.ConvTranspose3d)
        self.stride = 1
        self.padding = 0
        self.output_padding = 0
        self.dilation = 1

        # Kernel cache  {(N,C,D,H,W,dtype): compiled_kernel}
        self._kern_cache: Dict[Tuple, callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        # ---------------- ConvTranspose3d (cuDNN) -----------------------
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b1_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16)
        y = F.conv_transpose3d(
            x_fp16,
            w_fp16,
            b1_fp16,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
        ).contiguous()

        N, C, D, H, W = y.shape

        # ---------------- fused TileLang kernel -------------------------
        ker = self._get_kernel(N, C, D, H, W, "float16")
        bias2_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()
        out_fp16 = ker(y, bias2_fp16)

        return out_fp16.to(orig_dtype)