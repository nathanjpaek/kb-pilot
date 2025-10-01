"""
Problem Name: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=21.2 runtime_stats={'mean': 21.2, 'std': 0.0149, 'min': 21.2, 'max': 21.3, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 21.7, 'std': 0.0221, 'min': 21.6, 'max': 21.7, 'num_trials': 100}, 'speedup_ratio': 1.02}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :  softmax(dim=1) → subtract → swish → max(dim=1)  #
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
    zero_f = 0.0
    one_f = 1.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, D, H, W), dtype),     # pooled activations
        Sub: T.Tensor((C,), dtype),                # subtract vector
        Out: T.Tensor((N, D, H, W), dtype),        # final result
    ):
        zero_c = T.Cast(accum_dtype, zero_f)
        one_c  = T.Cast(accum_dtype, one_f)

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

                # -------------------- pass-1 : sum(exp) --------------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = zero_c
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    sum_exp[0] += T.exp(val)

                inv_sum = one_c / sum_exp[0]

                # ------------- pass-2 : swish & max across C ---------------
                max_val = T.alloc_local((1,), accum_dtype)
                max_val[0] = T.Cast(accum_dtype, -3.4e38)   # −FLT_MAX

                for c in T.serial(C):
                    val   = T.Cast(accum_dtype, X[n, c, d, h, w])
                    prob  = T.exp(val) * inv_sum                # softmax
                    s     = prob - T.Cast(accum_dtype, Sub[c])  # subtract
                    sig   = one_c / (one_c + T.exp(-s))         # sigmoid
                    g     = s * sig                             # swish
                    if g > max_val[0]:
                        max_val[0] = g

                Out[n, d, h, w] = T.Cast(dtype, max_val[0])

    return kernel


# --------------------------------------------------------------------------- #
# Optimised PyTorch module                                                    #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d → MaxPool3d (PyTorch)  →  fused TileLang kernel:
        softmax(dim=1) – subtract – swish – max(dim=1)
    Output shape : (N, D, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        pool_kernel_size: int,
        pool_stride: int,
        pool_padding: int,
    ):
        super().__init__()

        # ---------------- ConvTranspose3d parameters ----------------------
        w_shape = (in_channels,
                   out_channels,
                   kernel_size,
                   kernel_size,
                   kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias   = nn.Parameter(torch.empty(out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size ** 3
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---------------- MaxPool3d ---------------------------------------
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride      = pool_stride
        self.pool_padding     = pool_padding

        # ---------------- subtract parameter ------------------------------
        self.subtract = nn.Parameter(torch.randn(out_channels))

        # ---------------- kernel cache ------------------------------------
        self._kern_cache: Dict[Tuple, callable] = {}

        # store conv hyper-params
        self.stride         = stride
        self.padding        = padding
        self.output_padding = output_padding

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

        # ------------------ ConvTranspose3d (cuDNN) ----------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        y = F.conv_transpose3d(
            x_fp16,
            w_fp16,
            b_fp16,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )

        # -------------------- MaxPool3d (cuDNN) --------------------------
        y = F.max_pool3d(
            y,
            kernel_size=self.pool_kernel_size,
            stride=self.pool_stride,
            padding=self.pool_padding,
        ).contiguous()

        N, C, D, H, W = y.shape

        # ------------------- fused TileLang kernel -----------------------
        sub_fp16 = self.subtract.to(device="cuda", dtype=torch.float16).contiguous()
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        out_fp16 = kernel(y, sub_fp16)       # (N,D,H,W)

        return out_fp16.to(orig_dtype)