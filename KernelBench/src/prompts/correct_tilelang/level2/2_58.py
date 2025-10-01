"""
Problem Name: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=21.3 runtime_stats={'mean': 21.3, 'std': 0.0182, 'min': 21.3, 'max': 21.4, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 24.2, 'std': 0.00921, 'min': 24.2, 'max': 24.3, 'num_trials': 100}, 'speedup_ratio': 1.14}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory                                                     #
# --------------------------------------------------------------------------- #
def _build_postproc_kernel(
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
    three_c = T.Cast(accum_dtype, 3.0)
    six_c   = T.Cast(accum_dtype, 6.0)
    one_c   = T.Cast(accum_dtype, 1.0)
    clamp_lo= T.Cast(accum_dtype, -1.0)
    clamp_hi= T.Cast(accum_dtype,  1.0)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:    T.Tensor((N, C, D, H, W), dtype),   # conv-transpose output
        Bias: T.Tensor((C,),           dtype),    # bias to subtract
        Y:    T.Tensor((N, 1, D, H, W), dtype),   # final result
    ):
        with T.Kernel(T.ceildiv(spatial, threads_per_block),
                      threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < spatial:
                w  = idx % W
                t1 = idx // W
                h  = t1 % H
                t2 = t1 // H
                d  = t2 % D
                n  = t2 // D

                # ------------------- pass-1 : channel-wise max --------------
                max_val = T.alloc_local((1,), accum_dtype)
                max_val[0] = T.Cast(accum_dtype, X[n, 0, d, h, w])
                for c in T.serial(1, C):        # starting from 1
                    v = T.Cast(accum_dtype, X[n, c, d, h, w])
                    if v > max_val[0]:
                        max_val[0] = v

                # ------------------- pass-2 : sum(exp(x-max)) ---------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0)
                for c in T.serial(C):
                    v = T.Cast(accum_dtype, X[n, c, d, h, w])
                    sum_exp[0] += T.exp(v - max_val[0])

                lse = max_val[0] + T.log(sum_exp[0])

                # -------------- HardSwish  : x * sigmoid(x+3)/6 -------------
                tmp   = lse + three_c
                sigm  = one_c / (one_c + T.exp(-tmp))
                hs    = lse * sigm / six_c      # scalar value (shared for all C)

                # ------------------- pass-3 : max over channels -------------
                out_max = T.alloc_local((1,), accum_dtype)
                out_max[0] = clamp_lo          # lowest possible after clamp

                for c in T.serial(C):
                    val = hs - T.Cast(accum_dtype, Bias[c])
                    val = T.max(val, clamp_lo)
                    val = T.min(val, clamp_hi)
                    if val > out_max[0]:
                        out_max[0] = val

                # --------------------------- store --------------------------
                Y[n, 0, d, h, w] = T.Cast(dtype, out_max[0])

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper                                                             #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d → fused (LogSumExp → HardSwish → ‒bias → clamp[-1,1] →
    max over channels) implemented with TileLang.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias_shape: tuple,
    ):
        super().__init__()

        # ---------------- ConvTranspose3d (identical init) -----------------
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        # extra bias (same as original)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # kernel cache : {(N,D,H,W,dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            C = self.conv_transpose.out_channels
            self._kern_cache[key] = _build_postproc_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # -------------------- conv-transpose (cuDNN) ---------------------
        y = self.conv_transpose(x)                     # (N,C,D,H,W)
        y_fp16 = y.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, D, H, W = y_fp16.shape

        # -------------------- fused TileLang kernel ----------------------
        kernel = self._get_kernel(N, D, H, W, "float16")
        bias_fp16 = self.bias.view(-1).to(
            device="cuda", dtype=torch.float16
        ).contiguous()

        out_fp16 = kernel(y_fp16, bias_fp16)           # (N,1,D,H,W)

        return out_fp16.to(orig_dtype)