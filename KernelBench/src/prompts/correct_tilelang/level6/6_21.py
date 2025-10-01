"""
Problem Name: 21_Conv2d_Sigmoid_LogSumExp_Scale
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.95 runtime_stats={'mean': 0.95, 'std': 0.00823, 'min': 0.943, 'max': 1.03, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.23, 'std': 0.00764, 'min': 1.22, 'max': 1.3, 'num_trials': 100}, 'speedup_ratio': 1.29}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : sigmoid → logsumexp_C → *scale                    #
# --------------------------------------------------------------------------- #
def _build_post_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    scale_factor: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    spatial = N * H * W
    minus_inf = -1.0e30

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),        # conv output
        Y: T.Tensor((N, 1, H, W), dtype),        # final tensor
    ):
        neg_big = T.Cast(accum_dtype, minus_inf)
        one     = T.Cast(accum_dtype, 1.0)
        sfactor = T.Cast(accum_dtype, scale_factor)

        with T.Kernel(T.ceildiv(spatial, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < spatial:
                w = idx % W
                tmp = idx // W
                h = tmp % H
                n = tmp // H

                # ---------- pass-1 : max over channels of sigmoid(x) ----------
                max_val = T.alloc_local((1,), accum_dtype)
                max_val[0] = neg_big
                for c in T.serial(C):
                    v = T.Cast(accum_dtype, X[n, c, h, w])
                    sig = one / (one + T.exp(-v))
                    if sig > max_val[0]:
                        max_val[0] = sig

                # ---------- pass-2 : sum(exp(sigmoid(x) - max)) ---------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0.0)
                for c in T.serial(C):
                    v = T.Cast(accum_dtype, X[n, c, h, w])
                    sig = one / (one + T.exp(-v))
                    sum_exp[0] += T.exp(sig - max_val[0])

                lse = max_val[0] + T.log(sum_exp[0])
                out_val = lse * sfactor
                Y[n, 0, h, w] = T.Cast(dtype, out_val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper                                                             #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d  →  fused TileLang(kernel):  sigmoid → logsumexp(C,keepdim) → *scale
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: float,
    ):
        super().__init__()

        # -------- Conv2d parameters with identical initialisation ----------
        w_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.uniform_(self.bias, -bound, bound)

        # Scalar
        self.scale_factor = float(scale_factor)

        # Kernel cache : {(N,H,W,dtype): kernel}
        self._kern_cache: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_post_kernel(
                N, C, H, W, self.scale_factor, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Move to CUDA + fp16
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        # ------------------------ convolution --------------------------- #
        y = F.conv2d(x_fp16, w_fp16, b_fp16, stride=1, padding=0)  # (N,C,H,W)
        N, C, H, W = y.shape
        y = y.contiguous()

        # -------------------- fused TileLang kernel --------------------- #
        kernel = self._get_kernel(N, C, H, W, "float16")
        out_fp16 = kernel(y)               # (N,1,H,W)

        return out_fp16.to(orig_dtype)