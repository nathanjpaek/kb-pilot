"""
Problem Name: 40_ConvTranspose3d_Scaling_Softmax_InstanceNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.0 runtime_stats={'mean': 4.0, 'std': 0.00847, 'min': 3.99, 'max': 4.07, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.51, 'std': 0.0131, 'min': 4.5, 'max': 4.6, 'num_trials': 100}, 'speedup_ratio': 1.13}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :   softmax(dim=1)                                  #
# --------------------------------------------------------------------------- #
def _build_softmax_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    spatial = N * D * H * W
    grid = (spatial + block - 1) // block
    one_f = 1.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, D, H, W), dtype),   # in
        Out: T.Tensor((N, C, D, H, W), dtype),   # out
    ):
        one_c = T.Cast(accum_dtype, one_f)

        with T.Kernel(grid, threads=block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < spatial:
                w  = idx % W
                tmp = idx // W
                h  = tmp % H
                tmp //= H
                d  = tmp % D
                n  = tmp // D

                # ---------- pass 1 : denominator -------------------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0)
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    sum_exp[0] += T.exp(val)

                inv_sum = one_c / sum_exp[0]

                # ---------- pass 2 : final output ------------------------
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    sm  = T.exp(val) * inv_sum
                    Out[n, c, d, h, w] = T.Cast(dtype, sm)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper                                                             #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  trilinear up-sample  →  softmax(dim=1) [TileLang] →
    InstanceNorm3d   (computed manually for inference, affine=False)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        # ---------------- ConvTranspose3d parameters ----------------------
        wt_shape = (
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(wt_shape))
        self.bias   = nn.Parameter(torch.empty(out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size ** 3
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # ----- hyper-parameters -----------------------------------------
        self.scale_factor = int(scale_factor)
        self.eps          = float(eps)

        # ----- kernel cache :  {(N,C,D,H,W,dtype) : compiled_kernel} -----
        self._kern_cache: Dict[Tuple, callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_softmax_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16)

        # -------- ConvTranspose3d ---------------------------------------
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)
        y = F.conv_transpose3d(x_fp16, w_fp16, b_fp16)

        # -------- trilinear up-sample -----------------------------------
        y = F.interpolate(
            y,
            scale_factor=self.scale_factor,
            mode="trilinear",
            align_corners=False,
        ).contiguous()                                         # (N,C,D,H,W)

        N, C, D, H, W = y.shape

        # -------- fused softmax kernel ----------------------------------
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        y_sm = kernel(y)

        # -------- InstanceNorm3d (affine=False) -------------------------
        mean = y_sm.mean(dim=(2, 3, 4), keepdim=True)
        var  = y_sm.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
        y_norm = (y_sm - mean) / torch.sqrt(var + self.eps)

        return y_norm.to(orig_dtype)