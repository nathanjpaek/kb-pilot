"""
Problem Name: 36_ConvTranspose2d_Min_Sum_GELU_Add
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.178 runtime_stats={'mean': 0.178, 'std': 0.00155, 'min': 0.175, 'max': 0.184, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.187, 'std': 0.00164, 'min': 0.184, 'max': 0.195, 'num_trials': 100}, 'speedup_ratio': 1.05}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : min(chan) ➜ sum(h) ➜ GELU ➜ +bias                #
# --------------------------------------------------------------------------- #
def _build_reduce_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT = N * C * W
    GRID = (TOT + threads_per_block - 1) // threads_per_block
    INV_SQRT2 = 0.7071067811865476  # 1 / sqrt(2)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),        # conv-transpose output
        B: T.Tensor((C,), dtype),                # extra bias (flattened)
        Out: T.Tensor((N, C, 1, W), dtype),      # final result
    ):
        half_f       = T.Cast(accum_dtype, 0.5)
        one_f        = T.Cast(accum_dtype, 1.0)
        inv_sqrt2_f  = T.Cast(accum_dtype, INV_SQRT2)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                w  = idx % W
                tmp = idx // W
                oc  = tmp % C
                n   = tmp // C

                # ------------------------------------------------------ #
                #   sum_{h} min_{c'} X[n, c', h, w]  (float32 accum)     #
                # ------------------------------------------------------ #
                acc_sum = T.alloc_local((1,), accum_dtype)
                acc_sum[0] = T.Cast(accum_dtype, 0)

                for h in T.serial(H):
                    mval = T.alloc_local((1,), accum_dtype)
                    # initialise with first channel
                    mval[0] = T.Cast(accum_dtype, X[n, 0, h, w])
                    for c_it in T.serial(C):
                        v = T.Cast(accum_dtype, X[n, c_it, h, w])
                        mval[0] = T.min(mval[0], v)
                    acc_sum[0] += mval[0]

                # ------------------ GELU --------------------------------- #
                gelu_val = (
                    acc_sum[0]
                    * half_f
                    * (one_f + T.erf(acc_sum[0] * inv_sqrt2_f))
                )

                out_val = gelu_val + T.Cast(accum_dtype, B[oc])
                Out[n, oc, 0, w] = T.Cast(dtype, out_val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with TileLang fusion                                       #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d  →  min(channel)  →  sum(height)  →  GELU  →  +bias   (fused)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        bias_shape: Tuple[int, int, int],
    ):
        super().__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.output_padding = output_padding

        # ---- ConvTranspose2d weights/bias (identical init) ----------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size * kernel_size
        bound  = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # ---- extra bias ---------------------------------------------------
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # ---- kernel cache -------------------------------------------------
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_reduce_kernel(N, C, H, W, dtype=dtype)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        cb_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16)

        # ---------------- ConvTranspose2d (cuDNN) -------------------------
        y = F.conv_transpose2d(
            x_fp16,
            w_fp16,
            cb_fp16,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        ).contiguous()

        N, C, H, W = y.shape
        kernel = self._get_kernel(N, C, H, W, "float16")

        # extra bias: flatten to (C,)
        b_fp16 = self.bias.view(-1).to(device="cuda", dtype=torch.float16).contiguous()

        out_fp16 = kernel(y, b_fp16)
        return out_fp16.to(orig_dtype)