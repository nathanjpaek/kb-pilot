"""
Problem Name: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.409 runtime_stats={'mean': 0.409, 'std': 0.00375, 'min': 0.402, 'max': 0.417, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.472, 'std': 0.0102, 'min': 0.462, 'max': 0.552, 'num_trials': 100}, 'speedup_ratio': 1.15}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                         Kernel Factory                                      #
# --------------------------------------------------------------------------- #
def _build_mean_bias_softmax_tanh_scale_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    scale_val: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    elems = N * D * H * W
    grid = (elems + threads_per_block - 1) // threads_per_block
    c_inv = 1.0 / float(C)
    two_f = 2.0
    scale_f = float(scale_val)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D, H, W), dtype),    # conv-output
        B: T.Tensor((1,), dtype),               # scalar bias
        Out: T.Tensor((N, 1, D, H, W), dtype),  # final result
    ):
        cinv_c  = T.Cast(accum_dtype, c_inv)
        two_c   = T.Cast(accum_dtype, two_f)
        scale_c = T.Cast(accum_dtype, scale_f)

        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < elems:
                w  = idx % W
                tmp = idx // W
                h  = tmp % H
                tmp //= H
                d  = tmp % D
                n  = tmp // D

                # ---------------------- mean over channels -----------------
                mval = T.alloc_local((1,), accum_dtype)
                mval[0] = T.Cast(accum_dtype, 0)

                for c in T.serial(C):
                    mval[0] += T.Cast(accum_dtype, X[n, c, d, h, w])

                mean_val = mval[0] * cinv_c

                # --------------------------- + bias ------------------------
                bias_v = T.Cast(accum_dtype, B[0])
                val = mean_val + bias_v

                # -------- softmax(dim=1) with single channel => 1 ----------
                softmax_val = T.Cast(accum_dtype, 1.0)

                # -------------------------- tanh ---------------------------
                expv = T.exp(-two_c * softmax_val)
                tanh_v = (T.Cast(accum_dtype, 1.0) - expv) / (
                    T.Cast(accum_dtype, 1.0) + expv
                )

                # --------------------------- scale -------------------------
                out_v = tanh_v * scale_c
                Out[n, 0, d, h, w] = T.Cast(dtype, out_v)

    return kernel


# --------------------------------------------------------------------------- #
#                         PyTorch Wrapper                                     #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  mean(C)  → +bias  → softmax(dim=1)  → tanh  → *scale
    The first op is done by cuDNN; the rest is fused in a TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias_shape: tuple,
        scaling_factor: float,
    ):
        super().__init__()

        # ConvTranspose3d parameters (same init as PyTorch defaults)
        w_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * (kernel_size ** 3)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # Scalar bias added after mean
        self.bias = nn.Parameter(torch.randn(bias_shape).view(1))

        self.stride = int(stride)
        self.padding = int(padding)
        self.scale = float(scaling_factor)

        # kernel cache : {(N,C,D,H,W,dtype): kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self, N: int, C: int, D: int, H: int, W: int, dtype: str = "float16"
    ):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_mean_bias_softmax_tanh_scale_kernel(
                N, C, D, H, W, self.scale, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        # ---------------- ConvTranspose3d -------------------------------
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16)

        y = F.conv_transpose3d(
            x_fp16,
            w_fp16,
            b_fp16,
            stride=self.stride,
            padding=self.padding,
        ).contiguous()

        N, C, D, H, W = y.shape

        # ------------------ Fused TileLang kernel ------------------------
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        bias_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()
        out_fp16 = kernel(y, bias_fp16)  # (N,1,D,H,W)

        return out_fp16.to(orig_dtype)