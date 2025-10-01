"""
Problem Name: 1_Conv2D_ReLU_BiasAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0692 runtime_stats={'mean': 0.0692, 'std': 0.0085, 'min': 0.0612, 'max': 0.102, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0577, 'std': 0.00579, 'min': 0.0522, 'max': 0.0804, 'num_trials': 100}, 'speedup_ratio': 0.834}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory:  ReLU  +  bias-add (bias shape = (C,))                      #
# --------------------------------------------------------------------------- #
def _build_relu_add_bias_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total = N * C * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),
        B: T.Tensor((C,),          dtype),
        Y: T.Tensor((N, C, H, W),  dtype),
    ):
        zero_f = T.Cast(accum_dtype, 0.0)

        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total:
                w  = idx % W
                t1 = idx // W
                h  = t1 % H
                t2 = t1 // H
                c  = t2 % C
                n  = t2 // C

                val = T.Cast(accum_dtype, X[n, c, h, w])
                val = T.max(val, zero_f)                       # ReLU
                val = val + T.Cast(accum_dtype, B[c])          # + bias

                Y[n, c, h, w] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused TileLang kernel                                  #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d  →  ReLU  →  +bias   (ReLU & bias fused in TileLang)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias_shape):
        super().__init__()
        self.C_in   = int(in_channels)
        self.C_out  = int(out_channels)
        self.K      = int(kernel_size)

        # -------- Conv2d parameters (identical init to nn.Conv2d) ----------
        self.weight = nn.Parameter(
            torch.empty(self.C_out, self.C_in, self.K, self.K)
        )
        self.conv_bias = nn.Parameter(torch.empty(self.C_out))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = self.C_in * self.K * self.K
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # -------- Post-ReLU bias (same init as original code) --------------
        self.extra_bias = nn.Parameter(torch.randn(bias_shape))

        # -------- Kernel cache --------------------------------------------
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_relu_add_bias_kernel(
                N, C, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # --------------- Conv2d (cuDNN) ---------------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16).contiguous()

        conv_out = F.conv2d(x_fp16, w_fp16, b_fp16, stride=1, padding=0).contiguous()

        # --------------- Fused ReLU + bias ------------------------------
        N, C, H, W = conv_out.shape
        kernel = self._get_kernel(N, C, H, W, "float16")

        extra_b_fp16 = self.extra_bias.to(device="cuda", dtype=torch.float16).view(-1).contiguous()
        out_fp16 = kernel(conv_out, extra_b_fp16)

        return out_fp16.to(orig_dtype)