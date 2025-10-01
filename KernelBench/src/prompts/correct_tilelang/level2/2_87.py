"""
Problem Name: 87_Conv2d_Subtract_Subtract_Mish
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.059 runtime_stats={'mean': 0.059, 'std': 0.0171, 'min': 0.0498, 'max': 0.207, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.062, 'std': 0.0181, 'min': 0.054, 'max': 0.225, 'num_trials': 100}, 'speedup_ratio': 1.05}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory
# --------------------------------------------------------------------------- #
def _build_sub_mish_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    sub_val: float,
    block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOTAL = N * C * H * W
    sub_const = float(sub_val)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, H, W), dtype),          # convolution output
        Out: T.Tensor((N, C, H, W), dtype),          # final result
    ):
        sub_v  = T.Cast(accum_dtype, sub_const)
        one_v  = T.Cast(accum_dtype, 1.0)

        with T.Kernel(T.ceildiv(TOTAL, block), threads=block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < TOTAL:
                w  = idx % W
                tmp = idx // W
                h  = tmp % H
                tmp //= H
                c  = tmp % C
                n  = tmp // C

                val = T.Cast(accum_dtype, X[n, c, h, w])
                val = val - sub_v

                sp  = T.log(one_v + T.exp(val))
                val = val * T.tanh(sp)          # Mish

                Out[n, c, h, w] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → (−sub1 − sub2) → Mish   (fused TileLang kernel for element-wise ops)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        subtract_value_1: float,
        subtract_value_2: float,
    ):
        super().__init__()

        # ---- Conv2d parameters (identical initialisation) -----------------
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---- combined subtraction constant --------------------------------
        self.sub_val = float(subtract_value_1 + subtract_value_2)

        # ---- kernel cache --------------------------------------------------
        self._kernels: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_sub_mish_kernel(
                N,
                self.weight.shape[0],   # out_channels = C after conv
                H,
                W,
                self.sub_val,
                dtype=dtype,
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---------------- convolution in fp16 ------------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        y = F.conv2d(x_fp16, w_fp16, b_fp16, stride=1, padding=0)

        # ---------------- fused subtract + Mish kernel ---------------------
        N, C, H, W = y.shape
        kernel = self._get_kernel(N, H, W, "float16")
        out_fp16 = kernel(y.contiguous())

        return out_fp16.to(orig_dtype)