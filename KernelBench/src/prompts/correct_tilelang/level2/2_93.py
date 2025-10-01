"""
Problem Name: 93_ConvTranspose2d_Add_Min_GELU_Multiply
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.159 runtime_stats={'mean': 0.159, 'std': 0.0106, 'min': 0.153, 'max': 0.25, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.189, 'std': 0.00955, 'min': 0.183, 'max': 0.272, 'num_trials': 100}, 'speedup_ratio': 1.19}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :  +add → min(0) → GELU → ×mul                     #
# --------------------------------------------------------------------------- #
def _build_post_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    add_const: float,
    mul_const: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT = N * C * H * W
    GRID = (TOT + threads_per_block - 1) // threads_per_block
    HALF_F = 0.5
    ONE_F = 1.0
    INV_SQRT2 = 0.7071067811865476  # 1 / sqrt(2)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),
        Y: T.Tensor((N, C, H, W), dtype),
    ):
        add_c = T.Cast(accum_dtype, add_const)
        mul_c = T.Cast(accum_dtype, mul_const)
        half_c = T.Cast(accum_dtype, HALF_F)
        one_c = T.Cast(accum_dtype, ONE_F)
        inv_sqrt2_c = T.Cast(accum_dtype, INV_SQRT2)
        zero_c = T.Cast(accum_dtype, 0.0)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                w = idx % W
                tmp1 = idx // W
                h = tmp1 % H
                tmp2 = tmp1 // H
                c = tmp2 % C
                n = tmp2 // C

                v = T.Cast(accum_dtype, X[n, c, h, w]) + add_c
                v = T.min(v, zero_c)                       # clamp max 0
                gelu = v * half_c * (one_c + T.erf(v * inv_sqrt2_c))
                out_val = gelu * mul_c
                Y[n, c, h, w] = T.Cast(dtype, out_val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused TileLang post-processing                         #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d → (+add_value) → min(0) → GELU → ×multiply_value
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        add_value: float,
        multiply_value: float,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)

        # ---- ConvTranspose2d parameters (identical init) ------------------
        w_shape = (self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---- Kernel cache -------------------------------------------------
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str = "float16"):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_post_kernel(
                N,
                C,
                H,
                W,
                self.add_value,
                self.multiply_value,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        # ------ ConvTranspose2d (cuDNN) -----------------------------------
        y = F.conv_transpose2d(x_fp16, w_fp16, b_fp16, stride=self.stride).contiguous()

        N, C, H, W = y.shape
        kernel = self._get_kernel(N, C, H, W, "float16")

        out_fp16 = kernel(y)
        return out_fp16.to(orig_dtype)