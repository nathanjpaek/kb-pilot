"""
Problem Name: 64_ConvTranspose2d_GELU_Scale_Add_Add
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.264 runtime_stats={'mean': 0.264, 'std': 0.0175, 'min': 0.256, 'max': 0.43, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.373, 'std': 0.0231, 'min': 0.364, 'max': 0.597, 'num_trials': 100}, 'speedup_ratio': 1.41}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : GELU → *scale + bias  →  +2*residual              #
# --------------------------------------------------------------------------- #

def _build_post_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT = N * C * H * W
    GRID = (TOT + threads_per_block - 1) // threads_per_block
    INV_SQRT2 = 0.7071067811865476  # 1 / sqrt(2)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:       T.Tensor((N, C, H, W), dtype),   # conv-transpose output
        R:       T.Tensor((N, C, H, W), dtype),   # residual
        Scale:   T.Tensor((1,), dtype),           # scalar multiplier
        Bias:    T.Tensor((1,), dtype),           # scalar add
        Out:     T.Tensor((N, C, H, W), dtype),   # final output
    ):
        half_f      = T.Cast(accum_dtype, 0.5)
        inv_sqrt2_f = T.Cast(accum_dtype, INV_SQRT2)
        two_f       = T.Cast(accum_dtype, 2.0)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                w   = idx % W
                t1  = idx // W
                h   = t1 % H
                t2  = t1 // H
                c   = t2 % C
                n   = t2 // C

                x_val_f32 = T.Cast(accum_dtype, X[n, c, h, w])
                res_f32   = T.Cast(accum_dtype, R[n, c, h, w])

                # GELU
                gelu = half_f * x_val_f32 * (
                    T.Cast(accum_dtype, 1.0) + T.erf(x_val_f32 * inv_sqrt2_f)
                )

                scale_f = T.Cast(accum_dtype, Scale[0])
                bias_f  = T.Cast(accum_dtype, Bias[0])

                out_val = gelu * scale_f + bias_f + two_f * res_f32
                Out[n, c, h, w] = T.Cast(dtype, out_val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused TileLang kernel                                  #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d → GELU → *scale + bias → +residual → +residual (fused)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: float = 2.0,  # unused but kept for API parity
    ):
        super().__init__()
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size  = int(kernel_size)
        self.stride       = 2
        self.padding      = 1
        self.output_pad   = 1

        # ---------------- ConvTranspose2d parameters ----------------------
        w_shape = (self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound  = 1 / math.sqrt(fan_in)
        self.conv_bias = nn.Parameter(torch.empty(self.out_channels))
        nn.init.uniform_(self.conv_bias, -bound, bound)

        # ---------------- Extra learnable scalars -------------------------
        self.scale = nn.Parameter(torch.ones(1))  # multiplier
        self.bias  = nn.Parameter(torch.zeros(1))  # addend

        # Kernel cache  {(N,C,H,W,dtype): compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_post_kernel(N, C, H, W, dtype=dtype)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ------------------- ConvTranspose2d -----------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16)

        y = F.conv_transpose2d(
            x_fp16,
            w_fp16,
            b_fp16,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_pad,
        ).contiguous()  # shape (N,C,H,W)

        residual_fp16 = residual.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = y.shape
        kernel = self._get_kernel(N, C, H, W, "float16")

        scale_fp16 = self.scale.to(device="cuda", dtype=torch.float16).contiguous()
        bias_fp16  = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        out_fp16 = kernel(y, residual_fp16, scale_fp16, bias_fp16)
        return out_fp16.to(orig_dtype)