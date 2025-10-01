"""
Problem Name: 99_ConvTranspose2d_GELU_Add_Scale_Add
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.699 runtime_stats={'mean': 0.699, 'std': 0.00188, 'min': 0.694, 'max': 0.704, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.07, 'std': 0.00234, 'min': 1.06, 'max': 1.08, 'num_trials': 100}, 'speedup_ratio': 1.53}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : GELU → +add → *scale → +residual                  #
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
    INV_SQRT2 = 0.7071067811865476  # 1/√2

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:    T.Tensor((N, C, H, W), dtype),   # ConvTranspose output
        ADD:  T.Tensor((C,),           dtype), # per-channel add tensor
        SCALE:T.Tensor((1,),           dtype), # scalar multiplier
        RES:  T.Tensor((N, C, H, W),   dtype), # residual to add
        OUT:  T.Tensor((N, C, H, W),   dtype), # final output
    ):
        half_f       = T.Cast(accum_dtype, 0.5)
        one_f        = T.Cast(accum_dtype, 1.0)
        inv_sqrt2_f  = T.Cast(accum_dtype, INV_SQRT2)

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

                x_f32  = T.Cast(accum_dtype, X[n, c, h, w])
                res_f  = T.Cast(accum_dtype, RES[n, c, h, w])
                add_f  = T.Cast(accum_dtype, ADD[c])
                scale_f= T.Cast(accum_dtype, SCALE[0])

                # GELU
                gelu = half_f * x_f32 * (one_f + T.erf(x_f32 * inv_sqrt2_f))

                tmp  = (gelu + add_f) * scale_f + res_f
                OUT[n, c, h, w] = T.Cast(dtype, tmp)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused TileLang post-ops                                #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d → GELU → +add_tensor → *scale → +residual (fused)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: int,
    ):
        super().__init__()
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size  = int(kernel_size)
        self.stride       = int(scale_factor)
        self.padding      = 1
        self.output_pad   = 1

        # ---------------- ConvTranspose2d parameters -------------------- #
        w_shape = (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound  = 1 / math.sqrt(fan_in)
        self.conv_bias = nn.Parameter(torch.empty(self.out_channels))
        nn.init.uniform_(self.conv_bias, -bound, bound)

        # ---------------- Post-op learnable params ---------------------- #
        self.scale = nn.Parameter(torch.ones(1))               # scalar multiplier
        self.add_tensor = nn.Parameter(torch.randn(1, self.out_channels, 1, 1))

        # Kernel cache : {(N,C,H,W,dtype): kernel}
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

        # ---------------- ConvTranspose2d (cuDNN) ----------------------- #
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
        ).contiguous()  # (N,C,H,W)

        res_fp16 = residual.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = y.shape
        kernel = self._get_kernel(N, C, H, W, "float16")

        add_fp16   = self.add_tensor.view(-1).to(device="cuda", dtype=torch.float16).contiguous()  # (C,)
        scale_fp16 = self.scale.to(device="cuda", dtype=torch.float16).contiguous()

        out_fp16 = kernel(y, add_fp16, scale_fp16, res_fp16)
        return out_fp16.to(orig_dtype)