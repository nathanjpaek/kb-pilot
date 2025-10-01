"""
Problem Name: 18_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=21.3 runtime_stats={'mean': 21.3, 'std': 0.0239, 'min': 21.3, 'max': 21.4, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 22.4, 'std': 0.0294, 'min': 22.4, 'max': 22.6, 'num_trials': 100}, 'speedup_ratio': 1.05}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------- #
# TileLang kernel factory                                               #
# --------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    Din: int,
    Hin: int,
    Win: int,
    comb_scale: float,
    bias_scale: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Dout = (Din - 2) // 2 + 1
    Hout = (Hin - 2) // 2 + 1
    Wout = (Win - 2) // 2 + 1
    TOT = N * C * Dout * Hout * Wout

    eighth = comb_scale / 8.0  # (scale1*scale2)/8  – baked constant

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused(
        X: T.Tensor((N, C, Din, Hin, Win), dtype),
        B: T.Tensor((C,), dtype),
        Y: T.Tensor((N, C, Dout, Hout, Wout), dtype),
    ):
        cs  = T.Cast(accum_dtype, eighth)
        bs  = T.Cast(accum_dtype, bias_scale)
        with T.Kernel(T.ceildiv(TOT, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                w_out  = idx % Wout
                tmp1   = idx // Wout
                h_out  = tmp1 % Hout
                tmp2   = tmp1 // Hout
                d_out  = tmp2 % Dout
                tmp3   = tmp2 // Dout
                c      = tmp3 % C
                n      = tmp3 // C

                d0 = d_out * 2
                h0 = h_out * 2
                w0 = w_out * 2

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for kd in T.serial(2):
                    for kh in T.serial(2):
                        for kw in T.serial(2):
                            acc[0] += T.Cast(
                                accum_dtype,
                                X[n, c, d0 + kd, h0 + kh, w0 + kw],
                            )

                val = acc[0] * cs + T.Cast(accum_dtype, B[c]) * bs
                Y[n, c, d_out, h_out, w_out] = T.Cast(dtype, val)

    return fused


# --------------------------------------------------------------------- #
# Optimised PyTorch module                                              #
# --------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d (cuDNN) →
    fused TileLang kernel implementing
        ×scale1 → AvgPool3d(k=2) → +bias → ×scale2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        scale1: float,
        scale2: float,
        bias_shape: tuple,
    ):
        super().__init__()
        # ConvTranspose3d with default initialisation
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        # learnable scalars & bias
        self.scale1 = nn.Parameter(torch.tensor(float(scale1)))
        self.scale2 = nn.Parameter(torch.tensor(float(scale2)))
        self.bias   = nn.Parameter(torch.randn(bias_shape))

        # kernel cache {(N,D,H,W,comb_scale,bias_scale,dtype): kernel}
        self._kernels: Dict[Tuple[int, int, int, int, float, float, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self,
        N: int,
        D: int,
        H: int,
        W: int,
        comb_scale: float,
        bias_scale: float,
        dtype: str = "float16",
    ):
        key = (N, D, H, W, comb_scale, bias_scale, dtype)
        if key not in self._kernels:
            C = self.conv_transpose.out_channels
            self._kernels[key] = _build_fused_kernel(
                N, C, D, H, W, comb_scale, bias_scale, dtype=dtype
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # 1) ConvTranspose3d (cuDNN)
        x = self.conv_transpose(x)

        # 2) Prepare inputs for TileLang kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_fp16.shape

        comb_scale = float(self.scale1.item() * self.scale2.item())
        bias_scale = float(self.scale2.item())

        kernel = self._get_kernel(N, D, H, W, comb_scale, bias_scale, "float16")
        bias_flat = self.bias.to(device="cuda", dtype=torch.float16).contiguous().view(-1)

        # 3) Launch kernel
        y_fp16 = kernel(x_fp16, bias_flat)

        return y_fp16.to(orig_dtype)