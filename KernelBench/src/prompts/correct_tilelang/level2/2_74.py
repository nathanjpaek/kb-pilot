"""
Problem Name: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.658 runtime_stats={'mean': 0.658, 'std': 0.00469, 'min': 0.65, 'max': 0.68, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.12, 'std': 0.00998, 'min': 1.11, 'max': 1.21, 'num_trials': 100}, 'speedup_ratio': 1.7}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory: LeakyReLU → ×multiplier → LeakyReLU → MaxPool3d(2)
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    D_in: int,
    H_in: int,
    W_in: int,
    neg_slope: float = 0.2,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    # output shapes after k=2,s=2 max-pool
    D_out = (D_in - 2) // 2 + 1
    H_out = (H_in - 2) // 2 + 1
    W_out = (W_in - 2) // 2 + 1

    TOT  = N * C * D_out * H_out * W_out          # one thread -> one output
    GRID = (TOT + threads_per_block - 1) // threads_per_block

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:  T.Tensor((N, C, D_in, H_in, W_in), dtype),   # conv output
        M:  T.Tensor((C,), dtype),                       # per-channel multiplier
        Y:  T.Tensor((N, C, D_out, H_out, W_out), dtype),
    ):
        ns  = T.Cast(accum_dtype, neg_slope)
        zero = T.Cast(accum_dtype, 0)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                w_o  = idx % W_out
                tmp1 = idx // W_out
                h_o  = tmp1 % H_out
                tmp2 = tmp1 // H_out
                d_o  = tmp2 % D_out
                tmp3 = tmp2 // D_out
                c    = tmp3 % C
                n    = tmp3 // C

                d_base = d_o * 2
                h_base = h_o * 2
                w_base = w_o * 2

                m_c = T.Cast(accum_dtype, M[c])

                max_val = T.alloc_local((1,), accum_dtype)
                max_val[0] = -3.4028234663852886e38   # −inf in fp32

                # iterate 2×2×2 pool window
                for kd in T.serial(2):
                    for kh in T.serial(2):
                        for kw in T.serial(2):
                            v0 = T.Cast(
                                accum_dtype,
                                X[n, c, d_base + kd, h_base + kh, w_base + kw],
                            )
                            # first LeakyReLU
                            v1 = T.max(v0, zero) + ns * T.min(v0, zero)
                            # multiply
                            v2 = v1 * m_c
                            # second LeakyReLU
                            v3 = T.max(v2, zero) + ns * T.min(v2, zero)
                            # max reduction
                            max_val[0] = T.max(max_val[0], v3)

                Y[n, c, d_o, h_o, w_o] = T.Cast(dtype, max_val[0])

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch) → fused TileLang kernel implementing
        LeakyReLU(0.2) → ×multiplier → LeakyReLU(0.2) → MaxPool3d(k=2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        multiplier_shape: tuple,
    ):
        super().__init__()

        # ---- keep ConvTranspose3d exactly as in reference -----------------
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        # learnable multiplier (identical init)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        # cache : {(N,D,H,W,dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            C = self.conv_transpose.out_channels
            self._kern_cache[key] = _build_fused_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # 1) ConvTranspose3d on CUDA in fp16
        x = self.conv_transpose(x.to(device="cuda", dtype=torch.float16))

        N, C, D_in, H_in, W_in = x.shape

        # 2) fused TileLang kernel
        kernel = self._get_kernel(N, D_in, H_in, W_in, "float16")
        mult_fp16 = self.multiplier.view(-1).to(device="cuda", dtype=torch.float16)
        y_fp16 = kernel(x.contiguous(), mult_fp16)

        return y_fp16.to(orig_dtype)