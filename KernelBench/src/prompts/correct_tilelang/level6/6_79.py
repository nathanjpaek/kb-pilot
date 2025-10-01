"""
Problem Name: 79_Conv3d_Tanh_Sigmoid_Divide_Swish_Clamp
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.11 runtime_stats={'mean': 2.11, 'std': 0.0101, 'min': 2.1, 'max': 2.19, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.29, 'std': 0.0177, 'min': 2.28, 'max': 2.41, 'num_trials': 100}, 'speedup_ratio': 1.09}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                TileLang kernel factory  (full fused element-wise)           #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total = N * C * D * H * W
    eps   = 1e-5
    clamp_lo, clamp_hi = -0.5, 0.5

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, D, H, W), dtype),     # conv output
        Out: T.Tensor((N, C, D, H, W), dtype),     # final tensor
    ):
        one  = T.Cast(accum_dtype, 1.0)
        eps_ = T.Cast(accum_dtype, eps)
        lo   = T.Cast(accum_dtype, clamp_lo)
        hi   = T.Cast(accum_dtype, clamp_hi)

        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total:
                w   = idx % W
                t1  = idx // W
                h   = t1 % H
                t2  = t1 // H
                d_  = t2 % D
                t3  = t2 // D
                c   = t3 % C
                n   = t3 // C

                val = T.Cast(accum_dtype, X[n, c, d_, h, w])

                # tanh
                val = T.tanh(val)

                # sigmoid(val)
                s1 = one / (one + T.exp(-val))

                # sigmoid(s1)
                s2 = one / (one + T.exp(-s1))

                # divide
                divv = s1 / (s2 + eps_)

                # SiLU : x * sigmoid(x)
                s3 = one / (one + T.exp(-divv))
                val = divv * s3

                # clamp
                val = T.max(val, lo)
                val = T.min(val, hi)

                Out[n, c, d_, h, w] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
#                               PyTorch wrapper                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d  →  fused( tanh, sigmoid, ÷sigmoid+eps, SiLU, clamp ) via TileLang
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        # Conv3d with identical default initialisation
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

        # Cache: (N,C,D,H,W,dtype) -> compiled kernel
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

        # Move conv params to GPU / fp16 for fast cuDNN path
        self.conv.to(device="cuda", dtype=torch.float16)

    # ------------------------------------------------------------------ #
    def _get_kernel(self, shape, dtype: str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Conv3d (cuDNN) in fp16
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        y_fp16 = self.conv(x_fp16).contiguous()

        # Fused element-wise kernel
        N, C, D, H, W = y_fp16.shape
        kernel = self._get_kernel((N, C, D, H, W), "float16")
        out_fp16 = kernel(y_fp16)

        return out_fp16.to(orig_dtype)