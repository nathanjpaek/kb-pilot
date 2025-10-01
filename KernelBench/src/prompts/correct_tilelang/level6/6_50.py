"""
Problem Name: 50_Conv3d_Tanh_Clamp_Swish_Divide
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.82 runtime_stats={'mean': 1.82, 'std': 0.00931, 'min': 1.81, 'max': 1.9, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.11, 'std': 0.00921, 'min': 2.1, 'max': 2.19, 'num_trials': 100}, 'speedup_ratio': 1.16}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : tanh → clamp                                      #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    clamp_min: float,
    clamp_max: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total = N * C * D * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, D, H, W), dtype),     # conv output
        Out: T.Tensor((N, C, D, H, W), dtype),     # final result
    ):
        cmin = T.Cast(accum_dtype, float(clamp_min))
        cmax = T.Cast(accum_dtype, float(clamp_max))

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

                # clamp
                val = T.max(val, cmin)
                val = T.min(val, cmax)

                Out[n, c, d_, h, w] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper                                                             #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d  →  fused( tanh → clamp ) using TileLang
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        clamp_min: float,
        clamp_max: float,
    ):
        super().__init__()

        # Conv3d layer (default PyTorch init is already correct)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

        # store clamp constants
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # kernel cache  {(N,C,D,H,W,dtype): kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

        # move conv params to GPU / fp16 for fast cuDNN
        self.conv.to(device="cuda", dtype=torch.float16)

    # ------------------------------------------------------------------ #
    def _get_kernel(self, shape, dtype: str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
                N,
                C,
                D,
                H,
                W,
                self.clamp_min,
                self.clamp_max,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        # ensure input on CUDA / fp16 to match conv weights
        x_fp16 = x.to(device="cuda", dtype=torch.float16)

        # Conv3d via cuDNN
        y_fp16 = self.conv(x_fp16).contiguous()

        # fused element-wise kernel
        N, C, D, H, W = y_fp16.shape
        kernel = self._get_kernel((N, C, D, H, W), "float16")
        out_fp16 = kernel(y_fp16)

        return out_fp16.to(original_dtype)