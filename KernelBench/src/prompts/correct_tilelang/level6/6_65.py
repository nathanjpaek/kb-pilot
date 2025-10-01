"""
Problem Name: 65_ConvTranspose3d_Clamp_Swish_Softmax_BiasAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=5.23 runtime_stats={'mean': 5.23, 'std': 0.00881, 'min': 5.22, 'max': 5.31, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 6.19, 'std': 0.0101, 'min': 6.18, 'max': 6.26, 'num_trials': 100}, 'speedup_ratio': 1.18}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory   clamp[-2,2] → swish → softmax(dim=1) → +bias               #
# --------------------------------------------------------------------------- #

def _build_fused_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    spatial = N * D * H * W
    clamp_lo, clamp_hi = -2.0, 2.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:    T.Tensor((N, C, D, H, W), dtype),    # input from ConvT3d
        Bias: T.Tensor((C,), dtype),               # (C,) – broadcast later
        Out:  T.Tensor((N, C, D, H, W), dtype),    # output
    ):
        lo  = T.Cast(accum_dtype, clamp_lo)
        hi  = T.Cast(accum_dtype, clamp_hi)
        one = T.Cast(accum_dtype, 1.0)

        with T.Kernel(T.ceildiv(spatial, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < spatial:
                w   = idx % W
                tmp = idx // W
                h   = tmp % H
                tmp //= H
                d_  = tmp % D
                n   = tmp // D

                # -------- first pass : accumulate sum_exp -----------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0)
                for c in T.serial(C):
                    v = T.Cast(accum_dtype, X[n, c, d_, h, w])
                    v = T.max(v, lo)
                    v = T.min(v, hi)              # clamp
                    sig = one / (one + T.exp(-v))
                    v = v * sig                   # swish
                    sum_exp[0] += T.exp(v)

                inv_sum = one / sum_exp[0]

                # -------- second pass : write results ----------------------
                for c in T.serial(C):
                    v = T.Cast(accum_dtype, X[n, c, d_, h, w])
                    v = T.max(v, lo)
                    v = T.min(v, hi)
                    sig = one / (one + T.exp(-v))
                    v = v * sig
                    prob = T.exp(v) * inv_sum
                    out_val = prob + T.Cast(accum_dtype, Bias[c])
                    Out[n, c, d_, h, w] = T.Cast(dtype, out_val)

    return kernel


# --------------------------------------------------------------------------- #
#                           PyTorch wrapper                                   #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  fused TileLang kernel:
        clamp[-2,2] → swish → softmax(dim=1) → +bias
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias_shape: Tuple[int, ...]):
        super().__init__()

        # Keep PyTorch ConvTranspose3d (with identical default init)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        # move conv layer to GPU/FP16 for perf – weights stay correctly initialised
        self.conv_transpose.to(device="cuda", dtype=torch.float16)

        # Post-softmax bias (broadcasted).  Flatten to (C,) for kernel use.
        self.bias = nn.Parameter(torch.randn(bias_shape).view(-1))  # shape => (C,)

        # Kernel cache : (N,C,D,H,W,dtype) -> compiled kernel
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(N, C, D, H, W, dtype=dtype)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Ensure input is on CUDA / fp16 to match conv weights
        x_fp16 = x.to(device="cuda", dtype=torch.float16)

        # ConvTranspose3d via cuDNN
        y_fp16 = self.conv_transpose(x_fp16).contiguous()  # (N,C,D,H,W)

        N, C, D, H, W = y_fp16.shape

        # Prepare bias (C,)
        bias_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        # Fused kernel
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        out_fp16 = kernel(y_fp16, bias_fp16)

        return out_fp16.to(orig_dtype)