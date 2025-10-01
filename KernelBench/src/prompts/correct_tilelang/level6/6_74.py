"""
Problem Name: 74_ConvTranspose3d_BiasAdd_Swish_Clamp_Softmax
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=6.81 runtime_stats={'mean': 6.81, 'std': 0.0287, 'min': 6.79, 'max': 7.08, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 7.49, 'std': 0.0168, 'min': 7.47, 'max': 7.53, 'num_trials': 100}, 'speedup_ratio': 1.1}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """
    ConvTranspose3d   →   (+Bias) → Swish(SiLU) → clamp[-1,1] → softmax(dim=1)
    The entire post-conv chain is fused into a single TileLang kernel.
    """

    # ------------------------------------------------------------------ #
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias_shape: Tuple[int, ...]):
        super().__init__()

        # 1) ConvTranspose3d – identical to PyTorch defaults
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)

        # 2) Extra bias added before non-linearities
        self.bias = nn.Parameter(torch.randn(bias_shape).view(-1))  # (C,)

        # 3) Kernel cache      {(N,C,D,H,W,dtype): compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_kernel(N: int, C: int, D: int, H: int, W: int, dtype: str = "float16"):
        threads = 256
        spatial = N * D * H * W
        clamp_lo = -1.0
        clamp_hi = 1.0
        accum_dtype = "float32"

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            X:    T.Tensor((N, C, D, H, W), dtype),   # conv output
            Bias: T.Tensor((C,), dtype),              # (C,)
            Out:  T.Tensor((N, C, D, H, W), dtype),   # result
        ):
            lo  = T.Cast(accum_dtype, clamp_lo)
            hi  = T.Cast(accum_dtype, clamp_hi)
            one = T.Cast(accum_dtype, 1.0)

            with T.Kernel(T.ceildiv(spatial, threads), threads=threads) as bx:
                tx  = T.get_thread_binding(0)
                idx = bx * threads + tx
                if idx < spatial:
                    w   = idx % W
                    t1  = idx // W
                    h   = t1 % H
                    t2  = t1 // H
                    d_  = t2 % D
                    n   = t2 // D

                    # ---------------- pass-1 : accumulate denominator ------
                    sum_exp = T.alloc_local((1,), accum_dtype)
                    sum_exp[0] = T.Cast(accum_dtype, 0)
                    for c in T.serial(C):
                        v = T.Cast(accum_dtype, X[n, c, d_, h, w]) + T.Cast(accum_dtype, Bias[c])
                        sig = one / (one + T.exp(-v))        # sigmoid
                        v = v * sig                          # silu / swish
                        v = T.max(v, lo)
                        v = T.min(v, hi)                     # clamp
                        sum_exp[0] += T.exp(v)

                    inv_sum = one / sum_exp[0]

                    # ---------------- pass-2 : final write ------------------
                    for c in T.serial(C):
                        v = T.Cast(accum_dtype, X[n, c, d_, h, w]) + T.Cast(accum_dtype, Bias[c])
                        sig = one / (one + T.exp(-v))
                        v = v * sig
                        v = T.max(v, lo)
                        v = T.min(v, hi)
                        prob = T.exp(v) * inv_sum
                        Out[n, c, d_, h, w] = T.Cast(dtype, prob)

        return kernel

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = self._build_kernel(N, C, D, H, W, dtype)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ConvTranspose3d on GPU/FP16 for speed
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        y_fp16 = self.conv_transpose(x_fp16).contiguous()        # (N,C,D,H,W)

        N, C, D, H, W = y_fp16.shape

        # Prepare bias vector (C,)
        bias_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        # Get / compile fused kernel
        ker = self._get_kernel(N, C, D, H, W, "float16")
        out_fp16 = ker(y_fp16, bias_fp16)

        return out_fp16.to(orig_dtype)