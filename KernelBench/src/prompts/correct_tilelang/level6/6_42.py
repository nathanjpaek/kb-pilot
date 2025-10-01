"""
Problem Name: 42_Conv3d_Tanh_Clamp_Swish_Divide_Sigmoid
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.9 runtime_stats={'mean': 3.9, 'std': 0.0422, 'min': 3.88, 'max': 4.31, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.56, 'std': 0.0366, 'min': 4.54, 'max': 4.87, 'num_trials': 100}, 'speedup_ratio': 1.17}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : fused tanh→clamp→swish→norm→sigmoid               #
# --------------------------------------------------------------------------- #
def _build_fused_elem_kernel(
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
    def fused_kernel(
        X:         T.Tensor((N, C, D, H, W), dtype),       # conv output
        inv_den:   T.Tensor((1,), accum_dtype),            # 1 / (mean + 1e-5)
        Out:       T.Tensor((N, C, D, H, W), dtype),       # result
    ):
        one          = T.Cast(accum_dtype, 1.0)
        clamp_min_f  = T.Cast(accum_dtype, clamp_min)
        clamp_max_f  = T.Cast(accum_dtype, clamp_max)

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
                val = T.max(val, clamp_min_f)
                val = T.min(val, clamp_max_f)
                # swish : x * sigmoid(x)
                sig = one / (one + T.exp(-val))
                val = val * sig
                # divide by (mean + 1e-5)
                val = val * inv_den[0]
                # final sigmoid
                val = one / (one + T.exp(-val))

                Out[n, c, d_, h, w] = T.Cast(dtype, val)

    return fused_kernel


# --------------------------------------------------------------------------- #
# Optimised PyTorch wrapper                                                   #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d ‑> fused(tanh, clamp, swish, /mean, sigmoid) implemented in TileLang.
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

        # Keep PyTorch Conv3d (default initialisation is correct)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

        # Clamp range
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # Kernel cache : (N,C,D,H,W,dtype) -> compiled kernel
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_elem_kernel(
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
        orig_dtype = x.dtype

        # 1. convolution (keep in high precision for accuracy)
        x = self.conv(x)

        # 2. move to CUDA / fp16 for fused kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_fp16.shape

        # 3. compute inverse divisor : 1 / (mean + 1e-5)
        denom = x_fp16.mean(dtype=torch.float32) + 1e-5
        inv_den_tensor = torch.tensor([1.0 / denom.item()],
                                      dtype=torch.float32,
                                      device="cuda")

        # 4. fused elementwise kernel
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        y_fp16 = kernel(x_fp16, inv_den_tensor)

        return y_fp16.to(orig_dtype)