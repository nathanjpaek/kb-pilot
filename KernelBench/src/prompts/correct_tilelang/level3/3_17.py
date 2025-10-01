"""
Problem Name: 17_SqueezeNetFireModule
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.571 runtime_stats={'mean': 0.571, 'std': 0.00193, 'min': 0.567, 'max': 0.579, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.712, 'std': 0.00197, 'min': 0.708, 'max': 0.724, 'num_trials': 100}, 'speedup_ratio': 1.25}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : ReLU(A,B) + channel-concat                        #
# --------------------------------------------------------------------------- #
def _build_concat_relu_kernel(
    N: int,
    H: int,
    W: int,
    C1: int,
    C2: int,
    threads: int = 256,
    dtype: str = "float16",
):
    spatial = N * H * W
    Ctot = C1 + C2

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def concat_relu(
        A: T.Tensor((N, C1, H, W), dtype),
        B: T.Tensor((N, C2, H, W), dtype),
        Y: T.Tensor((N, Ctot, H, W), dtype),
    ):
        zero_val = T.Cast(dtype, 0.0)

        with T.Kernel(T.ceildiv(spatial, threads), threads=threads) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads + tx
            if idx < spatial:
                w = idx % W
                tmp = idx // W
                h = tmp % H
                n = tmp // H

                # ------- branch-1 -----------------------------------------
                for c in T.serial(C1):
                    v = A[n, c, h, w]
                    v = T.max(v, zero_val)      # ReLU
                    Y[n, c, h, w] = v

                # ------- branch-2 -----------------------------------------
                for c in T.serial(C2):
                    v = B[n, c, h, w]
                    v = T.max(v, zero_val)      # ReLU
                    Y[n, C1 + c, h, w] = v

    return concat_relu


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused TileLang kernel                                  #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    SqueezeNet “fire” module:
        squeeze(1×1) → ReLU →
        { expand1×1 , expand3×3 } → fused (ReLU + concat)
    """

    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int,
    ):
        super().__init__()

        # identical layers ---------------------------------------------------
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(
            squeeze_channels, expand3x3_channels, kernel_size=3, padding=1
        )

        # channel counts for kernel
        self.C1 = expand1x1_channels
        self.C2 = expand3x3_channels

        # kernel cache :  (N,H,W,dtype) -> kernel
        self._kern_cache: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_concat_relu_kernel(
                N, H, W, self.C1, self.C2, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _conv2d(inp, weight, bias, stride=1, padding=0):
        return F.conv2d(inp, weight, bias, stride=stride, padding=padding)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = "cuda"

        # move to CUDA-fp16
        x_fp16 = x.to(device=device, dtype=torch.float16)

        # ---------------- squeeze + ReLU ----------------------------------
        w_sq = self.squeeze.weight.to(device=device, dtype=torch.float16)
        b_sq = self.squeeze.bias.to(device=device, dtype=torch.float16)
        s = self._conv2d(x_fp16, w_sq, b_sq)
        s = torch.relu(s)

        # ---------------- expand1×1 (no ReLU here) ------------------------
        w1 = self.expand1x1.weight.to(device=device, dtype=torch.float16)
        b1 = self.expand1x1.bias.to(device=device, dtype=torch.float16)
        e1 = self._conv2d(s, w1, b1)

        # ---------------- expand3×3 (no ReLU here) ------------------------
        w3 = self.expand3x3.weight.to(device=device, dtype=torch.float16)
        b3 = self.expand3x3.bias.to(device=device, dtype=torch.float16)
        e3 = self._conv2d(s, w3, b3, padding=1)

        # ensure contiguous for kernel
        e1c = e1.contiguous()
        e3c = e3.contiguous()

        N, _, H, W = e1c.shape
        kernel = self._get_kernel(N, H, W, "float16")

        out_fp16 = kernel(e1c, e3c)        # (N,C1+C2,H,W)
        return out_fp16.to(orig_dtype)