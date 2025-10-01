"""
Problem Name: 8_ResNetBasicBlock
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.19 runtime_stats={'mean': 2.19, 'std': 0.00837, 'min': 2.18, 'max': 2.26, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.2, 'std': 0.00728, 'min': 2.18, 'max': 2.24, 'num_trials': 100}, 'speedup_ratio': 1.0}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : (A + B) → ReLU                                    #
# --------------------------------------------------------------------------- #
def _build_add_relu_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    threads: int = 256,
    dtype: str = "float16",
):
    total = N * C * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def add_relu(
        A: T.Tensor((N, C, H, W), dtype),
        B: T.Tensor((N, C, H, W), dtype),
        Y: T.Tensor((N, C, H, W), dtype),
    ):
        zero = T.Cast(dtype, 0.0)

        with T.Kernel(T.ceildiv(total, threads), threads=threads) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads + tx

            if idx < total:
                w  = idx % W
                t1 = idx // W
                h  = t1 % H
                t2 = t1 // H
                c  = t2 % C
                n  = t2 // C

                val = A[n, c, h, w] + B[n, c, h, w]
                Y[n, c, h, w] = T.max(val, zero)

    return add_relu


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused residual-add + ReLU                              #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # --- original layers (params identical to stock module) ------------
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

        # kernel cache  : (N,C,H,W,dtype) -> kernel
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_add_relu_kernel(N, C, H, W, dtype=dtype)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = x.device

        # ------------ main path ------------------------------------------
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # ------------ identity path --------------------------------------
        identity = self.downsample(x)

        # ------------ fused add + ReLU  ----------------------------------
        out_fp16 = out.to(device="cuda", dtype=torch.float16).contiguous()
        id_fp16  = identity.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = out_fp16.shape
        kernel = self._get_kernel(N, C, H, W, "float16")
        y_fp16 = kernel(out_fp16, id_fp16)

        return y_fp16.to(device=device, dtype=orig_dtype)