"""
Problem Name: 18_SqueezeNet
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.49 runtime_stats={'mean': 1.49, 'std': 0.00942, 'min': 1.47, 'max': 1.54, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.24, 'std': 0.0134, 'min': 1.22, 'max': 1.3, 'num_trials': 100}, 'speedup_ratio': 0.832}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : concat two tensors along channel dim              #
# --------------------------------------------------------------------------- #
def _build_concat2_kernel(
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
    def concat2(
        A: T.Tensor((N, C1, H, W), dtype),
        B: T.Tensor((N, C2, H, W), dtype),
        Y: T.Tensor((N, Ctot, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(spatial, threads), threads=threads) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads + tx
            if idx < spatial:
                w = idx % W
                tmp = idx // W
                h = tmp % H
                n = tmp // H

                for c in T.serial(C1):
                    Y[n, c, h, w] = A[n, c, h, w]

                for c in T.serial(C2):
                    Y[n, C1 + c, h, w] = B[n, c, h, w]

    return concat2


# --------------------------------------------------------------------------- #
# FireModule with TileLang concat                                             #
# --------------------------------------------------------------------------- #
class FireModuleTile(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int,
    ):
        super().__init__()
        # identical layers
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(
            squeeze_channels, expand3x3_channels, kernel_size=3, padding=1
        )

        self.relu = nn.ReLU(inplace=True)

        # concat sizes
        self.C1 = expand1x1_channels
        self.C2 = expand3x3_channels

        # kernel cache : (N,H,W,dtype) -> kernel
        self._kern_cache: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_concat2_kernel(
                N, H, W, self.C1, self.C2, dtype=dtype
            )
        return self._kern_cache[key]

    # simple helper to run conv2d with explicit weight/bias tensors
    @staticmethod
    def _conv2d(x, conv: nn.Conv2d, dtype):
        w = conv.weight.to(device="cuda", dtype=dtype)
        b = conv.bias.to(device="cuda", dtype=dtype) if conv.bias is not None else None
        return F.conv2d(
            x,
            w,
            b,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(device="cuda", dtype=torch.float16)

        # squeeze
        x = self.relu(self._conv2d(x, self.squeeze, torch.float16))

        # two expand paths
        e1 = self.relu(self._conv2d(x, self.expand1x1, torch.float16))
        e3 = self.relu(self._conv2d(x, self.expand3x3, torch.float16))

        e1c = e1.contiguous()
        e3c = e3.contiguous()

        N, _, H, W = e1c.shape
        kernel = self._get_kernel(N, H, W, "float16")
        out_fp16 = kernel(e1c, e3c)

        return out_fp16.to(orig_dtype)


# --------------------------------------------------------------------------- #
# Full model with FireModuleTile                                              #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleTile(96, 16, 64, 64),
            FireModuleTile(128, 16, 64, 64),
            FireModuleTile(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleTile(256, 32, 128, 128),
            FireModuleTile(256, 48, 192, 192),
            FireModuleTile(384, 48, 192, 192),
            FireModuleTile(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleTile(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    # ------------------------------------------------------------------ #
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)