"""
Problem Name: 25_ShuffleNetUnit
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=6.39 runtime_stats={'mean': 6.39, 'std': 0.0223, 'min': 6.37, 'max': 6.58, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 6.38, 'std': 0.0113, 'min': 6.36, 'max': 6.44, 'num_trials': 100}, 'speedup_ratio': 0.998}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : ChannelShuffle                                    #
# --------------------------------------------------------------------------- #
def _build_shuffle_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    groups: int,
    threads: int = 256,
    dtype: str = "float16",
):
    Cp = C // groups                       # channels per group
    total = N * C * H * W                  # total elements

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def channel_shuffle(
        X: T.Tensor((N, C, H, W), dtype),
        Y: T.Tensor((N, C, H, W), dtype),
    ):
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

                new_c = (c % Cp) * groups + (c // Cp)

                Y[n, new_c, h, w] = X[n, c, h, w]

    return channel_shuffle


# --------------------------------------------------------------------------- #
#                        PyTorch module with TileLang                         #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ShuffleNet unit with ChannelShuffle implemented as a TileLang kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 3):
        super().__init__()

        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        # Same layers as reference implementation -------------------------
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
            groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=1,
            groups=mid_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0,
            groups=groups, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # Channel-shuffle parameters --------------------------------------
        self.groups = int(groups)
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, shape, dtype_str: str):
        N, C, H, W = shape
        key = (N, C, H, W, dtype_str)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_shuffle_kernel(
                N, C, H, W, self.groups, dtype=dtype_str
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _conv2d(inp, weight, bias, stride=1, padding=0, groups=1):
        return F.conv2d(inp, weight, bias, stride=stride, padding=padding, groups=groups)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_orig = x.dtype
        device = x.device

        # -------------------- conv1 / bn1 / relu -------------------------
        w1 = self.conv1.weight.to(device=device, dtype=dtype_orig)
        out = self._conv2d(x, w1, None, groups=self.conv1.groups)
        out = self.bn1(out)
        out = F.relu(out)

        # -------------------- depthwise conv2 / bn2 ----------------------
        w2 = self.conv2.weight.to(device=device, dtype=dtype_orig)
        out = self._conv2d(
            out, w2, None, padding=1, groups=self.conv2.groups
        )
        out = self.bn2(out)

        # -------------------- ChannelShuffle (TileLang) ------------------
        out_fp16 = out.to(device="cuda", dtype=torch.float16).contiguous()
        kernel = self._get_kernel(out_fp16.shape, "float16")
        out_fp16 = kernel(out_fp16)
        out = out_fp16.to(device=device, dtype=dtype_orig)

        # -------------------- conv3 / bn3 / relu -------------------------
        w3 = self.conv3.weight.to(device=device, dtype=dtype_orig)
        out = self._conv2d(out, w3, None, groups=self.conv3.groups)
        out = self.bn3(out)
        out = F.relu(out)

        # -------------------- residual add -------------------------------
        out = out + self.shortcut(x)

        return out