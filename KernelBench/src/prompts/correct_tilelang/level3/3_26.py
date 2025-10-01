"""
Problem Name: 26_ShuffleNet
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=7.07 runtime_stats={'mean': 7.07, 'std': 0.0237, 'min': 7.01, 'max': 7.12, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 7.04, 'std': 0.0272, 'min': 6.97, 'max': 7.13, 'num_trials': 100}, 'speedup_ratio': 0.996}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# -------------------------------------------------------------------- #
# TileLang GEMM kernel factory                                         #
# -------------------------------------------------------------------- #
def _build_linear_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Return a TileLang kernel that computes
        Y[M,N] = X[M,K] @ W[K,N] + B[N]
    Shapes:
        X : (M, K)
        W : (K, N)      -- note: weight already transposed
        B : (N,)
        Y : (M, N)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((K, N), dtype),
        B: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),  # grid-x
            T.ceildiv(M, block_M),  # grid-y
            threads=128,
        ) as (bx, by):
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_K, block_N), dtype)
            C   = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C)

            # K-loop with software pipeline
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], X_s)          # load X tile
                T.copy(W[ko * block_K, bx * block_N], W_s)          # load W tile
                T.gemm(X_s, W_s, C)                                 # accumulate

            # add bias and store
            for i, j in T.Parallel(block_M, block_N):
                global_m = by * block_M + i
                global_n = bx * block_N + j
                if (global_m < M) and (global_n < N):               # bounds
                    val = C[i, j] + B[global_n]
                    Y[global_m, global_n] = T.Cast(dtype, val)

    return main


# -------------------------------------------------------------------- #
# PyTorch wrapper for TileLang GEMM                                    #
# -------------------------------------------------------------------- #
class LinearTile(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # --- identical parameter initialisation to nn.Linear -------------
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)
        # -----------------------------------------------------------------

        self._kernel_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # fetch / compile kernel for current batch
    def _get_kernel(self, batch: int, dtype: torch.dtype):
        key = (batch, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_kernel(
                M=batch,
                K=self.in_features,
                N=self.out_features,
                dtype="float16",
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, in_features)
        B = x.size(0)

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_t_fp16 = self.weight.t().contiguous().to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        kernel = self._get_kernel(B, torch.float16)
        y_fp16 = kernel(x_fp16, w_t_fp16, b_fp16)

        return y_fp16.to(dtype=x.dtype)


# -------------------------------------------------------------------- #
# Helper components (unchanged)                                        #
# -------------------------------------------------------------------- #
class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        g = self.groups
        x = x.view(b, g, c // g, h, w).transpose(1, 2).contiguous()
        return x.view(b, c, h, w)


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 3):
        super().__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, 1, bias=False, groups=groups
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(
            mid_channels, out_channels, 1, bias=False, groups=groups
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shuffle = ChannelShuffle(groups)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        return out + self.shortcut(x)


# -------------------------------------------------------------------- #
# ShuffleNet with TileLang-accelerated FC                              #
# -------------------------------------------------------------------- #
class ModelNew(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        groups: int = 3,
        stages_repeats=(3, 7, 3),
        stages_out_channels=(24, 240, 480, 960),
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(3, stages_out_channels[0], 3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1],
                                       stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2],
                                       stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3],
                                       stages_repeats[2], groups)

        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, 1, bias=False)
        self.bn5   = nn.BatchNorm2d(1024)

        # ---- TileLang FC -------------------------------------------------
        self.fc = LinearTile(1024, num_classes)

    # create ShuffleNet stage
    @staticmethod
    def _make_stage(in_ch, out_ch, repeats, groups):
        layers = [ShuffleNetUnit(in_ch, out_ch, groups)]
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_ch, out_ch, groups))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, 1024)

        x = self.fc(x)
        return x