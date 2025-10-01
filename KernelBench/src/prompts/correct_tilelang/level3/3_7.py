"""
Problem Name: 7_GoogleNetInceptionV1
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.83 runtime_stats={'mean': 2.83, 'std': 0.0648, 'min': 2.71, 'max': 3.18, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.79, 'std': 0.0674, 'min': 2.66, 'max': 3.04, 'num_trials': 100}, 'speedup_ratio': 0.986}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# ------------------------------------------------------------------ #
# TileLang kernel factory for Linear
# ------------------------------------------------------------------ #
def _linear_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),          # input
        B: T.Tensor((K, N), dtype),          # weight (transposed)
        bias: T.Tensor((N,), dtype),         # bias
        C: T.Tensor((M, N), dtype),          # output
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_loc)

            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if (m < M) and (n < N):
                    C_loc[i, j] += bias[n]
                    C[m, n] = C_loc[i, j]

    return main


# ------------------------------------------------------------------ #
# PyTorch wrapper for Linear layer
# ------------------------------------------------------------------ #
class LinearTile(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ---- identical initialisation to nn.Linear -----------------------
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)
        # ------------------------------------------------------------------

        self._cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # compile / fetch kernel
    def _get_kernel(self, batch: int, dtype: torch.dtype):
        key = (batch, dtype)
        if key not in self._cache:
            self._cache[key] = _linear_kernel(
                batch, self.in_features, self.out_features
            )
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        x_half = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_half_t = self.weight.t().contiguous().to(device="cuda", dtype=torch.float16)
        b_half = self.bias.to(device="cuda", dtype=torch.float16)

        kernel = self._get_kernel(B, torch.float16)
        out_half = kernel(x_half, w_half_t, b_half)
        return out_half.to(dtype=x.dtype)


# ------------------------------------------------------------------ #
# Inception block (unchanged)
# ------------------------------------------------------------------ #
class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        reduce_3x3,
        out_3x3,
        reduce_5x5,
        out_5x5,
        pool_proj,
    ):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b3 = self.branch3x3(x)
        b5 = self.branch5x5(x)
        bp = self.branch_pool(x)
        return torch.cat([b1, b3, b5, bp], dim=1)


# ------------------------------------------------------------------ #
# Full GoogLeNet model with TileLang-accelerated FC
# ------------------------------------------------------------------ #
class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)

        # ---- TileLang FC ---------------------------------------------------
        self.fc = LinearTile(1024, num_classes)

    # ------------------------------------------------------------------ #
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x