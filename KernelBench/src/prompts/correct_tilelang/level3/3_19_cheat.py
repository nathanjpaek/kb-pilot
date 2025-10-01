"""
Problem Name: 19_MobileNetV1
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.22 runtime_stats={'mean': 2.22, 'std': 0.117, 'min': 1.97, 'max': 2.54, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.22, 'std': 0.101, 'min': 1.95, 'max': 2.35, 'num_trials': 100}, 'speedup_ratio': 1.0}}
"""

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import tilelang
import tilelang.language as T


def _make_matmul_kernel(M: int, K: int, N: int, dtype: str = "float16"):
    block_M, block_N, block_K = 64, 64, 32

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float")

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha: float = 1.0):
        super().__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )

        self.in_features = int(1024 * alpha)
        self.out_features = num_classes

        self.weight = Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = Parameter(torch.empty(self.out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

        self._kernel_cache = {}

    def _fetch_kernel(self, M: int, dtype: str = "float16"):
        key = (M, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _make_matmul_kernel(
                M, self.in_features, self.out_features, dtype=dtype
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (B, 1024*alpha)

        B = x.size(0)
        weight_t = self.weight.t().contiguous().to(device=x.device, dtype=torch.float16)
        kernel = self._fetch_kernel(B, "float16")

        out = kernel(x, weight_t)
        out += self.bias.to(device=x.device, dtype=torch.float16)

        return out