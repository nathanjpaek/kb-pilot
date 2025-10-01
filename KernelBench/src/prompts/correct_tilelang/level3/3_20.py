"""
Problem Name: 20_MobileNetV2
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.31 runtime_stats={'mean': 4.31, 'std': 0.126, 'min': 4.03, 'max': 4.6, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.45, 'std': 0.293, 'min': 4.07, 'max': 5.58, 'num_trials': 100}, 'speedup_ratio': 1.03}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _make_linear_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        Out: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_features, block_N),
            T.ceildiv(batch_size, block_M),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                T.copy(X[by * block_M, ko * block_K], A_shared)
                T.copy(W[bx * block_N, ko * block_K], W_shared)
                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                global_m = by * block_M + i
                global_n = bx * block_N + j
                if (global_m < batch_size) and (global_n < out_features):
                    val = C_local[i, j] + B[global_n]
                    Out[global_m, global_n] = T.Cast(dtype, val)

    return linear_kernel


class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))

            layers.extend(
                [
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        stride,
                        1,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                ]
            )

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                layers, _ = _inverted_residual_block(
                    input_channel, output_channel, stride, expand_ratio=t
                )
                features.append(layers)
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        self.in_features = last_channel
        self.out_features = num_classes

        self.weight = nn.Parameter(torch.empty(num_classes, last_channel))
        self.bias = nn.Parameter(torch.zeros(num_classes))

        with torch.no_grad():
            nn.init.normal_(self.weight, 0, 0.01)
            nn.init.zeros_(self.bias)

        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _make_linear_kernel(
                batch_size,
                self.in_features,
                self.out_features,
                dtype=dtype,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        kernel = self._get_kernel(x_fp16.shape[0])
        out_fp16 = kernel(x_fp16, w_fp16, b_fp16)

        return out_fp16.to(dtype=x.dtype)