"""
Problem Name: 22_EfficientNetB0
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.63 runtime_stats={'mean': 3.63, 'std': 0.0251, 'min': 3.58, 'max': 3.69, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 3.63, 'std': 0.0262, 'min': 3.58, 'max': 3.7, 'num_trials': 100}, 'speedup_ratio': 1.0}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_linear_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Returns a TileLang kernel that computes
        Y = X @ W.T + B
    followed by ReLU activation.
    Shapes:
        X : (batch_size, in_features)
        W : (out_features, in_features)
        B : (out_features,)
        Y : (batch_size, out_features)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        Y: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_features, block_N),
            T.ceildiv(batch_size, block_M),
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_N, block_K), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_frag)

            for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], B_s)
                T.gemm(A_s, B_s, C_frag, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                g_m = by * block_M + i
                g_n = bx * block_N + j
                if (g_m < batch_size) and (g_n < out_features):
                    val = C_frag[i, j] + B[g_n]
                    val = T.max(val, T.Cast(accum_dtype, 0))  # ReLU
                    Y[g_m, g_n] = T.Cast(dtype, val)

    return kernel


class ModelNew(nn.Module):
    """
    EfficientNetB0 with TileLang-accelerated final Linear layer.
    Convolutional backbone reuses the original PyTorch implementation.
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # -- Backbone (original PyTorch model up to penultimate layer)
        from collections import OrderedDict
        self.backbone = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)),
            ("bn1", nn.BatchNorm2d(32)),
            ("relu1", nn.ReLU(inplace=True)),
        ]))

        # Re-create MBConv blocks and remaining layers unchanged
        class MBConv(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
                super().__init__()
                self.use_residual = (stride == 1 and in_channels == out_channels)
                hidden_dim = in_channels * expand_ratio

                if expand_ratio != 1:
                    self.expand_conv = nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU6(inplace=True)
                    )
                self.depthwise_conv = nn.Sequential(
                    nn.Conv2d(
                        hidden_dim, hidden_dim, kernel_size=kernel_size,
                        stride=stride, padding=(kernel_size - 1)//2,
                        groups=hidden_dim, bias=False
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True)
                )
                self.project_conv = nn.Sequential(
                    nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

            def forward(self, x):
                identity = x
                if hasattr(self, "expand_conv"):
                    x = self.expand_conv(x)
                x = self.depthwise_conv(x)
                x = self.project_conv(x)
                if self.use_residual:
                    x = x + identity
                return x

        self.backbone.add_module("blocks", nn.Sequential(
            MBConv(32, 16, 3, 1, 1),
            MBConv(16, 24, 3, 2, 6),
            MBConv(24, 24, 3, 1, 6),
            MBConv(24, 40, 5, 2, 6),
            MBConv(40, 40, 5, 1, 6),
            MBConv(40, 80, 3, 2, 6),
            MBConv(80, 80, 3, 1, 6),
            MBConv(80, 112, 5, 1, 6),
            MBConv(112, 112, 5, 1, 6),
            MBConv(112, 192, 5, 2, 6),
            MBConv(192, 192, 5, 1, 6),
            MBConv(192, 192, 5, 1, 6),
            MBConv(192, 320, 3, 1, 6),
        ))
        self.backbone.add_module("conv2", nn.Conv2d(320, 1280, kernel_size=1, bias=False))
        self.backbone.add_module("bn2", nn.BatchNorm2d(1280))
        self.backbone.add_module("relu2", nn.ReLU(inplace=True))
        self.backbone.add_module("gap", nn.AdaptiveAvgPool2d(1))

        # -- TileLang-accelerated fully-connected layer
        self.in_features = 1280
        self.out_features = num_classes

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(self.in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self._kernel_cache = {}

    # --------------------------------------------------------------------- #
    # TileLang kernel retrieval / compilation
    # --------------------------------------------------------------------- #
    def _get_linear_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_kernel(
                batch_size=batch_size,
                in_features=self.in_features,
                out_features=self.out_features,
                dtype=dtype,
            )
        return self._kernel_cache[key]

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone (float32 on CUDA for accuracy)
        x = self.backbone(x).flatten(1)  # (B, 1280)
        batch = x.size(0)

        # TileLang FC (float16 I/O, float accum)
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_linear_kernel(batch_size=batch, dtype="float16")
        out_fp16 = kernel(x_fp16, w_fp16, b_fp16)

        return out_fp16.to(dtype=x.dtype)