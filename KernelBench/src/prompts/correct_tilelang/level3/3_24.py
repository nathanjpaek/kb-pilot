"""
Problem Name: 24_EfficientNetB2
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.92 runtime_stats={'mean': 1.92, 'std': 0.139, 'min': 1.69, 'max': 2.9, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.99, 'std': 0.0867, 'min': 1.77, 'max': 2.38, 'num_trials': 100}, 'speedup_ratio': 1.04}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------
# TileLang kernel factory for the final linear layer
# ---------------------------------------------------------------------
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
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < batch_size) and (gn < out_features):
                    val = C_frag[i, j] + B[gn]
                    Y[gm, gn] = T.Cast(dtype, val)

    return kernel


# ---------------------------------------------------------------------
# EfficientNet-B2 with TileLang-accelerated final FC
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # ---------------- Stem ----------------
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # --------------- MBConv blocks ---------------
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)

        # ---------------- Head ----------------
        self.conv_final = nn.Conv2d(384, 1408, 1, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # ------------- TileLang Linear -------------
        self.in_features = 1408
        self.out_features = num_classes

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

        self._kernel_cache = {}

    # ------------------------------------------------
    # Helper to build MBConv blocks (identical to original)
    # ------------------------------------------------
    @staticmethod
    def _make_mbconv_block(in_channels, out_channels, stride, expand_ratio):
        layers = []
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                3,
                stride,
                1,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, expanded_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded_channels // 4, expanded_channels, 1, bias=False),
            nn.Sigmoid(),
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        return nn.Sequential(*layers)

    # ------------------------------------------------
    # Kernel retrieval / compilation
    # ------------------------------------------------
    def _get_linear_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_kernel(
                batch_size,
                self.in_features,
                self.out_features,
                dtype=dtype,
            )
        return self._kernel_cache[key]

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.relu(self.bn_final(self.conv_final(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 1408)

        # ----- TileLang FC -----
        batch = x.size(0)
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_linear_kernel(batch)
        out_fp16 = kernel(x_fp16, w_fp16, b_fp16)
        return out_fp16.to(dtype=x.dtype)