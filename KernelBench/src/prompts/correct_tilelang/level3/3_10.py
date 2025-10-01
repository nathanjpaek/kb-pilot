"""
Problem Name: 10_ResNet101
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=8.59 runtime_stats={'mean': 8.59, 'std': 0.0755, 'min': 8.49, 'max': 8.95, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 8.59, 'std': 0.0826, 'min': 8.44, 'max': 8.98, 'num_trials': 100}, 'speedup_ratio': 1.0}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """
    ResNet bottleneck network with the final Linear layer replaced by a
    high-performance TileLang GEMM kernel.
    """

    # ------------------------------------------------------------------ #
    #                   -------  TileLang kernel  -------                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_linear_kernel(
        batch_size: int,
        in_features: int,
        out_features: int,
        block_cols: int = 128,
        dtype: str = "float16",
        accum_dtype: str = "float32",
    ):
        """
        Kernel computes    Out = X @ W.T + B
          X  : (N, K)                    (row-major)
          W  : (K, M)  (transposed)      (row-major)
          B  : (M,)
          Out: (N, M)
        """
        grid_y = (out_features + block_cols - 1) // block_cols  # columns
        # grid_x == batch_size  (one row per block)

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def linear_kernel(
            X:   T.Tensor((batch_size, in_features), dtype),
            Wt:  T.Tensor((in_features, out_features), dtype),
            B:   T.Tensor((out_features,), dtype),
            Out: T.Tensor((batch_size, out_features), dtype),
        ):
            with T.Kernel(batch_size, grid_y, threads=block_cols) as (bx, by):
                tx   = T.get_thread_binding(0)
                row  = bx                        #     0 … N-1
                col  = by * block_cols + tx      #     0 … M-1

                if col < out_features:
                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = T.Cast(accum_dtype, B[col])

                    for k in T.serial(in_features):
                        acc[0] += (
                            T.Cast(accum_dtype, X[row, k])
                            * T.Cast(accum_dtype, Wt[k, col])
                        )

                    Out[row, col] = T.Cast(dtype, acc[0])

        return linear_kernel

    # ------------------------------------------------------------------ #
    #                            Constructor                             #
    # ------------------------------------------------------------------ #
    def __init__(self, layers, num_classes: int = 1000):
        super().__init__()
        self.in_channels = 64

        # ----- stem ----------------------------------------------------- #
        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ----- bottleneck layers --------------------------------------- #
        block = Bottleneck  # from original code context
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # ----- global pooling ------------------------------------------ #
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ----- fully-connected parameters ------------------------------ #
        in_feat = 512 * block.expansion  # 2048 for ResNet-50
        self.fc_weight = nn.Parameter(torch.empty(num_classes, in_feat))
        self.fc_bias   = nn.Parameter(torch.empty(num_classes))

        # correct initialisation (matches nn.Linear)
        torch.nn.init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_feat)
        torch.nn.init.uniform_(self.fc_bias, -bound, bound)

        # ----- kernel cache keyed by (batch_size, dtype str) ------------ #
        self._kern_cache: Dict[Tuple[int, str], callable] = {}

    # ------------------------------------------------------------------ #
    #                     ResNet helper: _make_layer                     #
    # ------------------------------------------------------------------ #
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    #                 Kernel retrieval / compilation cache               #
    # ------------------------------------------------------------------ #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, str(dtype))
        if key not in self._kern_cache:
            kern = self._build_linear_kernel(
                batch_size=batch_size,
                in_features=self.fc_weight.shape[1],
                out_features=self.fc_weight.shape[0],
                dtype="float16",
            )
            self._kern_cache[key] = kern
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    #                              forward                               #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----------- stem & bottlenecks (PyTorch/CuDNN) ---------------- #
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # --------------- global average pooling ------------------------ #
        x = self.avgpool(x)                  # (N, C, 1, 1)
        x = torch.flatten(x, 1)              # (N, C)

        # --------------- TileLang GEMM for final FC -------------------- #
        orig_dtype = x.dtype
        N, _ = x.shape

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.fc_weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.fc_bias.to(device="cuda", dtype=torch.float16).contiguous()

        # transpose weight for (K, M) layout expected by kernel
        wt_fp16 = w_fp16.t().contiguous()

        kernel = self._get_kernel(N, x_fp16.dtype)
        out_fp16 = kernel(x_fp16, wt_fp16, b_fp16)  # (N, num_classes)

        return out_fp16.to(orig_dtype)