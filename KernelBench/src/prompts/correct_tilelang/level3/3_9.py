"""
Problem Name: 9_ResNet18
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.07 runtime_stats={'mean': 2.07, 'std': 0.13, 'min': 1.8, 'max': 3.04, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.89, 'std': 0.532, 'min': 1.69, 'max': 6.15, 'num_trials': 100}, 'speedup_ratio': 0.913}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# -----------------------  TileLang GEMM (+bias) kernel  -------------------- #
# --------------------------------------------------------------------------- #
def _build_linear_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    stages: int = 2,
    threads: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),        # (batch, in_features)
        W: T.Tensor((N, K), dtype),        # row-major like nn.Linear.weight
        B: T.Tensor((N,), dtype),          # bias
        Y: T.Tensor((M, N), dtype),        # output
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)
            Acc  = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(Acc)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=stages):
                T.copy(
                    X[by * block_M : (by + 1) * block_M,
                      ko * block_K : (ko + 1) * block_K],
                    X_s,
                )
                T.copy(
                    W[bx * block_N : (bx + 1) * block_N,
                      ko * block_K : (ko + 1) * block_K],
                    W_s,
                )
                T.gemm(X_s, W_s, Acc, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    val = Acc[i, j] + B[gj].astype(accum_dtype)
                    Y[gi, gj] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
# ----------------------------  BasicBlock (unchanged) ---------------------- #
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# --------------------------------------------------------------------------- #
# ------------------------------  ResNet-like Model ------------------------- #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Same architecture as original, but the final Linear layer
    (512 → num_classes) is replaced by a TileLang GEMM(+bias) kernel.
    """

    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # -----------------------  Linear parameters  ---------------------- #
        in_features = 512 * BasicBlock.expansion
        self.out_features = num_classes

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.bias   = nn.Parameter(torch.empty(num_classes))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache : {(batch,dtype): kernel}
        self._kernel_cache: Dict[Tuple[int, str], callable] = {}

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

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    def _get_kernel(self, batch: int, dtype: str):
        key = (batch, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_kernel(
                M=batch,
                K=self.weight.shape[1],
                N=self.out_features,
                dtype=dtype,
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone (PyTorch / cuDNN)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)  # shape (batch, 512)

        # Prepare tensors
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        # TileLang GEMM(+bias)
        kernel = self._get_kernel(x_f16.shape[0], "float16")
        out_f16 = kernel(x_f16, w_f16, b_f16)

        return out_f16.to(x.dtype)

# --------------------------------------------------------------------------- #
# ------------------------------  helper IO  -------------------------------- #
# --------------------------------------------------------------------------- #
batch_size = 2
input_shape = (batch_size, 3, 224, 224)

def get_inputs():
    return [torch.randn(input_shape)]

def get_init_inputs():
    return []  # ModelNew takes default num_classes=1000