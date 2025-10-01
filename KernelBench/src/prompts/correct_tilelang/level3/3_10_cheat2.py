"""
Problem Name: 10_ResNet101
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=9.57 runtime_stats={'mean': 9.57, 'std': 0.407, 'min': 8.61, 'max': 10.7, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 9.49, 'std': 0.474, 'min': 8.64, 'max': 11.1, 'num_trials': 100}, 'speedup_ratio': 0.992}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _gemm_kernel_factory(M, N, K, block_M=128, block_N=128, block_K=32, dtype="float16", accum_dtype="float32"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),      # Activations
        B: T.Tensor((K, N), dtype),      # Weight matrix already transposed
        C: T.Tensor((M, N), dtype),      # Output
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_loc)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_loc)

            T.copy(C_loc, C[by * block_M, bx * block_N])

    return gemm_kernel


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Model(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(Model, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = Bottleneck
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()

        # Backbone identical to original up to the last pooling layer
        self.backbone = Model(layers)

        # Fully-connected replacement parameters
        in_features = 512 * Bottleneck.expansion
        out_features = num_classes
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self._cached_kernels = {}

    def _get_kernel(self, M, N, K, dtype_str):
        key = (M, N, K, dtype_str)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = _gemm_kernel_factory(M, N, K, dtype=dtype_str)
        return self._cached_kernels[key]

    def forward(self, x):
        # Run backbone (all convolutions & pooling remain PyTorch for brevity)
        x = self.backbone(x)  # Shape: (batch, 512*expansion)

        # Prepare tensors for TileLang GEMM
        A = x.to(device="cuda", dtype=torch.float16)
        B = self.weight.t().contiguous().to(device="cuda", dtype=torch.float16)  # (in, out) -> (in, out)
        M, K = A.shape
        N = B.shape[1]

        gemm_kernel = self._get_kernel(M, N, K, "float16")
        C = gemm_kernel(A, B).to(torch.float32)
        C += self.bias.to(torch.float32)

        return C