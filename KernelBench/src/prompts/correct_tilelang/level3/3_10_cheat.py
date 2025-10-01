"""
Problem Name: 10_ResNet101
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=9.63 runtime_stats={'mean': 9.63, 'std': 0.196, 'min': 9.37, 'max': 11.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 9.6, 'std': 0.149, 'min': 9.38, 'max': 10.2, 'num_trials': 100}, 'speedup_ratio': 0.997}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _linear_kernel(B, OUT, IN, block_M=128, block_N=128, block_K=32,
                   dtype="float16", accum_dtype="float", num_stages=2):
    """
    Build a TileLang kernel that performs:

        Y = X @ W + bias

    with shapes:
        X : (B, IN)
        W : (IN, OUT)   (note the transposed storage w.r.t. PyTorch)
        bias : (OUT,)
        Y : (B, OUT)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((B, IN), dtype),
        W: T.Tensor((IN, OUT), dtype),
        bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((B, OUT), dtype),
    ):
        grid_x = T.ceildiv(OUT, block_N)
        grid_y = T.ceildiv(B, block_M)

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_K, block_N), dtype)
            Y_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(Y_loc)

            num_k_tiles = T.ceildiv(IN, block_K)
            for ko in T.Pipelined(num_k_tiles, num_stages=num_stages):
                # Copy X tile
                T.copy(
                    X[by * block_M, ko * block_K],
                    X_s,
                )
                # Copy W tile
                T.copy(
                    W[ko * block_K, bx * block_N],
                    W_s,
                )

                T.gemm(X_s, W_s, Y_loc)

            # Write results with bias
            for i, j in T.Parallel(block_M, block_N):
                g_i = by * block_M + i
                g_j = bx * block_N + j
                if (g_i < B) and (g_j < OUT):
                    Y[g_i, g_j] = T.Cast(dtype, Y_loc[i, j] + bias[g_j])

    return main


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Stem
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

        # Replace nn.Linear with custom TileLang kernel
        self.in_features = 512 * block.expansion
        self.out_features = num_classes

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features))

        # Parameter initialization identical to nn.Linear default
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Cache for compiled kernels: {(batch_size, dtype): kernel}
        self._cached_kernels = {}

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

    def _get_kernel(self, batch_size, dtype):
        key = (batch_size, dtype)
        if key not in self._cached_kernels:
            kernel = _linear_kernel(
                batch_size,
                self.out_features,
                self.in_features,
            )
            self._cached_kernels[key] = kernel
        return self._cached_kernels[key]

    def forward(self, x):
        # Backbone forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Shape: (B, in_features)

        # Prepare inputs for TileLang kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        weight_t = self.weight.t().contiguous().to(device="cuda", dtype=torch.float16)  # (IN, OUT)
        bias_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        B = x_fp16.shape[0]
        kernel = self._get_kernel(B, x_fp16.dtype)

        y_fp16 = kernel(x_fp16, weight_t, bias_fp16)
        return y_fp16.to(dtype=x.dtype)