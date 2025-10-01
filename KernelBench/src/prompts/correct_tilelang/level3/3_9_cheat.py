"""
Problem Name: 9_ResNet18
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.82 runtime_stats={'mean': 1.82, 'std': 0.0142, 'min': 1.79, 'max': 1.86, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.84, 'std': 0.0184, 'min': 1.82, 'max': 1.92, 'num_trials': 100}, 'speedup_ratio': 1.01}}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class OptimizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(OptimizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.compiled_kernels = {}
        
        def fused_kernel(batch_size, block_M=64, block_N=64, block_K=32, dtype="float16", accum_dtype="float32"):
            @tilelang.jit(out_idx=-1)
            @T.prim_func
            def main(
                x: T.Tensor((batch_size, in_features), dtype),
                W: T.Tensor((out_features, in_features), dtype),
                bias: T.Tensor((out_features,), dtype),
                output: T.Tensor((batch_size, out_features), dtype),
            ):
                with T.Kernel(T.ceildiv(out_features, block_N), T.ceildiv(batch_size, block_M), threads=128) as (bx, by):
                    x_s = T.alloc_shared((block_M, block_K), dtype)
                    W_s = T.alloc_shared((block_K, block_N), dtype)
                    out_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
                    T.clear(out_loc)
                    for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                        T.copy(x[by*block_M, ko*block_K], x_s)
                        for k, j in T.Parallel(block_K, block_N):
                            W_s[k, j] = W[bx*block_N + j, ko*block_K + k]
                        T.gemm(x_s, W_s, out_loc)
                    for i, j in T.Parallel(block_M, block_N):
                        batch_idx = by*block_M + i
                        feat_idx = bx*block_N + j
                        if batch_idx < batch_size and feat_idx < out_features:
                            val = out_loc[i, j] + bias[feat_idx]
                            output[batch_idx, feat_idx] = val
            return main
        self.fused_kernel = fused_kernel
    
    def forward(self, x):
        x = x.to(device="cuda", dtype=torch.float16)
        batch_size = x.shape[0]
        if batch_size not in self.compiled_kernels:
            self.compiled_kernels[batch_size] = self.fused_kernel(batch_size)
        kernel = self.compiled_kernels[batch_size]
        W = self.weight.to(device="cuda", dtype=torch.float16)
        bias = self.bias.to(device="cuda", dtype=torch.float16)
        out = kernel(x, W, bias)
        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = OptimizedLinear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
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
        x = self.fc(x)

        return x