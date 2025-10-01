"""
Problem Name: 20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.54 runtime_stats={'mean': 1.54, 'std': 0.00459, 'min': 1.53, 'max': 1.56, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.31, 'std': 0.0243, 'min': 2.3, 'max': 2.55, 'num_trials': 100}, 'speedup_ratio': 1.5}}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


def _fused_elemwise_kernel(B, C, D, H, W, dtype="float16"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused(
        X: T.Tensor((B, C, D, H, W), dtype),
        Bias: T.Tensor((C, 1, 1, 1), dtype),
        O: T.Tensor((B, C, D, H, W), dtype),
    ):
        # Launch 2-D grid: (batch blocks, channel blocks)
        with T.Kernel(B, C, threads=128) as (bn, bc):
            bias_val = Bias[bc, 0, 0, 0]
            # Parallelize over spatial dims
            for d, h, w in T.Parallel(D, H, W):
                val = X[bn, bc, d, h, w]
                tmp1 = val + bias_val
                tmp2 = tmp1 + val
                tmp3 = tmp2 * val
                outv = tmp3 + val
                O[bn, bc, d, h, w] = outv

    return fused


class ModelNew(nn.Module):
    """
    Optimized model using TileLang.
    Performs 3-D transposed convolution via torch.nn.functional,
    then applies a fused TileLang kernel implementing:
        x = (((x + bias) + original_x) * original_x) + original_x
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # ConvTranspose3d parameters
        w_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias_conv = nn.Parameter(torch.empty(out_channels))
        fan_in = in_channels * kernel_size * kernel_size * kernel_size
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_conv, -bound, bound)

        # Extra bias added after convolution
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Cache for compiled TileLang kernels keyed by tensor shape
        self._kernel_cache = {}

    def _get_kernel(self, B, C, D, H, W, dtype="float16"):
        key = (B, C, D, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _fused_elemwise_kernel(
                B, C, D, H, W, dtype=dtype
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move inputs and parameters to CUDA
        device = "cuda"
        x = x.to(device=device)
        weight = self.weight.to(device=device)
        bias_conv = self.bias_conv.to(device=device)

        # 3-D transposed convolution
        x = F.conv_transpose3d(
            x,
            weight,
            bias_conv,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )

        # Prepare for fused element-wise kernel
        B, C, D, H, W = x.shape
        x_half = x.to(dtype=torch.float16)
        bias_half = self.bias.to(device=device, dtype=torch.float16)

        fused_kernel = self._get_kernel(B, C, D, H, W, dtype="float16")
        out = fused_kernel(x_half, bias_half)  # returns float16 tensor

        return out