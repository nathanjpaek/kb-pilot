"""
Problem Name: 85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0578 runtime_stats={'mean': 0.0578, 'std': 0.00101, 'min': 0.0562, 'max': 0.0619, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0437, 'std': 0.000842, 'min': 0.0428, 'max': 0.0486, 'num_trials': 100}, 'speedup_ratio': 0.756}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        assert groups == in_channels == out_channels
        self.in_channels = in_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.bias_flag = bias

        w = torch.empty(out_channels, 1, kernel_size_h, kernel_size_w)
        torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = nn.Parameter(w)

        if bias:
            fan_in = in_channels * kernel_size_h * kernel_size_w
            bound = 1 / math.sqrt(fan_in)
            b = torch.empty(out_channels)
            torch.nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.register_buffer("_zero_bias", torch.zeros(out_channels))

        self.block_h = 8
        self.block_w = 8
        self._cached_kernels = {}

    def _get_kernel(self, batch, in_h, in_w, dtype):
        key = (batch, in_h, in_w, dtype)
        if key in self._cached_kernels:
            return self._cached_kernels[key]

        B = batch
        C = self.in_channels
        IH = in_h
        IW = in_w
        KH = self.kernel_size_h
        KW = self.kernel_size_w
        SH = self.stride_h
        SW = self.stride_w
        PH = self.padding_h
        PW = self.padding_w
        DH = self.dilation_h
        DW = self.dilation_w
        OH = (IH + 2 * PH - DH * (KH - 1) - 1) // SH + 1
        OW = (IW + 2 * PW - DW * (KW - 1) - 1) // SW + 1
        BLOCK_H = self.block_h
        BLOCK_W = self.block_w

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def depthwise_conv(
            X: T.Tensor((B, C, IH, IW), "float16"),
            W: T.Tensor((C, 1, KH, KW), "float16"),
            BIAS: T.Tensor((C,), "float16"),
            Y: T.Tensor((B, C, OH, OW), "float16"),
        ):
            num_w = T.ceildiv(OW, BLOCK_W)
            num_h = T.ceildiv(OH, BLOCK_H)
            num_nc = B * C
            with T.Kernel(num_w, num_h, num_nc, threads=BLOCK_H * BLOCK_W) as (bx, by, bz):
                b_idx = bz // C
                c_idx = bz % C
                oh_base = by * BLOCK_H
                ow_base = bx * BLOCK_W
                acc = T.alloc_fragment((BLOCK_H, BLOCK_W), "float")
                T.clear(acc)
                for kh in range(KH):
                    ih_off = kh * DH - PH
                    for kw in range(KW):
                        iw_off = kw * DW - PW
                        for lh, lw in T.Parallel(BLOCK_H, BLOCK_W):
                            oh = oh_base + lh
                            ow = ow_base + lw
                            if (oh < OH) and (ow < OW):
                                ih = oh * SH + ih_off
                                iw = ow * SW + iw_off
                                if (ih >= 0) and (ih < IH) and (iw >= 0) and (iw < IW):
                                    acc[lh, lw] += (
                                        T.cast(X[b_idx, c_idx, ih, iw], "float")
                                        * T.cast(W[c_idx, 0, kh, kw], "float")
                                    )
                for lh, lw in T.Parallel(BLOCK_H, BLOCK_W):
                    oh = oh_base + lh
                    ow = ow_base + lw
                    if (oh < OH) and (ow < OW):
                        val = acc[lh, lw] + T.cast(BIAS[c_idx], "float")
                        Y[b_idx, c_idx, oh, ow] = T.cast(val, "float16")

        self._cached_kernels[key] = depthwise_conv
        return depthwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16)
        weight = self.weight.to(device="cuda", dtype=torch.float16)
        if self.bias_flag:
            bias = self.bias.to(device="cuda", dtype=torch.float16)
        else:
            bias = self._zero_bias.to(device="cuda", dtype=torch.float16)

        B, C, H, W = x.shape
        kernel = self._get_kernel(B, H, W, x.dtype)
        y = kernel(x, weight, bias)
        return y.to(torch.float32)