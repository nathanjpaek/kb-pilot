"""
Problem Name: 83_conv_depthwise_2D_square_input_asymmetric_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0558 runtime_stats={'mean': 0.0558, 'std': 0.0113, 'min': 0.0507, 'max': 0.156, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0517, 'std': 0.00711, 'min': 0.0486, 'max': 0.117, 'num_trials': 100}, 'speedup_ratio': 0.927}}
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
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.C = in_channels
        self.KH = kernel_size
        self.KW = 1
        self.stride = stride
        self.pad = padding
        self.dil = dilation
        self.has_bias = bias

        # ---- parameters ----
        w = torch.empty(in_channels, 1, self.KH, self.KW)
        torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = nn.Parameter(w)

        if bias:
            bound = 1 / math.sqrt(self.KH * self.KW)
            b = torch.empty(in_channels)
            torch.nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.register_buffer("_zero_bias", torch.zeros(in_channels))

        # kernel-cache
        self._kernels = {}
        # tiling
        self.BLOCK_H = 8
        self.BLOCK_W = 16

    # --------------------------------------------------------------------- #
    def _get_kernel(self, B: int, H: int, W: int, dtype: torch.dtype):
        key = (B, H, W, dtype)
        if key in self._kernels:
            return self._kernels[key]

        C = self.C
        KH = self.KH
        KW = self.KW
        STR = self.stride
        PAD = self.pad
        DIL = self.dil

        OH = (H + 2 * PAD - DIL * (KH - 1) - 1) // STR + 1
        OW = (W + 2 * PAD - 1) // STR + 1  # KW == 1

        BLOCK_H = self.BLOCK_H
        BLOCK_W = self.BLOCK_W

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def depthwise_kern(
            X: T.Tensor((B, C, H, W), "float16"),
            Wt: T.Tensor((C, 1, KH, KW), "float16"),
            BIAS: T.Tensor((C,), "float16"),
            Y: T.Tensor((B, C, OH, OW), "float16"),
        ):
            num_w = T.ceildiv(OW, BLOCK_W)
            num_h = T.ceildiv(OH, BLOCK_H)
            num_nc = B * C
            with T.Kernel(
                num_w,
                num_h,
                num_nc,
                threads=BLOCK_H * BLOCK_W,
            ) as (bx, by, bz):
                b_idx = bz // C
                c_idx = bz % C
                oh_base = by * BLOCK_H
                ow_base = bx * BLOCK_W

                acc = T.alloc_fragment((BLOCK_H, BLOCK_W), "float")
                T.clear(acc)

                for kh in range(KH):
                    ih_off = kh * DIL - PAD
                    # only one kw (KW == 1)
                    iw_off = -PAD
                    for lh, lw in T.Parallel(BLOCK_H, BLOCK_W):
                        oh = oh_base + lh
                        ow = ow_base + lw
                        if (oh < OH) and (ow < OW):
                            ih = oh * STR + ih_off
                            iw = ow * STR + iw_off
                            if (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W):
                                acc[lh, lw] += (
                                    T.cast(X[b_idx, c_idx, ih, iw], "float")
                                    * T.cast(Wt[c_idx, 0, kh, 0], "float")
                                )

                for lh, lw in T.Parallel(BLOCK_H, BLOCK_W):
                    oh = oh_base + lh
                    ow = ow_base + lw
                    if (oh < OH) and (ow < OW):
                        val = acc[lh, lw] + T.cast(BIAS[c_idx], "float")
                        Y[b_idx, c_idx, oh, ow] = T.cast(val, "float16")

        self._kernels[key] = depthwise_kern
        return depthwise_kern

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        bias_fp16 = (
            self.bias.to(device="cuda", dtype=torch.float16)
            if self.has_bias
            else self._zero_bias.to(device="cuda", dtype=torch.float16)
        )

        B, C, H, W = x_fp16.shape
        kernel = self._get_kernel(B, H, W, x_fp16.dtype)

        y_fp16 = kernel(x_fp16, w_fp16, bias_fp16)
        return y_fp16.to(torch.float32)