"""
Problem Name: 82_conv_depthwise_2D_square_input_square_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0609 runtime_stats={'mean': 0.0609, 'std': 0.0114, 'min': 0.0517, 'max': 0.166, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0473, 'std': 0.0101, 'min': 0.0436, 'max': 0.143, 'num_trials': 100}, 'speedup_ratio': 0.777}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """Depthwise 2-D convolution accelerated with TileLang (square kernel)."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        self.bias_flag = bias

        # ---- parameters ----
        w = torch.empty(in_channels, 1, kernel_size, kernel_size)
        torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = nn.Parameter(w)

        if bias:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            b = torch.empty(in_channels)
            torch.nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.register_buffer("_zero_bias", torch.zeros(in_channels))

        # ---- tiling params ----
        self._BLOCK_H = 8
        self._BLOCK_W = 8

        # ---- kernel cache ----
        self._kernels = {}

    # ------------------------------------------------------------------
    # Kernel factory
    def _get_kernel(self, B: int, IH: int, IW: int, dtype: torch.dtype):
        key = (B, IH, IW, dtype)
        if key in self._kernels:
            return self._kernels[key]

        C = self.in_channels
        K = self.kernel_size
        S = self.stride
        P = self.pad
        BLOCK_H = self._BLOCK_H
        BLOCK_W = self._BLOCK_W

        OH = (IH + 2 * P - K) // S + 1
        OW = (IW + 2 * P - K) // S + 1

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def depthwise_conv(
            X: T.Tensor((B, C, IH, IW), "float16"),
            W: T.Tensor((C, 1, K, K), "float16"),
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

                # local accumulator in registers
                acc = T.alloc_fragment((BLOCK_H, BLOCK_W), "float")
                T.clear(acc)

                # ----- convolution accumulation -----
                for kh in range(K):
                    ih_off = kh - P
                    for kw in range(K):
                        iw_off = kw - P
                        # parallel over tile
                        for lh, lw in T.Parallel(BLOCK_H, BLOCK_W):
                            oh = oh_base + lh
                            ow = ow_base + lw
                            if (oh < OH) and (ow < OW):
                                ih = oh * S + ih_off
                                iw = ow * S + iw_off
                                if (ih >= 0) and (ih < IH) and (iw >= 0) and (iw < IW):
                                    acc[lh, lw] += (
                                        T.cast(X[b_idx, c_idx, ih, iw], "float")
                                        * T.cast(W[c_idx, 0, kh, kw], "float")
                                    )

                # ----- write back -----
                for lh, lw in T.Parallel(BLOCK_H, BLOCK_W):
                    oh = oh_base + lh
                    ow = ow_base + lw
                    if (oh < OH) and (ow < OW):
                        val = acc[lh, lw] + T.cast(BIAS[c_idx], "float")
                        Y[b_idx, c_idx, oh, ow] = T.cast(val, "float16")

        self._kernels[key] = depthwise_conv
        return depthwise_conv

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        bias_fp16 = (
            self.bias.to(device="cuda", dtype=torch.float16) if self.bias_flag else self._zero_bias.to(device="cuda", dtype=torch.float16)
        )

        B, C, H, W = x_fp16.shape
        kernel = self._get_kernel(B, H, W, x_fp16.dtype)

        y = kernel(x_fp16, w_fp16, bias_fp16)
        return y