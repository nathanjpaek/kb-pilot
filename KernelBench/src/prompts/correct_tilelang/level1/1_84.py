"""
Problem Name: 84_conv_depthwise_2D_asymmetric_input_square_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0709 runtime_stats={'mean': 0.0709, 'std': 0.0153, 'min': 0.0635, 'max': 0.206, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0325, 'std': 0.00915, 'min': 0.0291, 'max': 0.118, 'num_trials': 100}, 'speedup_ratio': 0.458}}
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
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        # --- attributes ---
        assert in_channels == out_channels, "Depthwise conv must have C_out == C_in"
        self.C = in_channels
        self.K = kernel_size
        self.stride = stride
        self.pad = padding
        self.bias_flag = bias

        # --- parameters (identical init to PyTorch) ---
        w = torch.empty(out_channels, 1, self.K, self.K)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = nn.Parameter(w)

        if bias:
            fan_in = self.K * self.K  # groups == in_channels, fan_in == K^2
            bound = 1 / math.sqrt(fan_in)
            b = torch.empty(out_channels)
            nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.register_buffer("_zero_bias", torch.zeros(out_channels, dtype=torch.float16))

        # --- misc ---
        self.BLOCK_H = 8
        self.BLOCK_W = 8
        self._kernel_cache = {}

    # ------------------------------------------------------------------
    # Kernel factory ----------------------------------------------------
    def _get_kernel(self, B: int, H: int, W: int, dtype: torch.dtype):
        key = (B, H, W, dtype)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        C = self.C
        K = self.K
        S = self.stride
        P = self.pad
        OH = (H + 2 * P - (K - 1) - 1) // S + 1
        OW = (W + 2 * P - (K - 1) - 1) // S + 1
        BH = self.BLOCK_H
        BW = self.BLOCK_W

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def depthwise_conv2d(
            X: T.Tensor((B, H, W, C), "float16"),
            Wt: T.Tensor((C, 1, K, K), "float16"),
            BIAS: T.Tensor((C,), "float16"),
            Y: T.Tensor((B, OH, OW, C), "float16"),
        ):
            grid_x = T.ceildiv(OW, BW)        # tiles along width
            grid_y = T.ceildiv(OH, BH)        # tiles along height
            grid_z = B * C                    # one (batch,channel) per block-z
            with T.Kernel(grid_x, grid_y, grid_z,
                          threads=BH * BW) as (bx, by, bz):
                b_idx = bz // C
                c_idx = bz % C
                oh_base = by * BH
                ow_base = bx * BW

                acc = T.alloc_fragment((BH, BW), "float")
                T.clear(acc)

                # --- convolution ---
                for kh in range(K):
                    ih_offset = kh - P
                    for kw in range(K):
                        iw_offset = kw - P
                        for th, tw in T.Parallel(BH, BW):
                            oh = oh_base + th
                            ow = ow_base + tw
                            if (oh < OH) and (ow < OW):
                                ih = oh * S + ih_offset
                                iw = ow * S + iw_offset
                                if (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W):
                                    inp = T.cast(X[b_idx, ih, iw, c_idx], "float")
                                    wt  = T.cast(Wt[c_idx, 0, kh, kw], "float")
                                    acc[th, tw] += inp * wt

                # --- add bias & store ---
                for th, tw in T.Parallel(BH, BW):
                    oh = oh_base + th
                    ow = ow_base + tw
                    if (oh < OH) and (ow < OW):
                        val = acc[th, tw] + T.cast(BIAS[c_idx], "float")
                        Y[b_idx, oh, ow, c_idx] = T.cast(val, "float16")

        self._kernel_cache[key] = depthwise_conv2d
        return depthwise_conv2d

    # ------------------------------------------------------------------
    # Forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        bias_fp16 = (
            self.bias.to(device="cuda", dtype=torch.float16)
            if self.bias_flag
            else self._zero_bias.to(device="cuda")
        )

        B, C, H, W = x_fp16.shape
        kernel = self._get_kernel(B, H, W, x_fp16.dtype)

        # Convert to NHWC for memory coalescing
        x_nhwc = x_fp16.permute(0, 2, 3, 1).contiguous()

        y = kernel(x_nhwc, w_fp16, bias_fp16)

        # Back to NCHW and fp32
        y_nchw = y.permute(0, 3, 1, 2).contiguous().to(torch.float32)
        return y_nchw