"""
Problem Name: 62_conv_standard_2D__square_input__asymmetric_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.59 runtime_stats={'mean': 1.59, 'std': 0.0205, 'min': 1.57, 'max': 1.75, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.387, 'std': 0.0227, 'min': 0.378, 'max': 0.61, 'num_trials': 100}, 'speedup_ratio': 0.243}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def conv2d_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    F: int,
    KH: int,
    KW: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dil_h: int = 1,
    dil_w: int = 1,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1
    K_TOTAL = KH * KW * C
    M_TOTAL = N * OH * OW

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        x: T.Tensor((N, H, W, C), dtype),
        w: T.Tensor((KH, KW, C, F), dtype),
        y: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),
            T.ceildiv(M_TOTAL, block_M),
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_s = T.alloc_shared((block_M, block_N), dtype)

            w_flat = T.Tensor((K_TOTAL, F), dtype, w.data)
            y_flat = T.Tensor((M_TOTAL, F), dtype, y.data)

            T.clear(C_frag)

            for ko in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # Load input (im2col) tile to shared
                for i, j in T.Parallel(block_M, block_K):
                    m_idx = by * block_M + i
                    k_idx = ko * block_K + j

                    if (m_idx < M_TOTAL) and (k_idx < K_TOTAL):
                        n = m_idx // (OH * OW)
                        oh = (m_idx % (OH * OW)) // OW
                        ow = m_idx % OW

                        kh = k_idx // (KW * C)
                        kw = (k_idx // C) % KW
                        c_idx = k_idx % C

                        ih = oh * stride_h + kh * dil_h - pad_h
                        iw = ow * stride_w + kw * dil_w - pad_w

                        valid = (
                            (ih >= 0)
                            and (ih < H)
                            and (iw >= 0)
                            and (iw < W)
                        )
                        A_s[i, j] = T.if_then_else(
                            valid,
                            x[n, ih, iw, c_idx],
                            T.Cast(dtype, 0),
                        )
                    else:
                        A_s[i, j] = T.Cast(dtype, 0)

                # Load weight tile to shared
                T.copy(w_flat[ko * block_K, bx * block_N], B_s)

                # GEMM
                T.gemm(A_s, B_s, C_frag)

            # Store results
            T.copy(C_frag, out_s)
            T.copy(out_s, y_flat[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert groups == 1, "Grouped convolution not supported."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = kernel_size
        self.stride_h = stride if isinstance(stride, int) else stride[0]
        self.stride_w = stride if isinstance(stride, int) else stride[1]
        self.pad_h = padding if isinstance(padding, int) else padding[0]
        self.pad_w = padding if isinstance(padding, int) else padding[1]
        self.dil_h = dilation if isinstance(dilation, int) else dilation[0]
        self.dil_w = dilation if isinstance(dilation, int) else dilation[1]
        self.bias_flag = bias

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kh, self.kw)
        )
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias_flag:
            fan_in = in_channels * self.kh * self.kw
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            with torch.no_grad():
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self._kernels = {}

    def _get_kernel(self, N, C, H, W, dtype):
        key = (N, C, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = conv2d_kernel(
                N,
                C,
                H,
                W,
                self.out_channels,
                self.kh,
                self.kw,
                self.stride_h,
                self.stride_w,
                self.pad_h,
                self.pad_w,
                self.dil_h,
                self.dil_w,
            )
        return self._kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = x_fp16.shape
        kernel_fn = self._get_kernel(N, C, H, W, x_fp16.dtype)

        # Layout transforms
        x_nhwc = x_fp16.permute(0, 2, 3, 1).contiguous()
        w_hwcf = w_fp16.permute(2, 3, 1, 0).contiguous()

        y_nhwc = kernel_fn(x_nhwc, w_hwcf)
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)

        return y