"""
Problem Name: 56_conv_standard_2D__asymmetric_input__asymmetric_kernel
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.763 runtime_stats={'mean': 0.763, 'std': 0.0566, 'min': 0.692, 'max': 0.88, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.24, 'std': 0.0376, 'min': 0.196, 'max': 0.356, 'num_trials': 100}, 'speedup_ratio': 0.315}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _conv2d_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    F: int,
    KH: int,
    KW: int,
    stride_h: int,
    stride_w: int,
    dil_h: int,
    dil_w: int,
    pad_h: int,
    pad_w: int,
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

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        data: T.Tensor((N, H, W, C), dtype),
        weight: T.Tensor((KH, KW, C, F), dtype),
        output: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),
            T.ceildiv(N * OH * OW, block_M),
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_s = T.alloc_shared((block_M, block_N), dtype)

            weight_flat = T.Tensor((K_TOTAL, F), dtype, weight.data)
            output_flat = T.Tensor((N * OH * OW, F), dtype, output.data)

            T.clear(C_loc)

            for ko in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # Load data tile into shared memory (im2col on-the-fly)
                for ti, tj in T.Parallel(block_M, block_K):
                    m = by * block_M + ti
                    k = ko * block_K + tj

                    if (m < N * OH * OW) and (k < K_TOTAL):
                        n_idx = m // (OH * OW)
                        oh_idx = (m % (OH * OW)) // OW
                        ow_idx = m % OW

                        kh_idx = k // (KW * C)
                        kw_idx = (k // C) % KW
                        c_idx = k % C

                        ih = oh_idx * stride_h + kh_idx * dil_h - pad_h
                        iw = ow_idx * stride_w + kw_idx * dil_w - pad_w

                        in_bounds = (
                            (ih >= 0)
                            and (iw >= 0)
                            and (ih < H)
                            and (iw < W)
                        )

                        A_s[ti, tj] = T.if_then_else(
                            in_bounds,
                            data[n_idx, ih, iw, c_idx],
                            T.Cast(dtype, 0),
                        )
                    else:
                        A_s[ti, tj] = T.Cast(dtype, 0)

                # Load weight tile into shared memory
                T.copy(
                    weight_flat[ko * block_K, bx * block_N],
                    B_s,
                )

                # Matrix multiply
                T.gemm(A_s, B_s, C_loc)

            # Store results
            T.copy(C_loc, out_s)
            T.copy(out_s, output_flat[by * block_M, bx * block_N])

    return kernel


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert groups == 1, "Grouped convolution is not supported in this implementation."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = kernel_size
        self.stride_h, self.stride_w = stride
        self.pad_h, self.pad_w = padding
        self.dil_h, self.dil_w = dilation
        self.bias_flag = bias

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kh, self.kw)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias_flag:
            fan_in = in_channels * self.kh * self.kw
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # Cache for compiled kernels
        self._kernel_cache = {}

    def _get_kernel(self, N, C, H, W, dtype: str):
        key = (
            N,
            C,
            H,
            W,
            self.out_channels,
            self.kh,
            self.kw,
            self.stride_h,
            self.stride_w,
            self.dil_h,
            self.dil_w,
            self.pad_h,
            self.pad_w,
            dtype,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _conv2d_kernel(
                N,
                C,
                H,
                W,
                self.out_channels,
                self.kh,
                self.kw,
                self.stride_h,
                self.stride_w,
                self.dil_h,
                self.dil_w,
                self.pad_h,
                self.pad_w,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = x_fp16.shape

        kernel = self._get_kernel(N, C, H, W, str(x_fp16.dtype))

        # Reorder tensors for NHWC / HWCF layout
        x_nhwc = x_fp16.permute(0, 2, 3, 1).contiguous()
        w_hwcf = w_fp16.permute(2, 3, 1, 0).contiguous()

        out_nhwc = kernel(x_nhwc, w_hwcf)
        out_nchw = out_nhwc.permute(0, 3, 1, 2).contiguous()

        if self.bias is not None:
            out_nchw = out_nchw + self.bias.view(1, -1, 1, 1)

        return out_nchw