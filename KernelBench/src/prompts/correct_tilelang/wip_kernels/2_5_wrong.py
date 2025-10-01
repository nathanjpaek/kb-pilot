import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def deconv_bias_tanh(
    N,
    C_in,
    H_in,
    W_in,
    C_out,
    kH,
    kW,
    stride=2,
    padding=1,
    output_padding=1,
    dtype="float16",
    accum_dtype="float",
    block_threads=128,
):
    H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding
    total_pixels = H_out * W_out

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((N, C_in, H_in, W_in), dtype),
        W: T.Tensor((C_in, C_out, kH, kW), dtype),
        B: T.Tensor((C_out, 1, 1), dtype),
        Y: T.Tensor((N, C_out, H_out, W_out), dtype),
    ):
        with T.Kernel(
            T.ceildiv(total_pixels, block_threads), N * C_out, threads=block_threads
        ) as (bx, by):
            tx = T.get_thread_binding(0)
            pixel_idx = bx * block_threads + tx
            if pixel_idx < total_pixels:
                oh = pixel_idx // W_out
                ow = pixel_idx % W_out
                n = by // C_out
                co = by % C_out

                acc = T.Cast(accum_dtype, 0)
                for ci in range(C_in):
                    for kh in range(kH):
                        for kw in range(kW):
                            ih_tmp: T.int32 = oh + padding - kh
                            iw_tmp: T.int32 = ow + padding - kw
                            cond_stride = ((ih_tmp % stride) == 0) and (
                                (iw_tmp % stride) == 0
                            )
                            ih: T.int32 = ih_tmp // stride
                            iw: T.int32 = iw_tmp // stride
                            in_bound = (
                                cond_stride
                                and (ih >= 0)
                                and (iw >= 0)
                                and (ih < H_in)
                                and (iw < W_in)
                            )
                            acc += T.if_then_else(
                                in_bound,
                                T.Cast(accum_dtype, X[n, ci, ih, iw])
                                * T.Cast(accum_dtype, W[ci, co, kh, kw]),
                                T.Cast(accum_dtype, 0),
                            )
                acc -= T.Cast(accum_dtype, B[co, 0, 0])
                acc = T.tanh(acc)
                Y[n, co, oh, ow] = T.Cast(dtype, acc)

    return main


class ModelNew(nn.Module):
    """
    Optimized model using TileLang: fused transposed convolution, bias subtraction,
    and tanh activation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias_shape,
        stride=2,
        padding=1,
        output_padding=1,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        weight = torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        bias = torch.randn(bias_shape)
        self.bias = nn.Parameter(bias)

        self._cached_kernels = {}

    def _get_kernel(self, N, H_in, W_in, dtype):
        key = (N, H_in, W_in, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = deconv_bias_tanh(
                N,
                self.in_channels,
                H_in,
                W_in,
                self.out_channels,
                self.kernel_size,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dtype="float16" if dtype == torch.float16 else "float",
            )
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16)
        weight = self.weight.to(device="cuda", dtype=torch.float16)
        bias = self.bias.to(device="cuda", dtype=torch.float16)

        N, _, H_in, W_in = x.shape
        kernel = self._get_kernel(N, H_in, W_in, x.dtype)
        y = kernel(x, weight, bias)
        return y