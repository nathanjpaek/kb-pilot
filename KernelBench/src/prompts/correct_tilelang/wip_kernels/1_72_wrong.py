import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_conv_transpose3d_kernel(
    N,
    Cin,
    Cout,
    Din,
    Hin,
    Win,
    Kd,
    Kh,
    Kw,
    Sd,
    Sh,
    Sw,
    Pd,
    Ph,
    Pw,
    Od,
    Oh,
    Ow,
    groups,
    dtype="float16",
    accum_dtype="float32",
):
    Cin_per_g = Cin // groups
    Cout_per_g = Cout // groups

    Dout = (Din - 1) * Sd - 2 * Pd + Kd + Od
    Hout = (Hin - 1) * Sh - 2 * Ph + Kh + Oh
    Wout = (Win - 1) * Sw - 2 * Pw + Kw + Ow

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv_t3d(
        X: T.Tensor((N, Cin, Din, Hin, Win), dtype),
        W: T.Tensor((Cin, Cout_per_g, Kd, Kh, Kw), dtype),
        B: T.Tensor((Cout,), dtype),
        Out: T.Tensor((N, Cout, Dout, Hout, Wout), dtype),
    ):
        # Launch grid over spatial positions, output channels, and batch
        with T.Kernel(Dout * Hout * Wout, Cout, N, threads=64) as (bx, by, bn):
            d_out = bx // (Hout * Wout)
            rem = bx % (Hout * Wout)
            h_out = rem // Wout
            w_out = rem % Wout

            oc = by  # output channel
            g = oc // Cout_per_g  # group index

            acc = T.alloc_fragment((1,), accum_dtype)
            T.clear(acc)

            for ci_local in range(Cin_per_g):
                ci = g * Cin_per_g + ci_local
                for kd in range(Kd):
                    d_in_num = d_out + Pd - kd
                    if d_in_num % Sd == 0:
                        d_in = d_in_num // Sd
                        if 0 <= d_in < Din:
                            for kh in range(Kh):
                                h_in_num = h_out + Ph - kh
                                if h_in_num % Sh == 0:
                                    h_in = h_in_num // Sh
                                    if 0 <= h_in < Hin:
                                        for kw in range(Kw):
                                            w_in_num = w_out + Pw - kw
                                            if w_in_num % Sw == 0:
                                                w_in = w_in_num // Sw
                                                if 0 <= w_in < Win:
                                                    inp_val = T.Cast(
                                                        accum_dtype,
                                                        X[bn, ci, d_in, h_in, w_in],
                                                    )
                                                    w_val = T.Cast(
                                                        accum_dtype,
                                                        W[
                                                            ci,
                                                            oc % Cout_per_g,
                                                            kd,
                                                            kh,
                                                            kw,
                                                        ],
                                                    )
                                                    acc[0] += inp_val * w_val

            acc[0] += T.Cast(accum_dtype, B[oc])
            Out[bn, oc, d_out, h_out, w_out] = T.Cast(dtype, acc[0])

    return conv_t3d


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Re-use PyTorch’s parameter initialisation utilities
        self.weight = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels // groups,
                *kernel_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            fan_in = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

        # A simple cache so we only compile once per unique input shape
        self._kernel_cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Promote inputs / params to fp16 on GPU
        x16 = x.to(device="cuda", dtype=torch.float16)
        w16 = self.weight.to(device="cuda", dtype=torch.float16)
        if self.bias is None:
            b16 = torch.zeros(
                self.out_channels, device="cuda", dtype=torch.float16
            )
        else:
            b16 = self.bias.to(device="cuda", dtype=torch.float16)

        N, _, Din, Hin, Win = x16.shape
        (
            Sd,
            Sh,
            Sw,
        ) = self.stride
        (
            Pd,
            Ph,
            Pw,
        ) = self.padding
        (
            Od,
            Oh,
            Ow,
        ) = self.output_padding
        Kd, Kh, Kw = self.kernel_size

        key = (N, Din, Hin, Win)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose3d_kernel(
                N,
                self.in_channels,
                self.out_channels,
                Din,
                Hin,
                Win,
                Kd,
                Kh,
                Kw,
                Sd,
                Sh,
                Sw,
                Pd,
                Ph,
                Pw,
                Od,
                Oh,
                Ow,
                self.groups,
            )

        out16 = self._kernel_cache[key](x16, w16, b16)
        return out16
