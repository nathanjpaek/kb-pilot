"""
Problem Name: 100_ConvTranspose3d_Clamp_Min_Divide
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=7.04 runtime_stats={'mean': 7.04, 'std': 0.0293, 'min': 6.97, 'max': 7.15, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.486, 'std': 0.0387, 'min': 0.443, 'max': 0.552, 'num_trials': 100}, 'speedup_ratio': 0.069}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_transpose_conv3d_kernel(
    N: int,
    C_IN: int,
    C_OUT: int,
    D_IN: int,
    H_IN: int,
    W_IN: int,
    K: int,
    STRIDE: int,
    PADDING: int,
    MIN_VAL: float,
    DIVISOR: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    D_OUT = (D_IN - 1) * STRIDE - 2 * PADDING + K
    H_OUT = (H_IN - 1) * STRIDE - 2 * PADDING + K
    W_OUT = (W_IN - 1) * STRIDE - 2 * PADDING + K
    TOT_OUT = N * C_OUT * D_OUT * H_OUT * W_OUT

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C_IN, D_IN, H_IN, W_IN), dtype),
        Wt: T.Tensor((C_IN, C_OUT, K, K, K), dtype),
        B: T.Tensor((C_OUT,), dtype),
        Y: T.Tensor((N, C_OUT, D_OUT, H_OUT, W_OUT), dtype),
    ):
        min_const = T.Cast(accum_dtype, MIN_VAL)
        div_const = T.Cast(accum_dtype, DIVISOR)

        with T.Kernel(T.ceildiv(TOT_OUT, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            gid = bx * block_size + tx
            if gid < TOT_OUT:
                # Unravel flat index to 5-D coordinates
                rem = gid
                n = rem // (C_OUT * D_OUT * H_OUT * W_OUT)
                rem = rem % (C_OUT * D_OUT * H_OUT * W_OUT)
                co = rem // (D_OUT * H_OUT * W_OUT)
                rem = rem % (D_OUT * H_OUT * W_OUT)
                zo = rem // (H_OUT * W_OUT)
                rem = rem % (H_OUT * W_OUT)
                yo = rem // W_OUT
                xo = rem % W_OUT

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = 0.0

                for ci in range(C_IN):
                    for kd in range(K):
                        z_tmp = zo + PADDING - kd
                        if (z_tmp % STRIDE) == 0:
                            zi = z_tmp // STRIDE
                            if (zi >= 0) and (zi < D_IN):
                                for kh in range(K):
                                    y_tmp = yo + PADDING - kh
                                    if (y_tmp % STRIDE) == 0:
                                        yi = y_tmp // STRIDE
                                        if (yi >= 0) and (yi < H_IN):
                                            for kw in range(K):
                                                x_tmp = xo + PADDING - kw
                                                if (x_tmp % STRIDE) == 0:
                                                    xi = x_tmp // STRIDE
                                                    if (xi >= 0) and (xi < W_IN):
                                                        x_val = X[
                                                            n, ci, zi, yi, xi
                                                        ].astype(accum_dtype)
                                                        w_val = Wt[
                                                            ci, co, kd, kh, kw
                                                        ].astype(accum_dtype)
                                                        acc[0] += x_val * w_val

                acc[0] += B[co].astype(accum_dtype)
                acc[0] = T.max(acc[0], min_const)
                acc[0] = acc[0] / div_const
                Y[n, co, zo, yo, xo] = T.Cast(dtype, acc[0])

    return kernel


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        min_value: float,
        divisor: float,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.min_value = float(min_value)
        self.divisor = float(divisor)

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
            )
        )
        fan_in = (
            self.in_channels
            * self.kernel_size
            * self.kernel_size
            * self.kernel_size
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache
        self._kernels = {}

    def _get_kernel(self, x: torch.Tensor):
        (
            N,
            _,
            D_in,
            H_in,
            W_in,
        ) = x.shape
        key = (N, D_in, H_in, W_in, x.dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_transpose_conv3d_kernel(
                N,
                self.in_channels,
                self.out_channels,
                D_in,
                H_in,
                W_in,
                self.kernel_size,
                self.stride,
                self.padding,
                self.min_value,
                self.divisor,
            )
        return self._kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(x_fp16)
        y_fp16 = kernel(x_fp16, w_fp16, b_fp16)

        return y_fp16.to(orig_dtype)