"""
Problem Name: 55_ConvTranspose2d_Sigmoid_BatchNorm
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=9.95 runtime_stats={'mean': 9.95, 'std': 0.0426, 'min': 9.89, 'max': 10.1, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.51, 'std': 0.03, 'min': 0.446, 'max': 0.645, 'num_trials': 100}, 'speedup_ratio': 0.0513}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_convT_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    kH: int,
    kW: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
    block_size: int = 256,
):
    Hout = Hin + kH - 1
    Wout = Win + kW - 1
    numel = N * Cout * Hout * Wout

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, Cin, Hin, Win), dtype),
        W: T.Tensor((Cin, Cout, kH, kW), dtype),
        Bias: T.Tensor((Cout,), dtype),
        Out: T.Tensor((N, Cout, Hout, Wout), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                n_tot = Cout * Hout * Wout
                c_tot = Hout * Wout

                n = idx // n_tot
                rem = idx - n * n_tot
                co = rem // c_tot
                rem2 = rem - co * c_tot
                ho = rem2 // Wout
                wo = rem2 - ho * Wout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = 0.0

                for ci in T.serial(Cin):
                    for kh in T.serial(kH):
                        hi = ho - kh
                        if (hi >= 0) and (hi < Hin):
                            for kw in T.serial(kW):
                                wi = wo - kw
                                if (wi >= 0) and (wi < Win):
                                    acc[0] += (
                                        X[n, ci, hi, wi].astype(accum_dtype)
                                        * W[ci, co, kh, kw].astype(accum_dtype)
                                    )

                acc[0] += Bias[co].astype(accum_dtype)
                Out[n, co, ho, wo] = T.Cast(dtype, acc[0])

    return kernel


def _build_sigmoid_reduce_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
    block_size: int = 256,
):
    numel = N * C * H * W
    one_const = 1.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),
        S1: T.Tensor((C,), accum_dtype),
        S2: T.Tensor((C,), accum_dtype),
        Y: T.Tensor((N, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                c_tot = H * W
                n = idx // (C * c_tot)
                rem = idx - n * C * c_tot
                c = rem // c_tot
                rem2 = rem - c * c_tot
                h = rem2 // W
                w = rem2 - h * W

                val32 = X[n, c, h, w].astype(accum_dtype)
                sig_val = one_const / (one_const + T.exp(-val32))
                Y[n, c, h, w] = T.Cast(dtype, sig_val)

                T.atomic_add(S1[c], sig_val)
                T.atomic_add(S2[c], sig_val * sig_val)

    return kernel


def _build_batchnorm_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
    block_size: int = 256,
):
    numel = N * C * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),
        Mean: T.Tensor((C,), accum_dtype),
        InvStd: T.Tensor((C,), accum_dtype),
        Gamma: T.Tensor((C,), dtype),
        Beta: T.Tensor((C,), dtype),
        Out: T.Tensor((N, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                c_tot = H * W
                n = idx // (C * c_tot)
                rem = idx - n * C * c_tot
                c = rem // c_tot
                rem2 = rem - c * c_tot
                h = rem2 // W
                w = rem2 - h * W

                val32 = X[n, c, h, w].astype(accum_dtype)
                norm = (val32 - Mean[c]) * InvStd[c]
                out_val = norm.astype(accum_dtype) * Gamma[c].astype(accum_dtype) + Beta[
                    c
                ].astype(accum_dtype)
                Out[n, c, h, w] = T.Cast(dtype, out_val)

    return kernel


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kH = self.kW = int(kernel_size)

        self.weight = nn.Parameter(
            torch.empty(self.in_channels, self.out_channels, self.kH, self.kW)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * self.kH * self.kW
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self.bn_weight = nn.Parameter(torch.ones(self.out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(self.out_channels))
        self.eps = 1e-5

        self._conv_kernels = {}
        self._sig_kernels = {}
        self._bn_kernels = {}

    def _get_conv_kernel(self, N: int, Hin: int, Win: int):
        key = (N, Hin, Win)
        if key not in self._conv_kernels:
            self._conv_kernels[key] = _build_convT_kernel(
                N,
                self.in_channels,
                Hin,
                Win,
                self.out_channels,
                self.kH,
                self.kW,
            )
        return self._conv_kernels[key]

    def _get_sig_kernel(self, N: int, H: int, W: int):
        key = (N, H, W)
        if key not in self._sig_kernels:
            self._sig_kernels[key] = _build_sigmoid_reduce_kernel(
                N, self.out_channels, H, W
            )
        return self._sig_kernels[key]

    def _get_bn_kernel(self, N: int, H: int, W: int):
        key = (N, H, W)
        if key not in self._bn_kernels:
            self._bn_kernels[key] = _build_batchnorm_kernel(
                N, self.out_channels, H, W
            )
        return self._bn_kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(dtype=torch.float16, device="cuda").contiguous()
        w_f16 = self.weight.to(dtype=torch.float16, device="cuda")
        b_f16 = self.bias.to(dtype=torch.float16, device="cuda")

        N, Cin, Hin, Win = x_f16.shape
        conv_kernel = self._get_conv_kernel(N, Hin, Win)
        conv_out = conv_kernel(x_f16, w_f16, b_f16)

        _, C, Hout, Wout = conv_out.shape
        sums = torch.zeros(C, dtype=torch.float32, device="cuda")
        sums_sq = torch.zeros(C, dtype=torch.float32, device="cuda")

        sig_kernel = self._get_sig_kernel(N, Hout, Wout)
        sig_out = sig_kernel(conv_out, sums, sums_sq)

        numel_per_c = N * Hout * Wout
        mean = sums / numel_per_c
        var = sums_sq / numel_per_c - mean * mean
        inv_std = 1.0 / torch.sqrt(var + self.eps)

        bn_kernel = self._get_bn_kernel(N, Hout, Wout)
        gamma_f16 = self.bn_weight.to(dtype=torch.float16, device="cuda")
        beta_f16 = self.bn_bias.to(dtype=torch.float16, device="cuda")
        out = bn_kernel(sig_out, mean, inv_std, gamma_f16, beta_f16)

        return out.to(orig_dtype)