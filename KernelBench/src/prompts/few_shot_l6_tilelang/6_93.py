"""
Problem Name: 93_ConvTranspose3d_Tanh_AvgPool_BiasAdd
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=5.85 runtime_stats={'mean': 5.85, 'std': 0.0555, 'min': 5.71, 'max': 6.08, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.12, 'std': 0.00725, 'min': 2.1, 'max': 2.14, 'num_trials': 100}, 'speedup_ratio': 0.362}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------
def _build_conv_transpose3d_tanh_kernel(
    N: int,
    Ci: int,
    Di: int,
    Hi: int,
    Wi: int,
    Co: int,
    K: int,
    stride: int,
    pad: int,
    block_size: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Do = (Di - 1) * stride - 2 * pad + K
    Ho = (Hi - 1) * stride - 2 * pad + K
    Wo = (Wi - 1) * stride - 2 * pad + K
    total_out = N * Co * Do * Ho * Wo

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, Ci, Di, Hi, Wi), dtype),
        W: T.Tensor((Ci, Co, K, K, K), dtype),
        B: T.Tensor((Co,), dtype),
        Y: T.Tensor((N, Co, Do, Ho, Wo), dtype),
    ):
        with T.Kernel(T.ceildiv(total_out, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total_out:
                t = idx
                wo = t % Wo
                t //= Wo
                ho = t % Ho
                t //= Ho
                do = t % Do
                t //= Do
                co = t % Co
                n = t // Co

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for ci in T.serial(Ci):
                    for kd in T.serial(K):
                        din_nom = do + pad - kd
                        if (din_nom % stride) == 0:
                            di = din_nom // stride
                            if (0 <= di) and (di < Di):
                                for kh in T.serial(K):
                                    hin_nom = ho + pad - kh
                                    if (hin_nom % stride) == 0:
                                        hi = hin_nom // stride
                                        if (0 <= hi) and (hi < Hi):
                                            for kw in T.serial(K):
                                                win_nom = wo + pad - kw
                                                if (win_nom % stride) == 0:
                                                    wi = win_nom // stride
                                                    if (0 <= wi) and (wi < Wi):
                                                        acc[0] += (
                                                            X[n, ci, di, hi, wi].astype(accum_dtype)
                                                            * W[ci, co, kd, kh, kw].astype(accum_dtype)
                                                        )

                val = acc[0] + B[co].astype(accum_dtype)
                val = T.tanh(val)
                Y[n, co, do, ho, wo] = T.Cast(dtype, val)

    return kernel


def _build_avgpool_bias_kernel(
    N: int,
    Co: int,
    Di: int,
    Hi: int,
    Wi: int,
    pool: int,
    block_size: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Do = Di // pool
    Ho = Hi // pool
    Wo = Wi // pool
    total_out = N * Co * Do * Ho * Wo
    norm_factor = 1.0 / (pool ** 3)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, Co, Di, Hi, Wi), dtype),
        B: T.Tensor((Co, 1, 1, 1), dtype),
        Y: T.Tensor((N, Co, Do, Ho, Wo), dtype),
    ):
        with T.Kernel(T.ceildiv(total_out, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total_out:
                t = idx
                wo = t % Wo
                t //= Wo
                ho = t % Ho
                t //= Ho
                do = t % Do
                t //= Do
                co = t % Co
                n = t // Co

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for kd in T.serial(pool):
                    for kh in T.serial(pool):
                        for kw in T.serial(pool):
                            acc[0] += X[
                                n,
                                co,
                                do * pool + kd,
                                ho * pool + kh,
                                wo * pool + kw,
                            ].astype(accum_dtype)

                acc[0] = acc[0] * norm_factor + B[co, 0, 0, 0].astype(accum_dtype)
                Y[n, co, do, ho, wo] = T.Cast(dtype, acc[0])

    return kernel


# ---------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias_shape,
        pool_size,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.pool_size = int(pool_size)

        # ConvTranspose3d parameters
        self.weight = nn.Parameter(
            torch.empty(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
            )
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * (self.kernel_size ** 3)
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Post-pooling bias
        self.post_bias = nn.Parameter(torch.randn(bias_shape))

        self._conv_kernels = {}
        self._pool_kernels = {}

    # -----------------------------------------------------------------
    def _get_conv_kernel(self, N, Di, Hi, Wi, dtype):
        key = (N, Di, Hi, Wi, dtype)
        if key not in self._conv_kernels:
            self._conv_kernels[key] = _build_conv_transpose3d_tanh_kernel(
                N,
                self.in_channels,
                Di,
                Hi,
                Wi,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                dtype=dtype,
            )
        return self._conv_kernels[key]

    def _get_pool_kernel(self, N, Do, Ho, Wo, dtype):
        key = (N, Do, Ho, Wo, dtype)
        if key not in self._pool_kernels:
            self._pool_kernels[key] = _build_avgpool_bias_kernel(
                N,
                self.out_channels,
                Do,
                Ho,
                Wo,
                self.pool_size,
                dtype=dtype,
            )
        return self._pool_kernels[key]

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        w = self.weight.to(device="cuda", dtype=torch.float16, copy=False)
        b = self.bias.to(device="cuda", dtype=torch.float16, copy=False)
        post_b = self.post_bias.to(device="cuda", dtype=torch.float16, copy=False)

        N, _, Di, Hi, Wi = x.shape

        conv_kernel = self._get_conv_kernel(N, Di, Hi, Wi, "float16")
        y_mid = conv_kernel(x, w, b)

        _, _, Do, Ho, Wo = y_mid.shape
        pool_kernel = self._get_pool_kernel(N, Do, Ho, Wo, "float16")
        y = pool_kernel(y_mid, post_b)

        return y.to(orig_dtype)