"""
Problem Name: 97_Conv2d_HardSwish_GlobalAvgPool_Sum_HardSwish
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.97 runtime_stats={'mean': 4.97, 'std': 0.0124, 'min': 4.95, 'max': 5.04, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.933, 'std': 0.00286, 'min': 0.929, 'max': 0.942, 'num_trials': 100}, 'speedup_ratio': 0.188}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# TileLang kernel builders
# ---------------------------------------------------------------------------

def _build_conv_hswish_sum_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    block_size: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    Hout = Hin - K + 1
    Wout = Win - K + 1
    TOTAL = N * Hout * Wout * Cout

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv_hswish_sum(
        X: T.Tensor((N, Hin, Win, Cin), dtype),          # NHWC
        W: T.Tensor((K, K, Cin, Cout), dtype),           # KHWCout
        B: T.Tensor((Cout,), dtype),                     # bias
        S: T.Tensor((N, Cout), accum_dtype),             # spatial sum per (n, c)
    ):
        with T.Kernel(T.ceildiv(TOTAL, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOTAL:
                oc = idx % Cout
                t1 = idx // Cout
                ow = t1 % Wout
                t2 = t1 // Wout
                oh = t2 % Hout
                n  = t2 // Hout

                acc = T.Cast(accum_dtype, 0)

                # Convolution - naive loop
                for kh in T.serial(K):
                    ih = oh + kh
                    for kw in T.serial(K):
                        iw = ow + kw
                        for ic in T.serial(Cin):
                            acc += (
                                T.Cast(accum_dtype, X[n, ih, iw, ic])
                                * T.Cast(accum_dtype, W[kh, kw, ic, oc])
                            )

                # add bias
                acc += T.Cast(accum_dtype, B[oc])

                # HardSwish : x * clamp(x+3,0,6)/6
                tmp = T.clamp(acc + 3.0, 0.0, 6.0)
                hs  = acc * tmp / 6.0

                # accumulate spatial sum
                T.atomic_add(S[n, oc], hs)

    return conv_hswish_sum


def _build_avg_hswish_kernel(
    N: int,
    Cout: int,
    spatial_size: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    denom_const = float(spatial_size)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def avg_hswish(
        S:   T.Tensor((N, Cout), accum_dtype),   # accumulated sums
        Out: T.Tensor((N, Cout), dtype),         # final output
    ):
        TOTAL = N * Cout
        with T.Kernel(T.ceildiv(TOTAL, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOTAL:
                oc = idx % Cout
                n  = idx // Cout

                val = S[n, oc] / denom_const   # average

                tmp = T.clamp(val + 3.0, 0.0, 6.0)
                hs  = val * tmp / 6.0          # second HardSwish

                Out[n, oc] = T.Cast(dtype, hs)

    return avg_hswish


# ---------------------------------------------------------------------------
# PyTorch wrapper module
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Conv2d → HardSwish → GlobalAvgPool → Sum → HardSwish, all in TileLang
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)

        # Parameters (match nn.Conv2d defaults)
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.uniform_(self.bias, -bound, bound)

        # Kernel caches keyed by shape/dtype
        self._conv_cache = {}   # (N, H, W, dtype) → conv_hswish_sum kernel
        self._avg_cache  = {}   # (N, H, W, dtype) → avg_hswish kernel

    # ------------------------------------------------------------------ #
    def _get_kernels(self, N: int, H: int, W: int, dtype: torch.dtype):
        key = (N, H, W, dtype)
        if key not in self._conv_cache:
            tl_dtype = "float16"
            self._conv_cache[key] = _build_conv_hswish_sum_kernel(
                N, self.in_channels, H, W, self.out_channels,
                self.kernel_size, dtype=tl_dtype
            )

            Hout = H - self.kernel_size + 1
            Wout = W - self.kernel_size + 1
            spatial = Hout * Wout
            self._avg_cache[key] = _build_avg_hswish_kernel(
                N, self.out_channels, spatial, dtype=tl_dtype
            )
        return self._conv_cache[key], self._avg_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, Cin, H, W = x_f16.shape
        assert Cin == self.in_channels, "Input channel mismatch"

        # Reorder input & weight to NHWC / KHWCout
        x_nhwc = x_f16.permute(0, 2, 3, 1).contiguous()
        w_f16  = (
            self.weight
            .to(device="cuda", dtype=torch.float16)
            .permute(2, 3, 1, 0)          # K, K, Cin, Cout
            .contiguous()
        )
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16)

        conv_kernel, avg_kernel = self._get_kernels(N, H, W, x_f16.dtype)

        # Kernel 1: Conv2d + HardSwish + spatial sum  →
        sum_fp32 = conv_kernel(x_nhwc, w_f16, b_f16)   # (N, Cout) float32

        # Kernel 2: divide, HardSwish, cast           →
        out_f16 = avg_kernel(sum_fp32)                 # (N, Cout) float16

        return out_f16.to(orig_dtype)