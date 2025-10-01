"""
Problem Name: 82_ConvTranspose2d_Mean_LayerNorm_Hardtanh_LayerNorm
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.396 runtime_stats={'mean': 0.396, 'std': 0.0567, 'min': 0.351, 'max': 0.627, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.59, 'std': 0.0414, 'min': 0.532, 'max': 0.703, 'num_trials': 100}, 'speedup_ratio': 1.49}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

def _build_convtrans_mean_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    ksize: int,
    stride: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
    block_N: int = 64,
):
    # Output spatial size (padding=0, dilation=1, output_padding=0)
    Hout = (Hin - 1) * stride + ksize - 1 + 1
    Wout = (Win - 1) * stride + ksize - 1 + 1
    denom = float(Hout * Wout)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:     T.Tensor((N, Cin, Hin, Win), dtype),
        Wsum:  T.Tensor((Cin, Cout), dtype),   # weight summed over (kh, kw)
        B:     T.Tensor((Cout,), dtype),       # bias
        Out:   T.Tensor((N, Cout), dtype),     # created by TileLang
    ):
        grid_x = T.ceildiv(Cout, block_N)
        with T.Kernel(grid_x, N, threads=block_N) as (bx, by):
            tx  = T.get_thread_binding(0)
            oc  = bx * block_N + tx           # output channel
            if oc < Cout:
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = 0.0

                # loop over input channels
                for ic in range(Cin):
                    chan_sum = T.alloc_local((1,), accum_dtype)
                    chan_sum[0] = 0.0

                    # sum spatially over input
                    for h in range(Hin):
                        for w in range(Win):
                            chan_sum[0] += T.Cast(accum_dtype, X[by, ic, h, w])

                    acc[0] += chan_sum[0] * T.Cast(accum_dtype, Wsum[ic, oc])

                val = acc[0] / denom + T.Cast(accum_dtype, B[oc])
                Out[by, oc] = T.Cast(dtype, val)

    return kernel


def _build_ln_hardtanh_ln_kernel(
    N: int,
    C: int,
    eps: float = 1e-5,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    eps_const = float(eps)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:      T.Tensor((N, C), dtype),
        g1:     T.Tensor((C,), dtype),   # gamma1
        b1:     T.Tensor((C,), dtype),   # beta1
        g2:     T.Tensor((C,), dtype),   # gamma2
        b2:     T.Tensor((C,), dtype),   # beta2
        Out:    T.Tensor((N, C), dtype),
    ):
        # one block per sample for simplicity
        with T.Kernel(N, threads=1) as bx:
            n = bx

            # ----- first LayerNorm statistics --------------------------------
            mean1 = T.alloc_local((1,), accum_dtype)
            var1  = T.alloc_local((1,), accum_dtype)
            mean1[0] = 0.0
            for c in range(C):
                mean1[0] += T.Cast(accum_dtype, X[n, c])
            mean1[0] = mean1[0] / C

            var1[0] = 0.0
            for c in range(C):
                diff = T.Cast(accum_dtype, X[n, c]) - mean1[0]
                var1[0] += diff * diff
            var1[0] = var1[0] / C
            inv_std1 = 1.0 / T.sqrt(var1[0] + eps_const)

            # ----- buffer to store intermediate y ----------------------------
            y_buf = T.alloc_local((C,), accum_dtype)

            # ----- apply LN1 + Hardtanh -------------------------------------
            for c in range(C):
                norm = (T.Cast(accum_dtype, X[n, c]) - mean1[0]) * inv_std1
                val  = norm * T.Cast(accum_dtype, g1[c]) + T.Cast(accum_dtype, b1[c])
                # Hardtanh clamp [-1, 1]
                val = T.max(T.Cast(accum_dtype, -1.0), T.min(val, T.Cast(accum_dtype, 1.0)))
                y_buf[c] = val

            # ----- second LayerNorm statistics -------------------------------
            mean2 = T.alloc_local((1,), accum_dtype)
            var2  = T.alloc_local((1,), accum_dtype)
            mean2[0] = 0.0
            for c in range(C):
                mean2[0] += y_buf[c]
            mean2[0] = mean2[0] / C

            var2[0] = 0.0
            for c in range(C):
                diff2 = y_buf[c] - mean2[0]
                var2[0] += diff2 * diff2
            var2[0] = var2[0] / C
            inv_std2 = 1.0 / T.sqrt(var2[0] + eps_const)

            # ----- apply LN2 & store ----------------------------------------
            for c in range(C):
                norm2 = (y_buf[c] - mean2[0]) * inv_std2
                out_v = norm2 * T.Cast(accum_dtype, g2[c]) + T.Cast(accum_dtype, b2[c])
                Out[n, c] = T.Cast(dtype, out_v)

    return kernel


# ---------------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    TileLang-accelerated replacement for the reference model:
        ConvTranspose2d → Mean(HW) → LayerNorm → Hardtanh → LayerNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_shape):
        super().__init__()
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size  = int(kernel_size)
        self.stride       = int(stride)

        # ---------- ConvTranspose2d parameters -----------------------------
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size * kernel_size
        bound  = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(out_channels))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---------- LayerNorm #1 parameters --------------------------------
        self.ln1_weight = nn.Parameter(torch.ones(norm_shape))
        self.ln1_bias   = nn.Parameter(torch.zeros(norm_shape))

        # ---------- LayerNorm #2 parameters --------------------------------
        self.ln2_weight = nn.Parameter(torch.ones(norm_shape))
        self.ln2_bias   = nn.Parameter(torch.zeros(norm_shape))

        # kernel caches
        self._conv_cache = {}   # keyed by (N, H, W, dtype)
        self._ln_cache   = {}   # keyed by (N, dtype)

    # ------------------------------------------------------------------
    # helpers to fetch / compile kernels
    # ------------------------------------------------------------------
    def _get_conv_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._conv_cache:
            self._conv_cache[key] = _build_convtrans_mean_kernel(
                N, self.in_channels, H, W, self.out_channels,
                ksize=self.kernel_size,
                stride=self.stride,
                dtype=dtype,
            )
        return self._conv_cache[key]

    def _get_ln_kernel(self, N: int, dtype: str):
        key = (N, dtype)
        if key not in self._ln_cache:
            self._ln_cache[key] = _build_ln_hardtanh_ln_kernel(
                N, self.out_channels, dtype=dtype
            )
        return self._ln_cache[key]

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, Cin, H, W = x_f16.shape
        assert Cin == self.in_channels, "Input channel mismatch"

        # Pre-compute weight sums over (kh, kw)
        w_sum = self.weight.sum(dim=(2, 3))           # (Cin, Cout)
        w_sum_f16 = w_sum.to(device="cuda", dtype=torch.float16).contiguous()

        bias_f16 = self.bias.to(device="cuda", dtype=torch.float16)

        # ----- ConvTranspose2d + Mean(HW) --------------------------------
        conv_kernel = self._get_conv_kernel(N, H, W, "float16")
        y0 = conv_kernel(x_f16, w_sum_f16, bias_f16)   # (N, Cout)

        # ----- LayerNorm → Hardtanh → LayerNorm --------------------------
        g1 = self.ln1_weight.to(device="cuda", dtype=torch.float16)
        b1 = self.ln1_bias.to(device="cuda", dtype=torch.float16)
        g2 = self.ln2_weight.to(device="cuda", dtype=torch.float16)
        b2 = self.ln2_bias.to(device="cuda", dtype=torch.float16)

        ln_kernel = self._get_ln_kernel(N, "float16")
        y1 = ln_kernel(y0, g1, b1, g2, b2)             # (N, Cout)

        return y1.to(orig_dtype)