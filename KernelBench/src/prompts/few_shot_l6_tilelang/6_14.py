"""
Problem Name: 14_Conv3d_LeakyReLU_Sum_Clamp_GELU
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.136 runtime_stats={'mean': 0.136, 'std': 0.0602, 'min': 0.0941, 'max': 0.385, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.688, 'std': 0.0429, 'min': 0.629, 'max': 0.871, 'num_trials': 100}, 'speedup_ratio': 5.06}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# TileLang kernel factory ----------------------------------------------------
# ---------------------------------------------------------------------------

def _build_conv3d_fused_kernel(
    N: int,
    Cin: int,
    Din: int,
    Hin: int,
    Win: int,
    Cout: int,
    kD: int,
    kH: int,
    kW: int,
    *,
    negative_slope: float = 0.2,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    dtype: str = "float16",
    accum_dtype: str = "float32",
    block_size: int = 256,
):
    Dout = Din - kD + 1
    Hout = Hin - kH + 1
    Wout = Win - kW + 1
    numel = N * Cout * Dout * Hout * Wout

    ns_const = float(negative_slope)
    clamp_min_const = float(clamp_min)
    clamp_max_const = float(clamp_max)
    gelu_c0 = 0.7978845608028654  # sqrt(2/pi)
    gelu_c1 = 0.044715

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv3d_fused(
        X: T.Tensor((N, Cin, Din, Hin, Win), dtype),
        W_: T.Tensor((Cout, Cin, kD, kH, kW), dtype),
        B_: T.Tensor((Cout,), dtype),
        S_: T.Tensor((Cout,), dtype),                     # sum_tensor (flattened)
        Y: T.Tensor((N, Cout, Dout, Hout, Wout), dtype),  # created by TileLang
    ):
        ns = T.Cast(accum_dtype, ns_const)
        cmin = T.Cast(accum_dtype, clamp_min_const)
        cmax = T.Cast(accum_dtype, clamp_max_const)
        gc0 = T.Cast(accum_dtype, gelu_c0)
        gc1 = T.Cast(accum_dtype, gelu_c1)
        half = T.Cast(accum_dtype, 0.5)
        one  = T.Cast(accum_dtype, 1.0)

        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx

            if idx < numel:
                tmp = idx

                w_out = tmp % Wout
                tmp //= Wout
                h_out = tmp % Hout
                tmp //= Hout
                d_out = tmp % Dout
                tmp //= Dout
                co = tmp % Cout
                n  = tmp // Cout

                acc = T.Cast(accum_dtype, 0)

                for ci in T.serial(Cin):
                    for kd in T.serial(kD):
                        for kh in T.serial(kH):
                            for kw in T.serial(kW):
                                val_in = X[
                                    n,
                                    ci,
                                    d_out + kd,
                                    h_out + kh,
                                    w_out + kw,
                                ].astype(accum_dtype)
                                val_w = W_[
                                    co, ci, kd, kh, kw
                                ].astype(accum_dtype)
                                acc += val_in * val_w

                # bias
                acc += B_[co].astype(accum_dtype)

                # LeakyReLU
                acc = T.max(acc, acc * ns)

                # add sum tensor (broadcast)
                acc += S_[co].astype(accum_dtype)

                # clamp
                acc = T.max(acc, cmin)
                acc = T.min(acc, cmax)

                # GELU (tanh approximation)
                t = acc + gc1 * acc * acc * acc
                gelu = half * acc * (one + T.tanh(gc0 * t))

                Y[n, co, d_out, h_out, w_out] = T.Cast(dtype, gelu)

    return conv3d_fused


# ---------------------------------------------------------------------------
# PyTorch wrapper module -----------------------------------------------------
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    TileLang-accelerated version of the reference model
    (Conv3d → LeakyReLU → add → clamp → GELU).
    """

    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()

        # Parameters identical to nn.Conv3d defaults -------------------------
        if isinstance(kernel_size, int):
            self.kD = self.kH = self.kW = int(kernel_size)
        else:
            self.kD, self.kH, self.kW = [int(k) for k in kernel_size]

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        weight_shape = (self.out_channels,
                        self.in_channels,
                        self.kD, self.kH, self.kW)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * self.kD * self.kH * self.kW
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # sum tensor parameter ----------------------------------------------
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

        # kernel cache -------------------------------------------------------
        self._kern_cache = {}  # keyed by (N,D,H,W,dtype)

    # -----------------------------------------------------------------------
    # forward ---------------------------------------------------------------
    # -----------------------------------------------------------------------
    def _get_kernel(self, N, D, H, W, dtype: torch.dtype):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            kernel = _build_conv3d_fused_kernel(
                N=N,
                Cin=self.in_channels,
                Din=D,
                Hin=H,
                Win=W,
                Cout=self.out_channels,
                kD=self.kD,
                kH=self.kH,
                kW=self.kW,
            )
            self._kern_cache[key] = kernel
        return self._kern_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_f16 = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16, copy=False)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16, copy=False)
        s_f16 = self.sum_tensor.view(-1).to(device="cuda", dtype=torch.float16, copy=False)

        N, Cin, D, H, W = x_f16.shape
        assert Cin == self.in_channels, "Input channel mismatch"

        kernel = self._get_kernel(N, D, H, W, x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, b_f16, s_f16)

        return y_f16.to(orig_dtype)