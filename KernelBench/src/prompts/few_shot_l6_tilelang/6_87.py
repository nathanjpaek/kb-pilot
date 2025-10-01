"""
Problem Name: 87_Conv3d_Tanh_Clamp_Sigmoid_Divide_Swish
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.78 runtime_stats={'mean': 1.78, 'std': 0.0163, 'min': 1.77, 'max': 1.82, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.41, 'std': 0.0165, 'min': 2.39, 'max': 2.52, 'num_trials': 100}, 'speedup_ratio': 1.35}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------
# TileLang kernel factory
# ----------------------------------------------------------------------
def _build_conv3d_fused_kernel(
    N: int,
    Cin: int,
    Din: int,
    Hin: int,
    Win: int,
    Cout: int,
    Kd: int,
    Kh: int,
    Kw: int,
    clamp_min: float,
    clamp_max: float,
    *,
    block_size: int = 256,
    in_dtype: str = "float16",
    accum_dtype: str = "float",
):
    Dout = Din - Kd + 1
    Hout = Hin - Kh + 1
    Wout = Win - Kw + 1
    numel = N * Cout * Dout * Hout * Wout

    cmin = float(clamp_min)
    cmax = float(clamp_max)
    eps  = 1e-5

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, Cin, Din, Hin, Win), in_dtype),
        W: T.Tensor((Cout, Cin, Kd, Kh, Kw), in_dtype),
        B: T.Tensor((Cout,), in_dtype),
        Y: T.Tensor((N, Cout, Dout, Hout, Wout), in_dtype),
    ):
        one = T.Cast(accum_dtype, 1.0)
        two = T.Cast(accum_dtype, 2.0)
        cmin_c = T.Cast(accum_dtype, cmin)
        cmax_c = T.Cast(accum_dtype, cmax)
        eps_c  = T.Cast(accum_dtype, eps)

        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            gid = bx * block_size + tx
            if gid < numel:
                # ---------------------------------------------
                # unravel flat index -> (n, oc, od, oh, ow)
                # ---------------------------------------------
                tmp = gid
                ow  = tmp % Wout
                tmp //= Wout
                oh  = tmp % Hout
                tmp //= Hout
                od  = tmp % Dout
                tmp //= Dout
                oc  = tmp % Cout
                n   = tmp // Cout

                # ---------------------------------------------
                # local accumulator
                # ---------------------------------------------
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for ic in T.serial(Cin):
                    for kd in T.serial(Kd):
                        id_in = od + kd
                        for kh in T.serial(Kh):
                            ih_in = oh + kh
                            for kw_ in T.serial(Kw):
                                iw_in = ow + kw_
                                acc[0] += (
                                    X[n, ic, id_in, ih_in, iw_in].astype(accum_dtype)
                                    * W[oc, ic, kd, kh, kw_].astype(accum_dtype)
                                )

                acc[0] += B[oc].astype(accum_dtype)

                # ------------- activation & fused ops ----------------
                # tanh via exp trick to avoid missing intrinsic
                exp_val = T.exp(acc[0] * two)
                val = (exp_val - one) / (exp_val + one)          # tanh

                # clamp
                val = T.clamp(val, cmin_c, cmax_c)

                # sigmoid(s)
                s = one / (one + T.exp(-val))

                # sigmoid(s) again
                s2 = one / (one + T.exp(-s))

                div = s / (s2 + eps_c)

                # sigmoid(div)
                s3 = one / (one + T.exp(-div))

                out_val = div * s3

                Y[n, oc, od, oh, ow] = T.Cast(in_dtype, out_val)

    return kernel


# ----------------------------------------------------------------------
# PyTorch wrapper module
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    TileLang-accelerated model performing:
        Conv3d → tanh → clamp → sigmoid chain (fused as described).
    """

    def __init__(self, in_channels, out_channels, kernel_size, clamp_min, clamp_max):
        super().__init__()

        # ---------------- parameters (same as nn.Conv3d defaults) ------------
        if isinstance(kernel_size, int):
            kd = kh = kw = kernel_size
        else:
            kd, kh, kw = kernel_size
        self.kernel_size = (kd, kh, kw)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kd, kh, kw)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kd * kh * kw
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # kernel cache keyed by dynamic shapes & dtype
        self._kernel_cache = {}

    # ------------------------------------------------------------------
    def _get_kernel(
        self,
        N: int,
        Din: int,
        Hin: int,
        Win: int,
        dtype: torch.dtype,
    ):
        kd, kh, kw = self.kernel_size
        key = (N, Din, Hin, Win, dtype)
        if key not in self._kernel_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._kernel_cache[key] = _build_conv3d_fused_kernel(
                N,
                self.in_channels,
                Din,
                Hin,
                Win,
                self.out_channels,
                kd,
                kh,
                kw,
                self.clamp_min,
                self.clamp_max,
                in_dtype=tl_dtype,
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_f16 = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16, copy=False).contiguous()

        N, Cin, Din, Hin, Win = x_f16.shape
        kernel = self._get_kernel(N, Din, Hin, Win, x_f16.dtype)

        y_f16 = kernel(x_f16, w_f16, b_f16)
        return y_f16.to(orig_dtype)