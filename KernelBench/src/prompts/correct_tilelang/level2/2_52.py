"""
Problem Name: 52_Conv2d_Activation_BatchNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.193 runtime_stats={'mean': 0.193, 'std': 0.0022, 'min': 0.191, 'max': 0.21, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.154, 'std': 0.00318, 'min': 0.15, 'max': 0.17, 'num_trials': 100}, 'speedup_ratio': 0.798}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# Kernel factory : Conv2d + Mish
# ---------------------------------------------------------------------------
def _build_conv_mish_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Hout = Hin - K + 1
    Wout = Win - K + 1
    TOTAL = N * Cout * Hout * Wout

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, Cin, Hin, Win), dtype),
        Wt: T.Tensor((Cout, Cin, K, K), dtype),
        B: T.Tensor((Cout,), dtype),
        Y: T.Tensor((N, Cout, Hout, Wout), dtype),
    ):
        one_f = T.Cast(accum_dtype, 1.0)

        with T.Kernel(T.ceildiv(TOTAL, block), threads=block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < TOTAL:
                ow = idx % Wout
                tmp = idx // Wout
                oh = tmp % Hout
                tmp //= Hout
                oc = tmp % Cout
                n  = tmp // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, B[oc])

                for ic in T.serial(Cin):
                    for kh in T.serial(K):
                        ih = oh + kh
                        for kw in T.serial(K):
                            iw = ow + kw
                            acc[0] += (
                                T.Cast(accum_dtype, X[n, ic, ih, iw])
                                * T.Cast(accum_dtype, Wt[oc, ic, kh, kw])
                            )

                # -------------------- Mish activation -----------------------
                v   = acc[0]
                sp  = T.log(one_f + T.exp(v))
                out = v * T.tanh(sp)

                Y[n, oc, oh, ow] = T.Cast(dtype, out)

    return kernel


# ---------------------------------------------------------------------------
# Kernel factory : BatchNorm2d (affine)
# ---------------------------------------------------------------------------
def _build_bn_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT = N * C * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:     T.Tensor((N, C, H, W), dtype),
        SCALE: T.Tensor((C,), dtype),   # gamma * inv_std
        BIAS:  T.Tensor((C,), dtype),   # beta
        MEAN:  T.Tensor((C,), dtype),   # mean
        Y:     T.Tensor((N, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(TOT, block), threads=block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < TOT:
                w  = idx % W
                idx //= W
                h  = idx % H
                idx //= H
                c  = idx % C
                n  = idx // C

                val  = X[n, c, h, w].astype(accum_dtype)
                mean = MEAN[c].astype(accum_dtype)
                sc   = SCALE[c].astype(accum_dtype)
                bs   = BIAS[c].astype(accum_dtype)

                y = (val - mean) * sc + bs
                Y[n, c, h, w] = T.Cast(dtype, y)

    return kernel


# ---------------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Conv2d → Mish → BatchNorm2d implemented with TileLang kernels.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.in_c  = int(in_channels)
        self.out_c = int(out_channels)
        self.k     = int(kernel_size)
        self.eps   = float(eps)
        self.mom   = float(momentum)

        # ------------ Conv2d parameters (PyTorch-identical init) ------------
        self.weight = nn.Parameter(
            torch.empty(self.out_c, self.in_c, self.k, self.k)
        )
        self.bias = nn.Parameter(torch.empty(self.out_c))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = self.in_c * self.k * self.k
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # --------------- BatchNorm affine parameters ------------------------
        self.bn_weight = nn.Parameter(torch.ones(self.out_c))
        self.bn_bias   = nn.Parameter(torch.zeros(self.out_c))
        self.register_buffer("running_mean", torch.zeros(self.out_c))
        self.register_buffer("running_var",  torch.ones(self.out_c))

        # --------------- kernel caches --------------------------------------
        self._conv_kernels: Dict[Tuple[int, int, int, str], callable] = {}
        self._bn_kernels:   Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------
    def _get_conv_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._conv_kernels:
            self._conv_kernels[key] = _build_conv_mish_kernel(
                N, self.in_c, H, W,
                self.out_c, self.k,
                dtype=dtype,
            )
        return self._conv_kernels[key]

    def _get_bn_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._bn_kernels:
            self._bn_kernels[key] = _build_bn_kernel(
                N, self.out_c, H, W, dtype=dtype,
            )
        return self._bn_kernels[key]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = torch.device("cuda")

        x_f16 = x.to(device=device, dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device=device, dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device=device, dtype=torch.float16).contiguous()

        N, Cin, H_in, W_in = x_f16.shape
        assert Cin == self.in_c, "in_channels mismatch"

        # --------------- Conv2d + Mish ----------------
        conv_kernel = self._get_conv_kernel(N, H_in, W_in, "float16")
        y_f16 = conv_kernel(x_f16, w_f16, b_f16)  # (N, C, Hout, Wout)

        # --------------- Batch statistics -------------
        y_f32 = y_f16.to(torch.float32)
        reduce_dims = (0, 2, 3)
        if self.training:
            batch_mean = y_f32.mean(dim=reduce_dims)
            batch_var  = y_f32.var(dim=reduce_dims, unbiased=False)

            # update running stats
            with torch.no_grad():
                self.running_mean.mul_(1 - self.mom).add_(self.mom * batch_mean)
                self.running_var.mul_(1 - self.mom).add_(self.mom * batch_var)

            mean = batch_mean
            var  = batch_var
        else:
            mean = self.running_mean
            var  = self.running_var

        inv_std = torch.rsqrt(var + self.eps)          # (C,)
        scale   = self.bn_weight * inv_std             # (C,)

        mean_f16  = mean.to(device=device, dtype=torch.float16).contiguous()
        scale_f16 = scale.to(device=device, dtype=torch.float16).contiguous()
        bias_f16  = self.bn_bias.to(device=device, dtype=torch.float16).contiguous()

        # ---------------- BatchNorm kernel ------------
        bn_kernel = self._get_bn_kernel(N, y_f16.shape[2], y_f16.shape[3], "float16")
        out_f16   = bn_kernel(y_f16, scale_f16, bias_f16, mean_f16)

        return out_f16.to(orig_dtype)