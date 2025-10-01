"""
Problem Name: 73_Conv2d_BatchNorm_Scaling
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.19 runtime_stats={'mean': 0.19, 'std': 0.0219, 'min': 0.179, 'max': 0.375, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.133, 'std': 0.00826, 'min': 0.127, 'max': 0.176, 'num_trials': 100}, 'speedup_ratio': 0.7}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                           Conv2d  TileLang  kernel                          #
# --------------------------------------------------------------------------- #
def _build_conv2d_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    threads_per_block: int = 256,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Hout = Hin - K + 1
    Wout = Win - K + 1
    TOT  = N * Cout * Hout * Wout

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:  T.Tensor((N, Cin, Hin, Win), in_dtype),
        Wt: T.Tensor((Cout, Cin, K, K),  in_dtype),
        B:  T.Tensor((Cout,),            in_dtype),
        Y:  T.Tensor((N, Cout, Hout, Wout), in_dtype),   # created by TileLang
    ):
        with T.Kernel(T.ceildiv(TOT, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                ow = idx % Wout
                tmp = idx // Wout
                oh  = tmp % Hout
                tmp //= Hout
                oc  = tmp % Cout
                n   = tmp // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, B[oc])

                for ic in T.serial(Cin):
                    for kh in T.serial(K):
                        ih = oh + kh
                        for kw in T.serial(K):
                            iw = ow + kw
                            acc[0] += (
                                T.Cast(accum_dtype, X[n, ic, ih, iw]) *
                                T.Cast(accum_dtype, Wt[oc, ic, kh, kw])
                            )

                Y[n, oc, oh, ow] = T.Cast(in_dtype, acc[0])

    return kernel


# --------------------------------------------------------------------------- #
#                BatchNorm  +  constant-scale   TileLang kernel               #
# --------------------------------------------------------------------------- #
def _build_bn_scale_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    scaling_factor: float,
    threads_per_block: int = 256,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT = N * C * H * W
    scl = float(scaling_factor)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:     T.Tensor((N, C, H, W), in_dtype),   # conv output
        MEAN:  T.Tensor((C,),          in_dtype),
        INVSTD:T.Tensor((C,),          in_dtype),
        GAMMA: T.Tensor((C,),          in_dtype),  # BN weight
        BETA:  T.Tensor((C,),          in_dtype),  # BN bias
        Y:     T.Tensor((N, C, H, W),  in_dtype),  # output
    ):
        scl_f = T.Cast(accum_dtype, scl)

        with T.Kernel(T.ceildiv(TOT, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                w  = idx % W
                t1 = idx // W
                h  = t1 % H
                t2 = t1 // H
                c  = t2 % C
                n  = t2 // C

                x_val   = X[n, c, h, w].astype(accum_dtype)
                mean_v  = MEAN[c].astype(accum_dtype)
                invstd  = INVSTD[c].astype(accum_dtype)
                gamma_v = GAMMA[c].astype(accum_dtype)
                beta_v  = BETA[c].astype(accum_dtype)

                norm = (x_val - mean_v) * invstd
                outv = (norm * gamma_v + beta_v) * scl_f

                Y[n, c, h, w] = T.Cast(in_dtype, outv)

    return kernel


# --------------------------------------------------------------------------- #
#                              PyTorch wrapper                                #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → BatchNorm2d → multiply(const)  — heavy work in TileLang
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scaling_factor: float,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()

        self.in_c   = int(in_channels)
        self.out_c  = int(out_channels)
        self.k      = int(kernel_size)
        self.scl    = float(scaling_factor)

        self.eps     = float(eps)
        self.momentum = float(momentum)

        # ---------------- Conv parameters (identical to nn.Conv2d) -------- #
        self.weight = nn.Parameter(
            torch.empty(self.out_c, self.in_c, self.k, self.k)
        )
        self.conv_bias = nn.Parameter(torch.empty(self.out_c))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = self.in_c * self.k * self.k
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # ---------------- BatchNorm parameters / buffers ------------------ #
        self.bn_weight = nn.Parameter(torch.ones(self.out_c))   # γ
        self.bn_bias   = nn.Parameter(torch.zeros(self.out_c))  # β

        self.register_buffer("running_mean", torch.zeros(self.out_c))
        self.register_buffer("running_var",  torch.ones(self.out_c))

        # ---------------- Kernel caches ----------------------------------- #
        self._conv_kernels: Dict[Tuple[int, int, int, str], callable] = {}
        self._bn_kernels:   Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_conv_kernel(self, N: int, H_in: int, W_in: int, dtype: str):
        key = (N, H_in, W_in, dtype)
        if key not in self._conv_kernels:
            self._conv_kernels[key] = _build_conv2d_kernel(
                N, self.in_c, H_in, W_in, self.out_c, self.k, in_dtype=dtype
            )
        return self._conv_kernels[key]

    def _get_bn_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._bn_kernels:
            self._bn_kernels[key] = _build_bn_scale_kernel(
                N, self.out_c, H, W, self.scl, in_dtype=dtype
            )
        return self._bn_kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_f16 = self.conv_bias.to(device="cuda", dtype=torch.float16).contiguous()

        N, Cin, H_in, W_in = x_f16.shape
        assert Cin == self.in_c, "Input channel mismatch"

        # ---------------- Conv2d kernel ---------------------------------- #
        conv_kernel = self._get_conv_kernel(N, H_in, W_in, "float16")
        y_f16 = conv_kernel(x_f16, w_f16, b_f16)           # (N, C, H_out, W_out)

        # ------------------- BN statistics -------------------------------- #
        y_f32 = y_f16.to(torch.float32)
        dims  = (0, 2, 3)            # N,H,W
        batch_mean = y_f32.mean(dim=dims)
        batch_var  = y_f32.var(dim=dims, unbiased=False)

        if self.training:
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
            mean_used = batch_mean
            var_used  = batch_var
        else:
            mean_used = self.running_mean
            var_used  = self.running_var

        invstd = torch.rsqrt(var_used + self.eps)

        mean_f16   = mean_used.to(device="cuda", dtype=torch.float16).contiguous()
        invstd_f16 = invstd.to(device="cuda", dtype=torch.float16).contiguous()
        gamma_f16  = self.bn_weight.to(device="cuda", dtype=torch.float16).contiguous()
        beta_f16   = self.bn_bias.to(device="cuda", dtype=torch.float16).contiguous()

        # ---------------- BN + scale kernel ----------------------------- #
        bn_kernel = self._get_bn_kernel(N, y_f16.shape[2], y_f16.shape[3], "float16")
        out_f16   = bn_kernel(y_f16, mean_f16, invstd_f16, gamma_f16, beta_f16)

        return out_f16.to(orig_dtype)