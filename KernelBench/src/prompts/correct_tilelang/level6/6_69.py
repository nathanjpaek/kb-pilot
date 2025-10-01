"""
Problem Name: 69_Conv3d_Scaling_BatchNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.62 runtime_stats={'mean': 4.62, 'std': 0.0227, 'min': 4.61, 'max': 4.84, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 8.39, 'std': 0.0321, 'min': 8.35, 'max': 8.63, 'num_trials': 100}, 'speedup_ratio': 1.82}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                  Conv3d  +  scalar multiply   TileLang kernel               #
# --------------------------------------------------------------------------- #

def _build_conv3d_scale_kernel(
    N: int,
    Cin: int,
    Din: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    scale_val: float,
    threads_per_block: int = 256,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Dout = Din - K + 1
    Hout = Hin - K + 1
    Wout = Win - K + 1
    TOTAL = N * Cout * Dout * Hout * Wout
    GRID = (TOTAL + threads_per_block - 1) // threads_per_block

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:  T.Tensor((N, Cin, Din, Hin, Win), in_dtype),
        Wt: T.Tensor((Cout, Cin, K, K, K),     in_dtype),
        B:  T.Tensor((Cout,),                  in_dtype),
        Y:  T.Tensor((N, Cout, Dout, Hout, Wout), in_dtype),  # auto-allocated
    ):
        scl = T.Cast(accum_dtype, float(scale_val))

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOTAL:
                ow = idx % Wout
                t1 = idx // Wout
                oh = t1 % Hout
                t2 = t1 // Hout
                od = t2 % Dout
                t3 = t2 // Dout
                oc = t3 % Cout
                n  = t3 // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, B[oc])

                for ic in T.serial(Cin):
                    for kz in T.serial(K):
                        for ky in T.serial(K):
                            for kx in T.serial(K):
                                val_in = T.Cast(
                                    accum_dtype,
                                    X[n, ic, od + kz, oh + ky, ow + kx],
                                )
                                val_w = T.Cast(
                                    accum_dtype,
                                    Wt[oc, ic, kz, ky, kx],
                                )
                                acc[0] += val_in * val_w

                acc[0] *= scl
                Y[n, oc, od, oh, ow] = T.Cast(in_dtype, acc[0])

    return kernel


# --------------------------------------------------------------------------- #
#                  BatchNorm3d   TileLang kernel (no extra scale)             #
# --------------------------------------------------------------------------- #

def _build_bn_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    threads_per_block: int = 256,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOTAL = N * C * D * H * W
    GRID  = (TOTAL + threads_per_block - 1) // threads_per_block

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:      T.Tensor((N, C, D, H, W), in_dtype),
        MEAN:   T.Tensor((C,),             in_dtype),
        INVSTD: T.Tensor((C,),             in_dtype),
        GAMMA:  T.Tensor((C,),             in_dtype),
        BETA:   T.Tensor((C,),             in_dtype),
        Y:      T.Tensor((N, C, D, H, W), in_dtype),  # auto-allocated
    ):
        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOTAL:
                w = idx % W
                t1 = idx // W
                h = t1 % H
                t2 = t1 // H
                d = t2 % D
                t3 = t2 // D
                c = t3 % C
                n = t3 // C

                x_val   = X[n, c, d, h, w].astype(accum_dtype)
                mean_v  = MEAN[c].astype(accum_dtype)
                invstdv = INVSTD[c].astype(accum_dtype)
                gamma_v = GAMMA[c].astype(accum_dtype)
                beta_v  = BETA[c].astype(accum_dtype)

                norm = (x_val - mean_v) * invstdv
                outv = norm * gamma_v + beta_v

                Y[n, c, d, h, w] = T.Cast(in_dtype, outv)

    return kernel


# --------------------------------------------------------------------------- #
#                             PyTorch wrapper                                 #
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """Conv3d → (*scaling) fused inside conv kernel → BatchNorm3d via TileLang"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()

        self.in_c  = int(in_channels)
        self.out_c = int(out_channels)
        self.k     = int(kernel_size)

        # ----- Conv parameters (identical init) ------------------------- #
        w_shape = (self.out_c, self.in_c, self.k, self.k, self.k)
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_c * self.k ** 3
        bound  = 1 / math.sqrt(fan_in)
        self.conv_bias = nn.Parameter(torch.empty(self.out_c))
        nn.init.uniform_(self.conv_bias, -bound, bound)

        # ----- learnable scalar ----------------------------------------- #
        self.scaling = nn.Parameter(torch.ones(1))

        # ----- BatchNorm parameters / buffers --------------------------- #
        self.bn_weight = nn.Parameter(torch.ones(num_features))
        self.bn_bias   = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))

        self.eps = float(eps)
        self.momentum = float(momentum)

        # Kernel caches
        self._conv_kernels: Dict[Tuple, callable] = {}
        self._bn_kernels:   Dict[Tuple, callable] = {}

    # ------------------------------------------------------------------ #
    def _get_conv_kernel(self, N: int, D: int, H: int, W: int, dtype: str, scale_val: float):
        key = (N, D, H, W, dtype, scale_val)
        if key not in self._conv_kernels:
            self._conv_kernels[key] = _build_conv3d_scale_kernel(
                N,
                self.in_c,
                D,
                H,
                W,
                self.out_c,
                self.k,
                scale_val,
                in_dtype=dtype,
            )
        return self._conv_kernels[key]

    def _get_bn_kernel(self, N: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._bn_kernels:
            self._bn_kernels[key] = _build_bn_kernel(
                N, self.out_c, D, H, W, in_dtype=dtype
            )
        return self._bn_kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        device = "cuda"

        x_fp16 = x.to(device=device, dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device=device, dtype=torch.float16).contiguous()
        b_fp16 = self.conv_bias.to(device=device, dtype=torch.float16).contiguous()

        N, _, D_in, H_in, W_in = x_fp16.shape
        scale_val = float(self.scaling.item())

        # ---------------- Conv3d * scaling ------------------------------ #
        conv_kernel = self._get_conv_kernel(N, D_in, H_in, W_in, "float16", scale_val)
        y_fp16 = conv_kernel(x_fp16, w_fp16, b_fp16)  # (N,C,Dout,Hout,Wout)

        # ---------------- BatchNorm statistics -------------------------- #
        y_fp32 = y_fp16.to(torch.float32)
        dims = (0, 2, 3, 4)  # N,D,H,W
        batch_mean = y_fp32.mean(dim=dims)
        batch_var  = y_fp32.var(dim=dims, unbiased=False)

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

        mean_f16   = mean_used.to(device=device, dtype=torch.float16).contiguous()
        invstd_f16 = invstd.to(device=device, dtype=torch.float16).contiguous()
        gamma_f16  = self.bn_weight.to(device=device, dtype=torch.float16).contiguous()
        beta_f16   = self.bn_bias.to(device=device, dtype=torch.float16).contiguous()

        # ---------------- BN transform kernel --------------------------- #
        Dout = D_in - self.k + 1
        Hout = H_in - self.k + 1
        Wout = W_in - self.k + 1
        bn_kernel = self._get_bn_kernel(N, Dout, Hout, Wout, "float16")
        out_fp16 = bn_kernel(y_fp16, mean_f16, invstd_f16, gamma_f16, beta_f16)

        return out_fp16.to(orig_dtype)