"""
Problem Name: 100_Conv3d_Mish_BatchNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.5 runtime_stats={'mean': 3.5, 'std': 0.0066, 'min': 3.49, 'max': 3.52, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 8.36, 'std': 0.0218, 'min': 8.32, 'max': 8.42, 'num_trials': 100}, 'speedup_ratio': 2.39}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel : Mish activation                                           #
# --------------------------------------------------------------------------- #

def _build_mish_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block_size: int = 256,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOTAL = N * C * D * H * W
    GRID = (TOTAL + block_size - 1) // block_size

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D, H, W), in_dtype),
        Y: T.Tensor((N, C, D, H, W), in_dtype),  # auto-allocated
    ):
        one = T.Cast(accum_dtype, 1.0)
        with T.Kernel(GRID, threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOTAL:
                w = idx % W
                t1 = idx // W
                h = t1 % H
                t2 = t1 // H
                d = t2 % D
                t3 = t2 // D
                c = t3 % C
                n = t3 // C

                xval = T.Cast(accum_dtype, X[n, c, d, h, w])
                sp = T.log(one + T.exp(xval))
                out = xval * T.tanh(sp)  # Mish
                Y[n, c, d, h, w] = T.Cast(in_dtype, out)

    return kernel


# --------------------------------------------------------------------------- #
# TileLang kernel : BatchNorm3d (per-channel)                                 #
# --------------------------------------------------------------------------- #

def _build_bn_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block_size: int = 256,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOTAL = N * C * D * H * W
    GRID = (TOTAL + block_size - 1) // block_size

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:      T.Tensor((N, C, D, H, W), in_dtype),  # Mish output
        MEAN:   T.Tensor((C,), in_dtype),
        INVSTD: T.Tensor((C,), in_dtype),
        GAMMA:  T.Tensor((C,), in_dtype),
        BETA:   T.Tensor((C,), in_dtype),
        Y:      T.Tensor((N, C, D, H, W), in_dtype),  # auto-allocated
    ):
        with T.Kernel(GRID, threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOTAL:
                w = idx % W
                t1 = idx // W
                h = t1 % H
                t2 = t1 // H
                d = t2 % D
                t3 = t2 // D
                c = t3 % C
                n = t3 // C

                val = X[n, c, d, h, w].astype(accum_dtype)
                norm = (val - MEAN[c].astype(accum_dtype)) * INVSTD[c].astype(accum_dtype)
                outv = norm * GAMMA[c].astype(accum_dtype) + BETA[c].astype(accum_dtype)
                Y[n, c, d, h, w] = T.Cast(in_dtype, outv)

    return kernel


# --------------------------------------------------------------------------- #
#                             PyTorch wrapper                                 #
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """Conv3d  →  Mish  →  BatchNorm3d (TileLang-accelerated)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.in_c = int(in_channels)
        self.out_c = int(out_channels)
        self.k = int(kernel_size)

        # ---- Conv3d parameters (identical init) ------------------------- #
        w_shape = (self.out_c, self.in_c, self.k, self.k, self.k)
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_c * self.k ** 3
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(self.out_c))
        nn.init.uniform_(self.bias, -bound, bound)

        # ---- BatchNorm learnable params --------------------------------- #
        self.bn_weight = nn.Parameter(torch.ones(self.out_c))
        self.bn_bias = nn.Parameter(torch.zeros(self.out_c))
        self.register_buffer("running_mean", torch.zeros(self.out_c))
        self.register_buffer("running_var", torch.ones(self.out_c))
        self.eps = float(eps)
        self.momentum = float(momentum)

        # Kernel caches
        self._mish_kernels: Dict[Tuple, callable] = {}
        self._bn_kernels: Dict[Tuple, callable] = {}

    # ------------------------------------------------------------------ #
    def _get_mish_kernel(self, N: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._mish_kernels:
            self._mish_kernels[key] = _build_mish_kernel(N, self.out_c, D, H, W, in_dtype=dtype)
        return self._mish_kernels[key]

    def _get_bn_kernel(self, N: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._bn_kernels:
            self._bn_kernels[key] = _build_bn_kernel(N, self.out_c, D, H, W, in_dtype=dtype)
        return self._bn_kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = "cuda"

        # ---------------- Conv3d ---------------------------------------- #
        x_fp16 = x.to(device=device, dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device=device, dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device=device, dtype=torch.float16).contiguous()
        x_fp16 = F.conv3d(x_fp16, w_fp16, b_fp16, stride=1, padding=0)

        N, C, D, H, W = x_fp16.shape  # C == out_c

        # ---------------- Mish activation (TileLang) -------------------- #
        mish_kernel = self._get_mish_kernel(N, D, H, W, "float16")
        x_fp16 = mish_kernel(x_fp16.contiguous())  # now holds Mish output

        # ---------------- BatchNorm statistics -------------------------- #
        x_fp32 = x_fp16.to(torch.float32)
        dims = (0, 2, 3, 4)  # exclude channel
        batch_mean = x_fp32.mean(dim=dims)
        batch_var = x_fp32.var(dim=dims, unbiased=False)

        if self.training:
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
            mean_used = batch_mean
            var_used = batch_var
        else:
            mean_used = self.running_mean
            var_used = self.running_var

        invstd = torch.rsqrt(var_used + self.eps)

        mean_f16 = mean_used.to(device=device, dtype=torch.float16).contiguous()
        invstd_f16 = invstd.to(device=device, dtype=torch.float16).contiguous()
        gamma_f16 = self.bn_weight.to(device=device, dtype=torch.float16).contiguous()
        beta_f16 = self.bn_bias.to(device=device, dtype=torch.float16).contiguous()

        # ---------------- BN transform (TileLang) ----------------------- #
        bn_kernel = self._get_bn_kernel(N, D, H, W, "float16")
        out_fp16 = bn_kernel(x_fp16.contiguous(), mean_f16, invstd_f16, gamma_f16, beta_f16)

        return out_fp16.to(orig_dtype)