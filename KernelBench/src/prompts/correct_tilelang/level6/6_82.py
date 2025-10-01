"""
Problem Name: 82_ConvTranspose2d_Mean_LayerNorm_Hardtanh_LayerNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.86 runtime_stats={'mean': 3.86, 'std': 0.0118, 'min': 3.84, 'max': 3.91, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.607, 'std': 0.0226, 'min': 0.598, 'max': 0.827, 'num_trials': 100}, 'speedup_ratio': 0.157}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                     Kernel factory : fused LN-HT-LN                         #
# --------------------------------------------------------------------------- #

def _build_fused_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    eps: float = 1e-5,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    inv_hw = 1.0 / float(H * W)
    C_f    = float(C)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:      T.Tensor((N, C, H, W), dtype),         # (N,C,H,W)
        G1:     T.Tensor((C,), dtype),                 # gamma₁
        B1:     T.Tensor((C,), dtype),                 # beta₁
        G2:     T.Tensor((C,), dtype),                 # gamma₂
        B2:     T.Tensor((C,), dtype),                 # beta₂
        Out:    T.Tensor((N, C), dtype),               # (N,C) output
    ):
        eps_c  = T.Cast(accum_dtype, eps)
        inv_hw_c = T.Cast(accum_dtype, inv_hw)
        C_c    = T.Cast(accum_dtype, C_f)
        h_min  = T.Cast(accum_dtype, -1.0)
        h_max  = T.Cast(accum_dtype,  1.0)

        # one CUDA block per sample, single thread (loops inside)
        with T.Kernel(N) as bn:
            n = bn

            # ----------------------------------------------------------------
            # Step-1 : spatial mean per channel  →  tmp[C]
            # ----------------------------------------------------------------
            sp_mean = T.alloc_local((C,), accum_dtype)
            for c in T.serial(C):
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                for h in T.serial(H):
                    for w in T.serial(W):
                        acc[0] += T.Cast(accum_dtype, X[n, c, h, w])
                sp_mean[c] = acc[0] * inv_hw_c

            # ----------------------------------------------------------------
            # Step-2 : LayerNorm-1 (over C)
            # ----------------------------------------------------------------
            mu1 = T.alloc_local((1,), accum_dtype)
            mu1[0] = T.Cast(accum_dtype, 0)
            for c in T.serial(C):
                mu1[0] += sp_mean[c]
            mu1[0] = mu1[0] / C_c

            var1 = T.alloc_local((1,), accum_dtype)
            var1[0] = T.Cast(accum_dtype, 0)
            for c in T.serial(C):
                diff = sp_mean[c] - mu1[0]
                var1[0] += diff * diff
            var1[0] = var1[0] / C_c
            rstd1 = T.rsqrt(var1[0] + eps_c)

            ln1 = T.alloc_local((C,), accum_dtype)   # store after LN1 & HT
            for c in T.serial(C):
                gamma1 = T.Cast(accum_dtype, G1[c])
                beta1  = T.Cast(accum_dtype, B1[c])
                norm = (sp_mean[c] - mu1[0]) * rstd1
                val  = norm * gamma1 + beta1
                # HardTanh
                val = T.max(val, h_min)
                val = T.min(val, h_max)
                ln1[c] = val

            # ----------------------------------------------------------------
            # Step-3 : LayerNorm-2 (over C)
            # ----------------------------------------------------------------
            mu2 = T.alloc_local((1,), accum_dtype)
            mu2[0] = T.Cast(accum_dtype, 0)
            for c in T.serial(C):
                mu2[0] += ln1[c]
            mu2[0] = mu2[0] / C_c

            var2 = T.alloc_local((1,), accum_dtype)
            var2[0] = T.Cast(accum_dtype, 0)
            for c in T.serial(C):
                diff = ln1[c] - mu2[0]
                var2[0] += diff * diff
            var2[0] = var2[0] / C_c
            rstd2 = T.rsqrt(var2[0] + eps_c)

            for c in T.serial(C):
                gamma2 = T.Cast(accum_dtype, G2[c])
                beta2  = T.Cast(accum_dtype, B2[c])
                norm2 = (ln1[c] - mu2[0]) * rstd2
                out_val = norm2 * gamma2 + beta2
                Out[n, c] = T.Cast(dtype, out_val)

    return kernel


# --------------------------------------------------------------------------- #
#                                PyTorch wrapper                              #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d  →  mean over (H,W)  →  LayerNorm  →  HardTanh  →  LayerNorm
    All steps after ConvTranspose2d are fused into one TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_shape: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        # ---------------- ConvTranspose2d parameters ----------------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias   = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---------------- LayerNorm parameters ---------------------------
        # First LN
        self.gamma1 = nn.Parameter(torch.ones(norm_shape))
        self.beta1  = nn.Parameter(torch.zeros(norm_shape))
        # Second LN
        self.gamma2 = nn.Parameter(torch.ones(norm_shape))
        self.beta2  = nn.Parameter(torch.zeros(norm_shape))

        self.eps = float(eps)

        # Conv hyper-params
        self.stride = int(stride)

        # Kernel cache :  {(N,C,H,W,dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple, callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
                N, C, H, W, eps=self.eps, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---------------- ConvTranspose2d ------------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)
        y = F.conv_transpose2d(x_fp16, w_fp16, b_fp16, stride=self.stride)
        N, C, H, W = y.shape

        # ---------------- Fused kernel ---------------------------------
        g1 = self.gamma1.to(device="cuda", dtype=torch.float16).contiguous()
        b1 = self.beta1.to(device="cuda", dtype=torch.float16).contiguous()
        g2 = self.gamma2.to(device="cuda", dtype=torch.float16).contiguous()
        b2 = self.beta2.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(N, C, H, W, "float16")
        out_fp16 = kernel(y.contiguous(), g1, b1, g2, b2)

        return out_fp16.to(orig_dtype)