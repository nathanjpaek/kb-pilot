"""
Problem Name: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.516 runtime_stats={'mean': 0.516, 'std': 0.00225, 'min': 0.511, 'max': 0.522, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.601, 'std': 0.00217, 'min': 0.597, 'max': 0.609, 'num_trials': 100}, 'speedup_ratio': 1.16}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory:  AvgPool3d + clamp + softmax + scale(×2)          #
# --------------------------------------------------------------------------- #
def _build_pool_clamp_softmax_kernel(
    N: int,
    C: int,
    D_in: int,
    H_in: int,
    W_in: int,
    pool_k: int,
    clamp_min: float,
    clamp_max: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    D_out = D_in // pool_k
    H_out = H_in // pool_k
    W_out = W_in // pool_k
    spatial = N * D_out * H_out * W_out
    pool_cnt = pool_k ** 3

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D_in, H_in, W_in), dtype),
        Y: T.Tensor((N, C, D_out, H_out, W_out), dtype),
    ):
        clamp_min_c = T.Cast(accum_dtype, clamp_min)
        clamp_max_c = T.Cast(accum_dtype, clamp_max)
        pool_cnt_c  = T.Cast(accum_dtype, pool_cnt)
        two_c       = T.Cast(accum_dtype, 2.0)

        with T.Kernel(T.ceildiv(spatial, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < spatial:
                w_o = idx % W_out
                tmp1 = idx // W_out
                h_o = tmp1 % H_out
                tmp2 = tmp1 // H_out
                d_o = tmp2 % D_out
                n   = tmp2 // D_out

                d_base = d_o * pool_k
                h_base = h_o * pool_k
                w_base = w_o * pool_k

                # ------------- first pass : compute sum_exp -----------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0)

                for c in T.serial(C):
                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = T.Cast(accum_dtype, 0)
                    for kd in T.serial(pool_k):
                        for kh in T.serial(pool_k):
                            for kw in T.serial(pool_k):
                                acc[0] += T.Cast(
                                    accum_dtype,
                                    X[n, c,
                                      d_base + kd,
                                      h_base + kh,
                                      w_base + kw],
                                )
                    avg = acc[0] / pool_cnt_c
                    avg = T.max(avg, clamp_min_c)
                    avg = T.min(avg, clamp_max_c)
                    sum_exp[0] += T.exp(avg)

                inv_sum = T.Cast(accum_dtype, 1.0) / sum_exp[0]

                # ------------- second pass : write results ------------------
                for c in T.serial(C):
                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = T.Cast(accum_dtype, 0)
                    for kd in T.serial(pool_k):
                        for kh in T.serial(pool_k):
                            for kw in T.serial(pool_k):
                                acc[0] += T.Cast(
                                    accum_dtype,
                                    X[n, c,
                                      d_base + kd,
                                      h_base + kh,
                                      w_base + kw],
                                )
                    avg = acc[0] / pool_cnt_c
                    avg = T.max(avg, clamp_min_c)
                    avg = T.min(avg, clamp_max_c)
                    out_val = T.exp(avg) * inv_sum * two_c
                    Y[n, c, d_o, h_o, w_o] = T.Cast(dtype, out_val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with TileLang kernels                                      #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d → fused (AvgPool3d → clamp → softmax → ×2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        pool_kernel_size: int,
        clamp_min: float,
        clamp_max: float,
    ):
        super().__init__()

        # ---------------- ConvTranspose3d parameters -----------------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias   = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size ** 3
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Store hyper-params
        self.stride         = int(stride)
        self.padding        = int(padding)
        self.output_padding = int(output_padding)
        self.pool_k         = int(pool_kernel_size)
        self.clamp_min      = float(clamp_min)
        self.clamp_max      = float(clamp_max)

        # Kernel cache
        self._kern_cache: Dict[Tuple, callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self,
        N: int,
        C: int,
        D_in: int,
        H_in: int,
        W_in: int,
        dtype: str,
    ):
        key = (N, C, D_in, H_in, W_in, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_pool_clamp_softmax_kernel(
                N,
                C,
                D_in,
                H_in,
                W_in,
                self.pool_k,
                self.clamp_min,
                self.clamp_max,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Move to CUDA and fp16
        x_fp16 = x.to(device="cuda", dtype=torch.float16)

        # ---------------- ConvTranspose3d (PyTorch) -----------------------
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)
        y = F.conv_transpose3d(
            x_fp16,
            w_fp16,
            b_fp16,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        N, C, D_in, H_in, W_in = y.shape

        # ---------------- Fused TileLang kernel --------------------------
        kernel = self._get_kernel(N, C, D_in, H_in, W_in, "float16")
        out_fp16 = kernel(y)

        return out_fp16.to(orig_dtype)