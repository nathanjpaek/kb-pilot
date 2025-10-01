"""
Problem Name: 61_ConvTranspose3d_InstanceNorm_Softmax_Scaling
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=5.11 runtime_stats={'mean': 5.11, 'std': 0.0114, 'min': 5.1, 'max': 5.19, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 5.35, 'std': 0.0517, 'min': 5.3, 'max': 5.48, 'num_trials': 100}, 'speedup_ratio': 1.05}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :  (x * scale)  →  softmax(dim=1)                   #
# --------------------------------------------------------------------------- #
def _build_scale_softmax_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    scale_val: float,
    block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    spatial = N * D * H * W
    grid = (spatial + block - 1) // block
    scale_f = float(scale_val)
    one_f = 1.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, D, H, W), dtype),
        Out: T.Tensor((N, C, D, H, W), dtype),
    ):
        scale_c = T.Cast(accum_dtype, scale_f)
        one_c   = T.Cast(accum_dtype, one_f)

        with T.Kernel(grid, threads=block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < spatial:
                w  = idx % W
                tmp = idx // W
                h  = tmp % H
                tmp //= H
                d  = tmp % D
                n  = tmp // D

                # -------- pass 1 : denominator ----------------------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0)
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w]) * scale_c
                    sum_exp[0] += T.exp(val)

                inv_sum = one_c / sum_exp[0]

                # -------- pass 2 : final output ---------------------------
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w]) * scale_c
                    sm  = T.exp(val) * inv_sum
                    Out[n, c, d, h, w] = T.Cast(dtype, sm)

    return kernel


# --------------------------------------------------------------------------- #
# Optimised PyTorch module                                                    #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  InstanceNorm3d  →  (*scale) + softmax(dim=1) fused in TileLang
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        instance_norm_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        # ---------------- ConvTranspose3d parameters -----------------------
        w_shape = (
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias   = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size ** 3
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---------------- InstanceNorm settings ---------------------------
        self.eps = float(eps)  # same default as PyTorch
        # (affine=False in original model → no extra params)

        # ---------------- learnable scaling ------------------------------
        self.scaling = nn.Parameter(torch.ones(1))

        # kernel cache :  {(N,C,D,H,W,dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple, callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str, scale_val: float):
        key = (N, C, D, H, W, dtype, scale_val)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_scale_softmax_kernel(
                N, C, D, H, W, scale_val, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(device="cuda", dtype=torch.float16)

        # ---------------- ConvTranspose3d (cuDNN) ------------------------
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)
        y = F.conv_transpose3d(x, w_fp16, b_fp16)

        # ---------------- InstanceNorm3d (same semantics) ----------------
        # compute per-instance, per-channel mean/var over D,H,W
        mean = y.mean(dim=(2, 3, 4), keepdim=True)
        var  = y.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
        y = (y - mean) / torch.sqrt(var + self.eps)

        # ---------------- fused scale × softmax kernel -------------------
        N, C, D, H, W = y.shape
        scale_val = float(self.scaling.item())
        kernel = self._get_kernel(N, C, D, H, W, "float16", scale_val)
        y_fp16 = y.contiguous()
        out_fp16 = kernel(y_fp16)

        return out_fp16.to(orig_dtype)