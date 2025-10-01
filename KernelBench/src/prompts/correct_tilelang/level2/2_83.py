"""
Problem Name: 83_Conv3d_GroupNorm_Min_Clamp_Dropout
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.738 runtime_stats={'mean': 0.738, 'std': 0.0041, 'min': 0.733, 'max': 0.755, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.753, 'std': 0.00523, 'min': 0.745, 'max': 0.778, 'num_trials': 100}, 'speedup_ratio': 1.02}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :   torch.min(·, m) → clamp[ m , M ]                #
# --------------------------------------------------------------------------- #
def _build_min_clamp_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    MIN_VAL: float,
    MAX_VAL: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT = N * C * D * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X_flat: T.Tensor((TOT,), dtype),
        Y_flat: T.Tensor((TOT,), dtype),   # auto-allocated by jit
    ):
        min_c = T.Cast(accum_dtype, MIN_VAL)
        max_c = T.Cast(accum_dtype, MAX_VAL)

        with T.Kernel(T.ceildiv(TOT, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                val = X_flat[idx].astype(accum_dtype)
                val = T.min(val, min_c)       # element-wise minimum
                val = T.max(val, min_c)       # clamp lower bound
                val = T.min(val, max_c)       # clamp upper bound
                Y_flat[idx] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch module with fused TileLang kernel                                   #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d → GroupNorm → (TileLang fused min+clamp) → Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        min_value: float,
        max_value: float,
        dropout_p: float,
    ):
        super().__init__()
        # Fast cuDNN layers kept as-is
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)

        # Constants for element-wise kernel
        self._min_val = float(min_value)
        self._max_val = float(max_value)

        # Kernel cache :  (N,C,D,H,W,dtype) → callable
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self,
        N: int,
        C: int,
        D: int,
        H: int,
        W: int,
        dtype: str,
    ):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_min_clamp_kernel(
                N,
                C,
                D,
                H,
                W,
                self._min_val,
                self._max_val,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---- fast cuDNN layers ----------------------------------------
        y = self.conv(x)
        y = self.norm(y)

        # ---- fused TileLang kernel ------------------------------------
        y_fp16 = y.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = y_fp16.shape
        kernel = self._get_kernel(N, C, D, H, W, "float16")

        y_fp16_out = kernel(y_fp16.view(-1)).view_as(y_fp16)

        # ---- cast back + Dropout --------------------------------------
        y_out = y_fp16_out.to(dtype=orig_dtype)
        y_out = self.dropout(y_out)
        return y_out