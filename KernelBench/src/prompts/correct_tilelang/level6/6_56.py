"""
Problem Name: 56_Conv2d_GELU_GroupNorm_Sum_Max
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=5.28 runtime_stats={'mean': 5.28, 'std': 0.0285, 'min': 5.26, 'max': 5.53, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.47, 'std': 0.0284, 'min': 4.46, 'max': 4.71, 'num_trials': 100}, 'speedup_ratio': 0.847}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :  (GN_out + residual)  →  max_{h,w}                #
# --------------------------------------------------------------------------- #
def _build_add_max_hw_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    tot_nc = N * C
    minus_inf = -3.4e38

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def add_max_hw(
        G: T.Tensor((N, C, H, W), dtype),     # GroupNorm output
        R: T.Tensor((N, C, H, W), dtype),     # residual (conv output)
        Y: T.Tensor((N, C, 1, 1), dtype),     # final result
    ):
        neg_big = T.Cast(accum_dtype, minus_inf)

        with T.Kernel(T.ceildiv(tot_nc, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < tot_nc:
                n = idx // C
                c = idx % C

                max_val = T.alloc_local((1,), accum_dtype)
                max_val[0] = neg_big

                for h in T.serial(H):
                    for w in T.serial(W):
                        val = (
                            G[n, c, h, w].astype(accum_dtype)
                            + R[n, c, h, w].astype(accum_dtype)
                        )
                        max_val[0] = T.max(max_val[0], val)

                Y[n, c, 0, 0] = T.Cast(dtype, max_val[0])

    return add_max_hw


# --------------------------------------------------------------------------- #
#                    PyTorch wrapper using TileLang                           #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → GELU → GroupNorm → (+ residual) → max(H) → max(W)
    The last three ops are fused into a TileLang kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_groups: int):
        super().__init__()

        # Conv2d (same initialisation as nn.Conv2d default)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

        # GroupNorm identical to reference
        self.gn = nn.GroupNorm(num_groups, out_channels)

        # Kernel cache  {(N,C,H,W,dtype): compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_add_max_hw_kernel(
                N, C, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---------------- PyTorch ops (conv, GELU, GN) ------------------ #
        conv_out = self.conv(x)
        gn_in    = torch.nn.functional.gelu(conv_out)
        gn_out   = self.gn(gn_in)

        # ---------------- Prepare tensors for TileLang ------------------ #
        gn_fp16 = gn_out.to(device="cuda", dtype=torch.float16).contiguous()
        res_fp16 = conv_out.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = gn_fp16.shape
        kernel = self._get_kernel(N, C, H, W, "float16")

        y_fp16 = kernel(gn_fp16, res_fp16)          # (N,C,1,1)

        return y_fp16.to(orig_dtype)