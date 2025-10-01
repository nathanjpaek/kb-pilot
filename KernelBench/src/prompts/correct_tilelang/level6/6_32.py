"""
Problem Name: 32_ConvTranspose2d_GELU_Scale_Add_Add_GELU
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.06 runtime_stats={'mean': 2.06, 'std': 0.00528, 'min': 2.05, 'max': 2.07, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.38, 'std': 0.00898, 'min': 2.37, 'max': 2.45, 'num_trials': 100}, 'speedup_ratio': 1.16}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : GELU → +scale*bias +add → GELU                    #
# --------------------------------------------------------------------------- #
def _build_post_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT = N * C * H * W
    GRID = (TOT + threads_per_block - 1) // threads_per_block
    INV_SQRT2 = 0.7071067811865476  # 1 / sqrt(2)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:     T.Tensor((N, C, H, W), dtype),   # conv-transpose output
        Bias:  T.Tensor((C,),           dtype), # learnable bias
        Add:   T.Tensor((C,),           dtype), # second add tensor
        Scale: T.Tensor((1,),           dtype), # scalar scale
        Out:   T.Tensor((N, C, H, W),   dtype), # final output
    ):
        half_f       = T.Cast(accum_dtype, 0.5)
        one_f        = T.Cast(accum_dtype, 1.0)
        inv_sqrt2_f  = T.Cast(accum_dtype, INV_SQRT2)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                w  = idx % W
                t1 = idx // W
                h  = t1 % H
                t2 = t1 // H
                c  = t2 % C
                n  = t2 // C

                x_val_f32 = T.Cast(accum_dtype, X[n, c, h, w])

                # ------------------- first GELU ------------------------- #
                gelu1 = half_f * x_val_f32 * (
                    one_f + T.erf(x_val_f32 * inv_sqrt2_f)
                )

                scale_f = T.Cast(accum_dtype, Scale[0])
                bias_f  = T.Cast(accum_dtype, Bias[c])
                add_f   = T.Cast(accum_dtype, Add[c])

                tmp = gelu1 + scale_f * bias_f + add_f

                # ------------------- second GELU ------------------------ #
                gelu2 = half_f * tmp * (
                    one_f + T.erf(tmp * inv_sqrt2_f)
                )

                Out[n, c, h, w] = T.Cast(dtype, gelu2)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused TileLang post-processing                         #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d → GELU → +scale*bias → +add → GELU   (post ops fused)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: float,
        bias_shape: Tuple[int, int, int, int],
    ):
        super().__init__()
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size  = int(kernel_size)

        # -------- ConvTranspose2d weights / bias (identical init) ----------
        w_shape = (self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound  = 1 / math.sqrt(fan_in)
        self.conv_bias = nn.Parameter(torch.empty(self.out_channels))
        torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # ---------------- Extra learnable tensors --------------------------
        self.scale = nn.Parameter(torch.tensor(float(scale_factor)))
        # bias_shape = (1, C, 1, 1)  →  flatten to (C,)
        self.bias = nn.Parameter(torch.randn(self.out_channels))
        self.add  = nn.Parameter(torch.randn(self.out_channels))

        # Kernel cache : {(N,C,H,W,dtype) : kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str = "float16"):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_post_kernel(N, C, H, W, dtype=dtype)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ------------- ConvTranspose2d via cuDNN -------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16).contiguous()

        y = F.conv_transpose2d(x_fp16, w_fp16, b_fp16).contiguous()

        N, C, H, W = y.shape
        kernel = self._get_kernel(N, C, H, W, "float16")

        bias_fp16  = self.bias.to(device="cuda", dtype=torch.float16).contiguous()
        add_fp16   = self.add.to(device="cuda", dtype=torch.float16).contiguous()
        scale_fp16 = self.scale.to(device="cuda", dtype=torch.float16).contiguous().view(1)

        out_fp16 = kernel(y, bias_fp16, add_fp16, scale_fp16)
        return out_fp16.to(orig_dtype)