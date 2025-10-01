"""
Problem Name: 20_Conv3d_Subtract_Mish
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.75 runtime_stats={'mean': 3.75, 'std': 0.04, 'min': 3.73, 'max': 4.12, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.26, 'std': 0.0157, 'min': 4.24, 'max': 4.31, 'num_trials': 100}, 'speedup_ratio': 1.14}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory : subtract global mean then Mish activation                  #
# --------------------------------------------------------------------------- #

def _build_sub_mean_mish_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total = N * C * D * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, D, H, W), dtype),   # conv output
        mean: T.Tensor((1,),        accum_dtype), # scalar mean
        Out:  T.Tensor((N, C, D, H, W), dtype),   # result
    ):
        one = T.Cast(accum_dtype, 1.0)
        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total:
                w   = idx % W
                t1  = idx // W
                h   = t1 % H
                t2  = t1 // H
                d_  = t2 % D
                t3  = t2 // D
                c   = t3 % C
                n   = t3 // C

                val = T.Cast(accum_dtype, X[n, c, d_, h, w]) - mean[0]
                sp  = T.log(one + T.exp(val))     # softplus
                val = val * T.tanh(sp)            # Mish
                Out[n, c, d_, h, w] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
#                          PyTorch wrapper                                    #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """Conv3d  →  subtract global mean  →  Mish   (TileLang-fused)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        # Conv3d with identical PyTorch defaults (weights/bias correctly init)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # kernel cache  {(N,C,D,H,W,dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, shape, dtype: str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_sub_mean_mish_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # 1. Conv3d in high precision (fp32)
        x = self.conv(x)
        # 2. Move to CUDA / fp16 for fused kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_fp16.shape

        # 3. Compute global mean (fp32 for accuracy)
        mean_scalar = x_fp16.mean(dtype=torch.float32)
        mean_tensor = torch.tensor([mean_scalar.item()], dtype=torch.float32, device="cuda")

        # 4. Run fused TileLang kernel
        kernel = self._get_kernel((N, C, D, H, W), "float16")
        y_fp16 = kernel(x_fp16, mean_tensor)

        return y_fp16.to(orig_dtype)