"""
Problem Name: 43_Conv3d_Max_LogSumExp_ReLU
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.707 runtime_stats={'mean': 0.707, 'std': 0.00746, 'min': 0.699, 'max': 0.767, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.758, 'std': 0.0153, 'min': 0.747, 'max': 0.903, 'num_trials': 100}, 'speedup_ratio': 1.07}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : channel-wise logsumexp + ReLU                    #
# --------------------------------------------------------------------------- #
def _build_lse_relu_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    spatial = N * D * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def lse_relu(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, 1, D, H, W), dtype),
    ):
        minus_inf = T.Cast(accum_dtype, -1.0e30)
        zero_f    = T.Cast(accum_dtype, 0.0)

        with T.Kernel(T.ceildiv(spatial, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < spatial:
                w = idx % W
                tmp1 = idx // W
                h = tmp1 % H
                tmp2 = tmp1 // H
                d = tmp2 % D
                n = tmp2 // D

                # pass-1 : max
                max_val = T.alloc_local((1,), accum_dtype)
                max_val[0] = minus_inf
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    if val > max_val[0]:
                        max_val[0] = val

                # pass-2 : sum(exp(x - max))
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = zero_f
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    sum_exp[0] += T.exp(val - max_val[0])

                lse = max_val[0] + T.log(sum_exp[0])
                lse = T.max(lse, zero_f)          # ReLU

                Y[n, 0, d, h, w] = T.Cast(dtype, lse)

    return lse_relu


# --------------------------------------------------------------------------- #
# Optimised PyTorch module                                                   #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d → MaxPool3d → (fused TileLang) logsumexp(dim=1,keepdim=True) + ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()

        # Retain Conv3d and MaxPool3d exactly as original
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Kernel cache  :  (N,D,H,W,dtype) -> compiled kernel
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # --------------------------------------------------------------------- #
    # kernel retrieval / compilation
    # --------------------------------------------------------------------- #
    def _get_kernel(
        self,
        N: int,
        C: int,
        D: int,
        H: int,
        W: int,
        dtype: str,
    ):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_lse_relu_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # --------------------------------------------------------------------- #
    # forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Convolution and max-pooling via cuDNN
        x = self.conv(x)
        x = self.max_pool(x)

        # Prepare for TileLang kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_fp16.shape

        kernel = self._get_kernel(N, C, D, H, W, "float16")

        y_fp16 = kernel(x_fp16)            # (N,1,D,H,W)

        return y_fp16.to(orig_dtype)