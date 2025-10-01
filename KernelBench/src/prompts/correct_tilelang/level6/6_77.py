"""
Problem Name: 77_ConvTranspose2d_Sigmoid_BiasAdd_Sigmoid
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.258 runtime_stats={'mean': 0.258, 'std': 0.0094, 'min': 0.25, 'max': 0.341, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.29, 'std': 0.0106, 'min': 0.283, 'max': 0.387, 'num_trials': 100}, 'speedup_ratio': 1.12}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------- #
#              fused  sigmoid → +bias → sigmoid  kernel factory         #
# --------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total = N * C * H * W
    grid = (total + threads_per_block - 1) // threads_per_block

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:    T.Tensor((N, C, H, W), dtype),   # ConvT output
        Bias: T.Tensor((C,), dtype),           # (C,) bias, broadcast
        Out:  T.Tensor((N, C, H, W), dtype),   # result
    ):
        one = T.Cast(accum_dtype, 1.0)

        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < total:
                w  = idx % W
                tmp = idx // W
                h  = tmp % H
                tmp //= H
                c  = tmp % C
                n  = tmp // C

                val = T.Cast(accum_dtype, X[n, c, h, w])

                # first sigmoid
                s1 = one / (one + T.exp(-val))

                # add bias
                s1 += T.Cast(accum_dtype, Bias[c])

                # second sigmoid
                out = one / (one + T.exp(-s1))

                Out[n, c, h, w] = T.Cast(dtype, out)

    return kernel


# --------------------------------------------------------------------- #
#                        PyTorch wrapper Module                         #
# --------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d → sigmoid → +bias → sigmoid
    The two sigmoids and bias add are fused into a single TileLang kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias_shape: Tuple[int, ...]):
        super().__init__()

        # ConvTranspose2d with default PyTorch initialisation (correct)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)

        # Learnable bias (same creation as original model)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # kernel cache : {(N,C,H,W,dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

        # Move conv layer to GPU/FP16 for faster runtime (weights already initialised)
        self.conv_transpose.to(device="cuda", dtype=torch.float16)

    # ---------------------------------------------------------------- #
    def _get_kernel(self, shape, dtype: str):
        N, C, H, W = shape
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(N, C, H, W, dtype=dtype)
        return self._kern_cache[key]

    # ---------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        # ConvTranspose2d
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        y_fp16 = self.conv_transpose(x_fp16).contiguous()  # (N,C,H,W)

        # Fused sigmoid-bias-sigmoid kernel
        N, C, H, W = y_fp16.shape
        kernel = self._get_kernel((N, C, H, W), "float16")

        bias_fp16 = self.bias.view(-1).to(device="cuda", dtype=torch.float16).contiguous()
        out_fp16 = kernel(y_fp16, bias_fp16)

        return out_fp16.to(original_dtype)