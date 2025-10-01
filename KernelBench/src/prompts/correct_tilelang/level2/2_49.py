"""
Problem Name: 49_ConvTranspose3d_Softmax_Sigmoid
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.28 runtime_stats={'mean': 1.28, 'std': 0.003, 'min': 1.27, 'max': 1.29, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.91, 'std': 0.00512, 'min': 1.9, 'max': 1.92, 'num_trials': 100}, 'speedup_ratio': 1.49}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory :  channel-wise softmax followed by sigmoid
# --------------------------------------------------------------------------- #
def _build_softmax_sigmoid_kernel(
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
    grid = (spatial + block_size - 1) // block_size
    one_f = 1.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:  T.Tensor((N, C, D, H, W), dtype),
        Out: T.Tensor((N, C, D, H, W), dtype),
    ):
        one_c = T.Cast(accum_dtype, one_f)

        with T.Kernel(grid, threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < spatial:
                w  = idx % W
                tmp = idx // W
                h  = tmp % H
                tmp //= H
                d  = tmp % D
                n  = tmp // D

                # ------------------ first pass : sum(exp) -------------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0)
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    sum_exp[0] += T.exp(val)

                inv_sum = one_c / sum_exp[0]

                # --------------- second pass : softmax + sigmoid ------------
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    s   = T.exp(val) * inv_sum               # softmax
                    y   = one_c / (one_c + T.exp(-s))        # sigmoid
                    Out[n, c, d, h, w] = T.Cast(dtype, y)

    return kernel


# --------------------------------------------------------------------------- #
# Optimised PyTorch module
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d → fused (Softmax + Sigmoid) implemented with TileLang.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        bias: bool = True,
    ):
        super().__init__()

        # ----------- ConvTranspose3d parameters (identical init) ------------
        w_shape = (
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(w_shape))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in = in_channels * kernel_size ** 3
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            with torch.no_grad():
                torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # store conv hyper-parameters
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # kernel cache :  {(N,C,D,H,W,dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple, callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_softmax_sigmoid_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ------------------ ConvTranspose3d via cuDNN ----------------------
        weight = self.weight
        bias = self.bias
        y = F.conv_transpose3d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=1,
        )

        # ---------------- fused softmax+sigmoid ---------------------------
        y_fp16 = y.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = y_fp16.shape
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        out_fp16 = kernel(y_fp16)

        return out_fp16.to(orig_dtype)