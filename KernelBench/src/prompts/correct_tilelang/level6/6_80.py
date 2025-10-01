"""
Problem Name: 80_ConvTranspose3d_Softmax_Clamp_Swish_BiasAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=14.0 runtime_stats={'mean': 14.0, 'std': 0.0117, 'min': 13.9, 'max': 14.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 15.2, 'std': 0.0397, 'min': 15.1, 'max': 15.4, 'num_trials': 100}, 'speedup_ratio': 1.09}}
"""

import math
from typing import Dict, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                       TileLang kernel factory                               #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    clamp_min: float,
    clamp_max: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
) -> Callable:
    """softmax(dim=2) → clamp → swish → +bias"""
    NDHW = N * C * H * W              # one thread covers depth-slice
    grid = (NDHW + threads_per_block - 1) // threads_per_block
    cmin_f = float(clamp_min)
    cmax_f = float(clamp_max)

    @tilelang.jit(out_idx=-1)         # output tensor created automatically
    @T.prim_func
    def kernel(
        X:    T.Tensor((N, C, D, H, W), dtype),   # conv-transpose result
        Bias: T.Tensor((C,), dtype),              # (C,) broadcast later
        Out:  T.Tensor((N, C, D, H, W), dtype),   # final tensor
    ):
        cmin = T.Cast(accum_dtype, cmin_f)
        cmax = T.Cast(accum_dtype, cmax_f)
        one  = T.Cast(accum_dtype, 1.0)

        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            linear_idx = bx * threads_per_block + tx
            if linear_idx < NDHW:
                w  = linear_idx % W
                tmp = linear_idx // W
                h  = tmp % H
                tmp //= H
                c_ = tmp % C
                n  = tmp // C

                # ---------------- pass-1 : Σexp over depth ----------------
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0)
                for d in T.serial(D):
                    v = T.Cast(accum_dtype, X[n, c_, d, h, w])
                    sum_exp[0] += T.exp(v)
                inv_sum = one / sum_exp[0]

                # ---------------- pass-2 : produce output -----------------
                bias_val = T.Cast(accum_dtype, Bias[c_])
                for d in T.serial(D):
                    v = T.Cast(accum_dtype, X[n, c_, d, h, w])
                    soft = T.exp(v) * inv_sum            # softmax along D
                    v2  = T.max(soft, cmin)              # clamp
                    v2  = T.min(v2, cmax)
                    sig = one / (one + T.exp(-v2))       # sigmoid
                    sw  = v2 * sig                       # swish
                    out = sw + bias_val
                    Out[n, c_, d, h, w] = T.Cast(dtype, out)

    return kernel


# --------------------------------------------------------------------------- #
#                               PyTorch wrapper                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d ➜ softmax(depth) ➜ clamp ➜ swish ➜ +bias   (fused)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias_shape: Tuple[int, ...],
        clamp_min: float,
        clamp_max: float,
    ):
        super().__init__()

        # ---------------- conv-transpose parameters --------------------- #
        wt_shape = (
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(wt_shape))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size ** 3
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        self.stride = stride
        self.padding = padding
        self.output_padding = 0
        self.dilation = 1

        # post-softmax bias (broadcast) : store flattened (C,)
        self.post_bias = nn.Parameter(torch.randn(bias_shape).view(-1))

        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # kernel cache : (N,C,D,H,W,dtype) ➜ compiled kernel
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], Callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
                N,
                C,
                D,
                H,
                W,
                self.clamp_min,
                self.clamp_max,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b1_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16)

        # ConvTranspose3d via cuDNN
        y_fp16 = F.conv_transpose3d(
            x_fp16,
            w_fp16,
            b1_fp16,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
        ).contiguous()   # (N,C,D,H,W)

        N, C, D, H, W = y_fp16.shape

        bias_fp16 = self.post_bias.to(device="cuda", dtype=torch.float16).contiguous()

        ker = self._get_kernel(N, C, D, H, W, "float16")
        out_fp16 = ker(y_fp16, bias_fp16)

        return out_fp16.to(orig_dtype)