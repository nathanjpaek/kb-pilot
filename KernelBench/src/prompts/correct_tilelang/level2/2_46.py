"""
Problem Name: 46_Conv2d_Subtract_Tanh_Subtract_AvgPool
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0606 runtime_stats={'mean': 0.0606, 'std': 0.00204, 'min': 0.058, 'max': 0.072, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0691, 'std': 0.00254, 'min': 0.0668, 'max': 0.0837, 'num_trials': 100}, 'speedup_ratio': 1.14}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory
# --------------------------------------------------------------------------- #

def _build_fused_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    sub1: float,
    sub2: float,
    pool_k: int,
    block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    H_out = (H - pool_k) // pool_k + 1
    W_out = (W - pool_k) // pool_k + 1
    TOTAL = N * C * H_out * W_out

    sub1_c = float(sub1)
    sub2_c = float(sub2)
    inv_area = 1.0 / (pool_k * pool_k)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),
        Y: T.Tensor((N, C, H_out, W_out), dtype),
    ):
        sub1_v = T.Cast(accum_dtype, sub1_c)
        sub2_v = T.Cast(accum_dtype, sub2_c)
        inv_v  = T.Cast(accum_dtype, inv_area)

        with T.Kernel(T.ceildiv(TOTAL, block), threads=block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < TOTAL:
                ow  = idx % W_out
                tmp = idx // W_out
                oh  = tmp % H_out
                tmp //= H_out
                c   = tmp % C
                n   = tmp // C

                base_h = oh * pool_k
                base_w = ow * pool_k

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for kh in range(pool_k):
                    for kw in range(pool_k):
                        ih = base_h + kh
                        iw = base_w + kw
                        v  = T.Cast(accum_dtype, X[n, c, ih, iw])
                        v  = v - sub1_v               # first subtraction
                        v  = T.tanh(v)                # tanh activation
                        acc[0] += v                   # accumulate

                avg = acc[0] * inv_v                 # average pooling
                avg = avg - sub2_v                   # second subtraction
                Y[n, c, oh, ow] = T.Cast(dtype, avg)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → (−sub1) → tanh → AvgPool(k) → (−sub2)
    The post-convolution pipeline is fused into a single TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        subtract1_value: float,
        subtract2_value: float,
        kernel_size_pool: int,
    ):
        super().__init__()

        # -------- Conv2d parameters with identical initialisation ----------
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # -------- scalar constants & pool size ----------------------------
        self.sub1 = float(subtract1_value)
        self.sub2 = float(subtract2_value)
        self.pool_k = int(kernel_size_pool)

        # -------- kernel cache -------------------------------------------
        self._kernels: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_fused_kernel(
                N,
                self.weight.shape[0],  # out_channels = C after conv
                H,
                W,
                self.sub1,
                self.sub2,
                self.pool_k,
                dtype=dtype,
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ------------------ move & cast to fp16 ---------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        # ------------------ convolution via cuDNN -------------------------
        y = F.conv2d(x_fp16, w_fp16, b_fp16, stride=1, padding=0)

        # ------------------ fused TileLang kernel -------------------------
        N, C, H, W = y.shape
        kernel = self._get_kernel(N, H, W, "float16")
        out_fp16 = kernel(y.contiguous())

        return out_fp16.to(orig_dtype)