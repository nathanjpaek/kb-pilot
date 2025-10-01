"""
Problem Name: 97_Conv2d_HardSwish_GlobalAvgPool_Sum_HardSwish
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.58 runtime_stats={'mean': 1.58, 'std': 0.00787, 'min': 1.57, 'max': 1.65, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.871, 'std': 0.00781, 'min': 0.864, 'max': 0.941, 'num_trials': 100}, 'speedup_ratio': 0.551}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#             Kernel-factory : HardSwish → GAP → HardSwish (N,C)             #
# --------------------------------------------------------------------------- #

def _build_gap_hswish_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    NC = N * C
    area = float(H * W)
    inv6 = 1.0 / 6.0
    inv_area = 1.0 / area

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, H, W), dtype),   # conv output
        Out: T.Tensor((N, C), dtype),         # (N,C)
    ):
        three_v = T.Cast(accum_dtype, 3.0)
        six_v   = T.Cast(accum_dtype, 6.0)
        inv6_v  = T.Cast(accum_dtype, inv6)
        invA_v  = T.Cast(accum_dtype, inv_area)
        zero_v  = T.Cast(accum_dtype, 0.0)

        with T.Kernel(T.ceildiv(NC, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < NC:
                n = idx // C
                c = idx % C

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = zero_v

                for h in T.serial(H):
                    for w in T.serial(W):
                        v = T.Cast(accum_dtype, X[n, c, h, w])
                        t = v + three_v
                        t = T.min(six_v, T.max(zero_v, t))
                        hsw = v * t * inv6_v
                        acc[0] += hsw

                mean_v = acc[0] * invA_v

                # second HardSwish
                t2 = mean_v + three_v
                t2 = T.min(six_v, T.max(zero_v, t2))
                out_val = mean_v * t2 * inv6_v

                Out[n, c] = T.Cast(dtype, out_val)

    return kernel


# --------------------------------------------------------------------------- #
#                               PyTorch wrapper                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → HardSwish → GlobalAvgPool → Sum → HardSwish
    Fused in TileLang (everything after Conv2d).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        # ---------------- Conv2d parameters (PyTorch-identical init) ---------
        w_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias   = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound  = 1.0 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # --------------- kernel cache  {(N,C,H,W,dtype): kernel} -------------
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_gap_hswish_kernel(N, C, H, W, dtype=dtype)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Conv2d on CUDA/FP16 for speed
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        y = F.conv2d(x_fp16, w_fp16, b_fp16, stride=1, padding=0).contiguous()
        N, C, H, W = y.shape

        # Fused HardSwish + GAP + HardSwish kernel
        kernel = self._get_kernel(N, C, H, W, "float16")
        out_fp16 = kernel(y)

        return out_fp16.to(orig_dtype)