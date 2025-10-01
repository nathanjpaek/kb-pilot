"""
Problem Name: 27_Conv2d_HardSwish_AvgPool
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.818 runtime_stats={'mean': 0.818, 'std': 0.0137, 'min': 0.808, 'max': 0.943, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.03, 'std': 0.0198, 'min': 1.02, 'max': 1.22, 'num_trials': 100}, 'speedup_ratio': 1.26}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory: HardSwish → AvgPool2d(k=pool_k, s=pool_k)                   #
# --------------------------------------------------------------------------- #
def _build_hswish_avgpool_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    pool_k: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    H_out = (H - pool_k) // pool_k + 1
    W_out = (W - pool_k) // pool_k + 1
    TOTAL = N * C * H_out * W_out

    three_c = 3.0
    six_c   = 6.0
    inv_six = 1.0 / 6.0
    inv_area = 1.0 / (pool_k * pool_k)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),
        Y: T.Tensor((N, C, H_out, W_out), dtype),
    ):
        three_v = T.Cast(accum_dtype, three_c)
        six_v   = T.Cast(accum_dtype, six_c)
        inv6_v  = T.Cast(accum_dtype, inv_six)
        invA_v  = T.Cast(accum_dtype, inv_area)
        zero_v  = T.Cast(accum_dtype, 0.0)

        with T.Kernel(T.ceildiv(TOTAL, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOTAL:
                ow  = idx % W_out
                t1  = idx // W_out
                oh  = t1 % H_out
                t1 //= H_out
                c   = t1 % C
                n   = t1 // C

                base_h = oh * pool_k
                base_w = ow * pool_k

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = zero_v

                for kh in range(pool_k):
                    for kw in range(pool_k):
                        ih = base_h + kh
                        iw = base_w + kw
                        x_val = T.Cast(accum_dtype, X[n, c, ih, iw])

                        tmp = x_val + three_v
                        tmp = T.max(tmp, zero_v)
                        tmp = T.min(tmp, six_v)
                        hsw = x_val * tmp * inv6_v

                        acc[0] += hsw

                avg = acc[0] * invA_v
                Y[n, c, oh, ow] = T.Cast(dtype, avg)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper module                                                      #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → HardSwish → AvgPool2d(pool_size)
    HardSwish and pooling are fused into one TileLang kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super().__init__()

        # ---------------- Conv2d parameters (identical init) ----------------
        w_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias   = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # ----------- pooling size ------------------------------------------
        self.pool_k = int(pool_size)

        # ----------- kernel cache ------------------------------------------
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_hswish_avgpool_kernel(
                N, C, H, W, self.pool_k, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Move inputs & params to CUDA / fp16 for fast conv
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        # 1. Convolution (cuDNN)
        y = F.conv2d(x_fp16, w_fp16, b_fp16, stride=1, padding=0).contiguous()

        # 2. Fused HardSwish + AvgPool kernel
        N, C, H, W = y.shape
        kernel = self._get_kernel(N, C, H, W, "float16")
        out_fp16 = kernel(y)

        return out_fp16.to(orig_dtype)