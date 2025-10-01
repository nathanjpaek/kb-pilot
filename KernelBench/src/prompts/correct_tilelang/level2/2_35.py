"""
Problem Name: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0698 runtime_stats={'mean': 0.0698, 'std': 0.0219, 'min': 0.0566, 'max': 0.224, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0908, 'std': 0.0423, 'min': 0.0716, 'max': 0.461, 'num_trials': 100}, 'speedup_ratio': 1.3}}
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
    subtract_val: float,
    pool_k: int = 2,
    block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Ho = (H - pool_k) // pool_k + 1
    Wo = (W - pool_k) // pool_k + 1
    TOTAL = N * C * Ho * Wo

    sub_c   = float(subtract_val)
    zero_c  = 0.0
    three_c = 3.0
    six_c   = 6.0
    one_c   = 1.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:  T.Tensor((N, C, H, W), dtype),
        Out: T.Tensor((N, C, Ho, Wo), dtype),
    ):
        sub_v   = T.Cast(accum_dtype, sub_c)
        zero_v  = T.Cast(accum_dtype, zero_c)
        three_v = T.Cast(accum_dtype, three_c)
        six_v   = T.Cast(accum_dtype, six_c)
        inv_six = T.Cast(accum_dtype, 1.0 / six_c)
        one_v   = T.Cast(accum_dtype, one_c)

        with T.Kernel(T.ceildiv(TOTAL, block), threads=block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < TOTAL:
                ow  = idx % Wo
                tmp = idx // Wo
                oh  = tmp % Ho
                tmp //= Ho
                c   = tmp % C
                n   = tmp // C

                max_val = T.alloc_local((1,), accum_dtype)
                max_val[0] = -3.4e38   # -FLT_MAX

                base_h = oh * pool_k
                base_w = ow * pool_k

                for kh in T.serial(pool_k):
                    for kw in T.serial(pool_k):
                        ih = base_h + kh
                        iw = base_w + kw
                        inp = T.Cast(accum_dtype, X[n, c, ih, iw])

                        # subtract & HardSwish
                        v        = inp - sub_v
                        relu6_in = v + three_v
                        relu6_in = T.max(relu6_in, zero_v)
                        relu6_in = T.min(relu6_in, six_v)
                        hswish   = v * relu6_in * inv_six

                        max_val[0] = T.max(max_val[0], hswish)

                # Mish activation
                sp   = T.log(one_v + T.exp(max_val[0]))
                mish = max_val[0] * T.tanh(sp)

                Out[n, c, oh, ow] = T.Cast(dtype, mish)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → (−subtract) → HardSwish → MaxPool2d(kernel=2,stride=2) → Mish
    The post-conv pipeline is fused into a single TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        subtract_value: float,
        pool_kernel_size: int,
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

        # -------- scalar & pool --------------------------------------------
        self.sub_val = float(subtract_value)
        self.pool_k  = int(pool_kernel_size)

        # -------- kernel cache ---------------------------------------------
        self._cache: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._cache:
            self._cache[key] = _build_fused_kernel(
                N,
                self.weight.shape[0],   # C_out
                H,
                W,
                subtract_val=self.sub_val,
                pool_k=self.pool_k,
                dtype=dtype,
            )
        return self._cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # --------------------- convolution in cuDNN ------------------------
        w = self.weight
        b = self.bias
        x = F.conv2d(x, w, b, stride=1, padding=0)

        # --------------------- fused TileLang kernel -----------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, H, W = x_fp16.shape

        kernel = self._get_kernel(N, H, W, "float16")
        y_fp16 = kernel(x_fp16)

        return y_fp16.to(orig_dtype)