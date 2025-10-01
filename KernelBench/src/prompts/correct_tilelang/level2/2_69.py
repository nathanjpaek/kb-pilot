"""
Problem Name: 69_Conv2d_HardSwish_ReLU
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0625 runtime_stats={'mean': 0.0625, 'std': 0.00201, 'min': 0.0598, 'max': 0.0732, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.056, 'std': 0.00245, 'min': 0.0538, 'max': 0.0728, 'num_trials': 100}, 'speedup_ratio': 0.896}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

import tilelang
import tilelang.language as T

# --------------------------------------------------------------------------- #
# Kernel factory: HardSwish followed by ReLU (y = max(0, hswish(x)))         #
# --------------------------------------------------------------------------- #

def _build_hswish_relu(
    N: int,
    C: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total = N * C * H * W

    three_c = 3.0
    six_c   = 6.0
    inv_six = 1.0 / 6.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),
        Y: T.Tensor((N, C, H, W), dtype),
    ):
        three_f  = T.Cast(accum_dtype, three_c)
        six_f    = T.Cast(accum_dtype, six_c)
        inv_sixf = T.Cast(accum_dtype, inv_six)
        zero_f   = T.Cast(accum_dtype, 0.0)

        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total:
                w  = idx % W
                t1 = idx // W
                h  = t1 % H
                t2 = t1 // H
                c  = t2 % C
                n  = t2 // C

                x_val = T.Cast(accum_dtype, X[n, c, h, w])

                tmp    = x_val + three_f
                tmp    = T.max(tmp,   zero_f)
                tmp    = T.min(tmp,   six_f)
                hswish = x_val * (tmp * inv_sixf)
                out_v  = T.max(hswish, zero_f)  # ReLU

                Y[n, c, h, w] = T.Cast(dtype, out_v)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper module                                                      #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """Conv2d → HardSwish → ReLU (fused in TileLang)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        # ---------------- Conv2d parameters (identical init) ----------------
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        self.bias   = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.kernel_size = int(kernel_size)

        # kernel cache  {(N,C,H,W,dtype): compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_hswish_relu(N, C, H, W, dtype=dtype)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # -------------------- convolution (cuDNN) -------------------------
        x = F.conv2d(x, self.weight, self.bias, stride=1, padding=0)

        # -------------------- fused activation kernel --------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, H, W = x_fp16.shape

        kernel = self._get_kernel(N, C, H, W, "float16")
        y_fp16 = kernel(x_fp16)

        return y_fp16.to(orig_dtype)