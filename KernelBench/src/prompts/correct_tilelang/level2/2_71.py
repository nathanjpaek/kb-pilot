"""
Problem Name: 71_Conv2d_Divide_LeakyReLU
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0522 runtime_stats={'mean': 0.0522, 'std': 0.00102, 'min': 0.0507, 'max': 0.0567, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0532, 'std': 0.0018, 'min': 0.0515, 'max': 0.0642, 'num_trials': 100}, 'speedup_ratio': 1.02}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory: Conv2d → divide (by const) → LeakyReLU
# --------------------------------------------------------------------------- #
def _build_conv_div_leaky_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    divisor_const: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    OH = Hin - K + 1
    OW = Win - K + 1
    TOTAL = N * Cout * OH * OW

    inv_div = 1.0 / float(divisor_const)
    neg_slope = 0.01

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, Cin, Hin, Win), dtype),
        Wt: T.Tensor((Cout, Cin, K, K), dtype),   # weight tensor
        B: T.Tensor((Cout,), dtype),              # bias
        Y: T.Tensor((N, Cout, OH, OW), dtype),    # output (created by TileLang)
    ):
        inv_div_f   = T.Cast(accum_dtype, inv_div)
        neg_slope_f = T.Cast(accum_dtype, neg_slope)

        with T.Kernel(T.ceildiv(TOTAL, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOTAL:
                ow = idx % OW
                tmp = idx // OW
                oh = tmp % OH
                tmp //= OH
                oc = tmp % Cout
                n  = tmp // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, B[oc])

                for ic in range(Cin):
                    for kh in range(K):
                        ih = oh + kh
                        for kw in range(K):
                            iw = ow + kw
                            acc[0] += (
                                T.Cast(accum_dtype, X[n, ic, ih, iw])
                                * T.Cast(accum_dtype, Wt[oc, ic, kh, kw])
                            )

                # divide by constant
                acc[0] = acc[0] * inv_div_f

                # LeakyReLU
                acc_neg = acc[0] * neg_slope_f
                acc[0]  = T.max(acc[0], acc_neg)

                Y[n, oc, oh, ow] = T.Cast(dtype, acc[0])

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused TileLang kernel
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → divide(by constant) → LeakyReLU (all fused in TileLang)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, divisor: float):
        super().__init__()
        self.in_c = int(in_channels)
        self.out_c = int(out_channels)
        self.k = int(kernel_size)
        self.divisor = float(divisor)

        # ---- Conv2d parameters with identical initialisation ---------------
        self.weight = nn.Parameter(
            torch.empty(self.out_c, self.in_c, self.k, self.k)
        )
        self.bias = nn.Parameter(torch.empty(self.out_c))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = self.in_c * self.k * self.k
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---- Kernel cache ---------------------------------------------------
        self._kern_cache: Dict[Tuple[int, int, int, torch.dtype], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: torch.dtype):
        key = (N, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_conv_div_leaky_kernel(
                N,
                self.in_c,
                H,
                W,
                self.out_c,
                self.k,
                self.divisor,
                dtype=str(dtype).split(".")[-1],  # "float16"
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        w_f16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        N, Cin, H, W = x_f16.shape
        assert Cin == self.in_c, "in_channels mismatch"

        kernel = self._get_kernel(N, H, W, x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, b_f16)

        return y_f16.to(orig_dtype)