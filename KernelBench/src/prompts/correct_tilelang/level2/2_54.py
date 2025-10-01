"""
Problem Name: 54_Conv2d_Multiply_LeakyReLU_GELU
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0599 runtime_stats={'mean': 0.0599, 'std': 0.00454, 'min': 0.0562, 'max': 0.0795, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0703, 'std': 0.0169, 'min': 0.0593, 'max': 0.144, 'num_trials': 100}, 'speedup_ratio': 1.17}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """
    Conv2d → per-channel multiply → LeakyReLU → GELU (fused TileLang)
    """

    # ------------------------------------------------------------------ #
    #                             KERNEL FACTORY                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_fused_kernel(
        N: int,
        Cin: int,
        Hin: int,
        Win: int,
        Cout: int,
        K: int,
        block_size: int = 256,
        dtype: str = "float16",
        accum_dtype: str = "float32",
    ):
        OH = Hin - K + 1
        OW = Win - K + 1
        TOTAL = N * Cout * OH * OW
        neg_slope = 0.01
        inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def fused_conv(
            X: T.Tensor((N, Cin, Hin, Win), dtype),
            Wt: T.Tensor((Cout, Cin, K, K), dtype),
            B: T.Tensor((Cout,), dtype),
            Scale: T.Tensor((Cout,), dtype),
            Y: T.Tensor((N, Cout, OH, OW), dtype),
        ):
            half_f     = T.Cast(accum_dtype, 0.5)
            inv_s2_f   = T.Cast(accum_dtype, inv_sqrt2)
            neg_slope_f = T.Cast(accum_dtype, neg_slope)

            with T.Kernel(T.ceildiv(TOTAL, block_size), threads=block_size) as bx:
                tx  = T.get_thread_binding(0)
                idx = bx * block_size + tx
                if idx < TOTAL:
                    ow  = idx % OW
                    tmp = idx // OW
                    oh  = tmp % OH
                    tmp = tmp // OH
                    oc  = tmp % Cout
                    n   = tmp // Cout

                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = T.Cast(accum_dtype, B[oc])

                    for ic in range(Cin):
                        for kh in range(K):
                            ih = oh + kh
                            for kw in range(K):
                                iw = ow + kw
                                x_val = T.Cast(
                                    accum_dtype, X[n, ic, ih, iw]
                                )
                                w_val = T.Cast(
                                    accum_dtype, Wt[oc, ic, kh, kw]
                                )
                                acc[0] += x_val * w_val

                    # multiply by learnable per-channel scalar
                    acc[0] *= T.Cast(accum_dtype, Scale[oc])

                    # LeakyReLU
                    acc_pos = acc[0]
                    acc_neg = acc[0] * neg_slope_f
                    acc[0] = T.max(acc_pos, acc_neg)

                    # GELU
                    acc_gelu = (
                        half_f
                        * acc[0]
                        * (T.Cast(accum_dtype, 1.0) + T.erf(acc[0] * inv_s2_f))
                    )

                    Y[n, oc, oh, ow] = T.Cast(dtype, acc_gelu)

        return fused_conv

    # ------------------------------------------------------------------ #
    #                             INITIALISATION                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        multiplier_shape,
    ):
        super().__init__()
        self.in_c = int(in_channels)
        self.out_c = int(out_channels)
        self.k = int(kernel_size)

        # -------- Conv parameters (same as nn.Conv2d defaults) -------- #
        weight_shape = (
            self.out_c,
            self.in_c,
            self.k,
            self.k,
        )
        self.weight = nn.Parameter(torch.empty(weight_shape))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_c * self.k * self.k
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(self.out_c))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # -------- Learnable multiplier -------- #
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        # -------- Kernel cache -------- #
        self._kern_cache: Dict[
            Tuple[int, int, int, torch.dtype], tilelang.PrimFunc
        ] = {}

    # ------------------------------------------------------------------ #
    #                        KERNEL RETRIEVAL                            #
    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: torch.dtype):
        key = (N, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = self._build_fused_kernel(
                N,
                self.in_c,
                H,
                W,
                self.out_c,
                self.k,
                dtype=str(dtype).split(".")[-1],  # "float16"
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    #                               FORWARD                              #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        w_f16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()
        s_f16 = (
            self.multiplier.to(device="cuda", dtype=torch.float16)
            .view(self.out_c)
            .contiguous()
        )

        N, Cin, H, W = x_f16.shape
        assert Cin == self.in_c, "in_channels mismatch"

        kernel = self._get_kernel(N, H, W, x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, b_f16, s_f16)

        return y_f16.to(orig_dtype)