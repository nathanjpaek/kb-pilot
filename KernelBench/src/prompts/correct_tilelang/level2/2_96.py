"""
Problem Name: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=27.2 runtime_stats={'mean': 27.2, 'std': 0.0415, 'min': 27.2, 'max': 27.5, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 22.0, 'std': 0.0144, 'min': 22.0, 'max': 22.1, 'num_trials': 100}, 'speedup_ratio': 0.809}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory :  scale × MaxPool3d(k=2,s=2) → global-avg → clamp[0,1]      #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    D_in: int,
    H_in: int,
    W_in: int,
    scale: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    K = 2  # max-pool kernel & stride
    D1 = (D_in - K) // K + 1
    H1 = (H_in - K) // K + 1
    W1 = (W_in - K) // K + 1
    pool_cnt = D1 * H1 * W1

    TOT = N * C
    GRID = (TOT + threads_per_block - 1) // threads_per_block
    neg_inf = -3.4028234663852886e38  # −inf in fp32

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused(
        X: T.Tensor((N, C, D_in, H_in, W_in), dtype),
        Y: T.Tensor((N, C, 1, 1, 1), dtype),
    ):
        scale_c   = T.Cast(accum_dtype, scale)
        inv_cnt_c = T.Cast(accum_dtype, 1.0 / pool_cnt)
        zero_f    = T.Cast(accum_dtype, 0.0)
        one_f     = T.Cast(accum_dtype, 1.0)
        ninf_f    = T.Cast(accum_dtype, neg_inf)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                c = idx % C
                n = idx // C

                sum_max = T.alloc_local((1,), accum_dtype)
                sum_max[0] = zero_f

                for d1 in T.serial(D1):
                    base_d = d1 * K
                    for h1 in T.serial(H1):
                        base_h = h1 * K
                        for w1 in T.serial(W1):
                            base_w = w1 * K

                            mval = T.alloc_local((1,), accum_dtype)
                            mval[0] = ninf_f
                            for kd in T.serial(K):
                                for kh in T.serial(K):
                                    for kw in T.serial(K):
                                        val = (
                                            T.Cast(
                                                accum_dtype,
                                                X[n, c, base_d + kd, base_h + kh, base_w + kw],
                                            )
                                            * scale_c
                                        )
                                        mval[0] = T.max(mval[0], val)
                            sum_max[0] += mval[0]

                avg = sum_max[0] * inv_cnt_c
                avg = T.max(avg, zero_f)
                avg = T.min(avg, one_f)
                Y[n, c, 0, 0, 0] = T.Cast(dtype, avg)

    return fused


# --------------------------------------------------------------------------- #
# Optimised PyTorch wrapper                                                   #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch) →
    fused TileLang kernel implementing (×scale → MaxPool3d(k=2) → GAP → clamp)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        scale: float,
        maxpool_kernel_size: int,  # present for signature parity; must be 2
    ):
        super().__init__()
        assert maxpool_kernel_size == 2, "Kernel factory assumes k=2 max-pool"

        # Keep ConvTranspose3d exactly as in the reference implementation
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        self.scale = float(scale)

        # kernel cache : {(N,D,H,W,dtype) : kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            C = self.conv_transpose.out_channels
            self._kern_cache[key] = _build_fused_kernel(
                N, C, D, H, W, self.scale, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ------ ConvTranspose3d (cuDNN) ---------------------------------
        x = self.conv_transpose(x)

        # ------ Fused TileLang kernel -----------------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_fp16.shape
        kernel = self._get_kernel(N, D, H, W, "float16")
        y_fp16 = kernel(x_fp16)

        return y_fp16.to(orig_dtype)