"""
Problem Name: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.601 runtime_stats={'mean': 0.601, 'std': 0.00331, 'min': 0.594, 'max': 0.609, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.603, 'std': 0.00361, 'min': 0.597, 'max': 0.621, 'num_trials': 100}, 'speedup_ratio': 1.0}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# --------------------------------------------------------------------------- #
# TileLang kernel factory : GlobalAvgPool3d (N,C,D,H,W) -> (N,C)
# --------------------------------------------------------------------------- #

def _build_gap_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    SPATIAL = D * H * W
    SPATIAL_F = float(SPATIAL)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gap_kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, C), dtype),  # output without the trailing 1x1x1 dims
    ):
        denom = T.Cast(accum_dtype, SPATIAL_F)

        with T.Kernel(N * C, threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            gidx = bx  # global channel index
            n = gidx // C
            c = gidx % C

            part = T.alloc_local((1,), accum_dtype)
            part[0] = T.Cast(accum_dtype, 0)

            for it in T.serial(T.ceildiv(SPATIAL, block_size)):
                idx = it * block_size + tx
                if idx < SPATIAL:
                    w = idx % W
                    t1 = idx // W
                    h = t1 % H
                    d_ = t1 // H
                    part[0] += T.Cast(accum_dtype, X[n, c, d_, h, w])

            total = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        part[0],
                        True,
                        total[0],
                        tx,
                        dtype="handle",
                    )
                )

            if tx == 0:
                Y[n, c] = T.Cast(dtype, total[0] / denom)

    return gap_kernel


# --------------------------------------------------------------------------- #
# Optimised PyTorch module                                                    #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  *scale  →  BatchNorm3d  →  GlobalAvgPool3d
    (last stage implemented with TileLang)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: float,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()

        # ----- ConvTranspose3d (use PyTorch initialisation) ----------------
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
        )

        # ----- BatchNorm3d --------------------------------------------------
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)

        # Scaling factor (pre-BN)
        self.scale_factor = float(scale_factor)

        # Kernel cache : (N,C,D,H,W,dtype) -> compiled kernel
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_gap_kernel(N, C, D, H, W, dtype=dtype)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---------------- ConvTranspose3d + scale + BN --------------------
        x = self.conv_transpose(x)
        x = x * self.scale_factor
        x = self.batch_norm(x)

        # -------------------- fused GlobalAvgPool3d -----------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_fp16.shape
        kernel = self._get_kernel(N, C, D, H, W, "float16")

        y_fp16 = kernel(x_fp16)  # (N,C)
        y_fp16 = y_fp16.view(N, C, 1, 1, 1)  # match AdaptiveAvgPool3d output

        return y_fp16.to(orig_dtype)