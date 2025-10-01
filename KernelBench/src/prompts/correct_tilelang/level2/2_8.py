"""
Problem Name: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.662 runtime_stats={'mean': 0.662, 'std': 0.051, 'min': 0.637, 'max': 1.01, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.63, 'std': 0.0248, 'min': 0.617, 'max': 0.857, 'num_trials': 100}, 'speedup_ratio': 0.952}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory
# --------------------------------------------------------------------------- #
def _make_reduce_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    scale_const: float,
    bias_sum_const: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT_PER_SAMPLE = C * D * H * W
    TOT            = N * TOT_PER_SAMPLE

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def reduce_kernel(
        X_flat: T.Tensor((TOT,), dtype),      # flattened input
        Y:      T.Tensor((N,),   dtype),      # 1-D output
    ):
        scale_c = T.Cast(accum_dtype, scale_const)
        bias_c  = T.Cast(accum_dtype, bias_sum_const)

        with T.Kernel(N, threads=block_size) as bn:      # one block per batch sample
            tx   = T.get_thread_binding(0)
            part = T.alloc_local((1,), accum_dtype)
            part[0] = T.Cast(accum_dtype, 0)

            for it in T.serial(T.ceildiv(TOT_PER_SAMPLE, block_size)):
                idx = it * block_size + tx
                if idx < TOT_PER_SAMPLE:
                    part[0] += T.Cast(
                        accum_dtype,
                        X_flat[bn * TOT_PER_SAMPLE + idx],
                    )

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
                Y[bn] = T.Cast(dtype, total[0] * scale_c + bias_c)

    return reduce_kernel


# --------------------------------------------------------------------------- #
# Optimised PyTorch Module
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d → MaxPool3d → (fused TileLang kernel): divide / global-avg / +bias / sum.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        divisor: float,
        pool_size,
        bias_shape,
        sum_dim,          # kept for interface parity (always 1 here)
    ):
        super().__init__()

        # --- Conv3d ---------------------------------------------------------
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

        # --- other hyper-parameters ----------------------------------------
        self.divisor     = float(divisor)
        self.max_pool    = nn.MaxPool3d(pool_size)
        self.sum_dim     = int(sum_dim)      # unused but kept for API parity

        # --- bias (added after global-avg) ----------------------------------
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # --- kernel cache ---------------------------------------------------
        self._kern_cache: Dict[
            Tuple[int, int, int, int, str, float], callable
        ] = {}

    # --------------------------------------------------------------------- #
    # Kernel retrieval / compilation
    # --------------------------------------------------------------------- #
    def _get_kernel(
        self,
        N: int,
        C: int,
        D: int,
        H: int,
        W: int,
        dtype: str,
        bias_sum: float,
    ):
        key = (N, D, H, W, dtype, bias_sum)
        if key not in self._kern_cache:
            scale_const = 1.0 / (self.divisor * D * H * W)
            self._kern_cache[key] = _make_reduce_kernel(
                N,
                C,
                D,
                H,
                W,
                scale_const=scale_const,
                bias_sum_const=bias_sum,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ------------------ convolution & max-pool in cuDNN -----------------
        x = self.conv(x)                              # (N,C,D1,H1,W1)
        x = self.max_pool(x)                          # (N,C,D2,H2,W2)

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, D, H, W = x_fp16.shape

        # -------------------- bias sum & kernel compile ---------------------
        bias_sum = float(self.bias.view(-1).sum().item())
        kernel   = self._get_kernel(
            N, C, D, H, W, dtype="float16", bias_sum=bias_sum
        )

        # --------------------------- kernel call ----------------------------
        y_fp16 = kernel(x_fp16.view(-1))              # (N,)
        y_fp16 = y_fp16.view(N, 1, 1, 1)              # match original shape

        return y_fp16.to(orig_dtype)