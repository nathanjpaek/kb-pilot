"""
Problem Name: 55_Matmul_MaxPool_Sum_Scale
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0384 runtime_stats={'mean': 0.0384, 'std': 0.00118, 'min': 0.0366, 'max': 0.0437, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0747, 'std': 0.0371, 'min': 0.0683, 'max': 0.442, 'num_trials': 100}, 'speedup_ratio': 1.95}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_fused_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    kernel_size: int,
    scale_factor: float,
    threads_per_block: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float",
):
    """TileLang kernel: Linear → MaxPool(k,stride=k) → Sum → Scale"""

    groups = out_features // kernel_size  # complete windows only

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_features), in_dtype),
        W: T.Tensor((out_features, in_features), in_dtype),
        B: T.Tensor((out_features,), in_dtype),
        Out: T.Tensor((batch_size,), in_dtype),
    ):
        grid = T.ceildiv(batch_size, threads_per_block)

        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            row = bx * threads_per_block + tx

            if row < batch_size:
                # accumulator for final scalar
                total = T.alloc_local((1,), accum_dtype)
                total[0] = T.Cast(accum_dtype, 0)

                # running max within current window
                curr_max = T.alloc_local((1,), accum_dtype)
                curr_max[0] = T.Cast(accum_dtype, -3.4e38)  # -inf
                cnt = T.alloc_local((1,), "int32")
                cnt[0] = 0

                # iterate over every output feature
                for j in range(out_features):
                    # dot product x[row] with W[j]
                    dp = T.alloc_local((1,), accum_dtype)
                    dp[0] = B[j].astype(accum_dtype)
                    for k in range(in_features):
                        dp[0] += (
                            X[row, k].astype(accum_dtype)
                            * W[j, k].astype(accum_dtype)
                        )

                    # update window max
                    curr_max[0] = T.max(curr_max[0], dp[0])
                    cnt[0] += 1

                    # when window filled, accumulate and reset
                    if cnt[0] == kernel_size:
                        total[0] += curr_max[0]
                        curr_max[0] = T.Cast(accum_dtype, -3.4e38)
                        cnt[0] = 0

                # trailing incomplete window is ignored (matches PyTorch)

                # scale and store
                total[0] *= scale_factor
                Out[row] = T.Cast(in_dtype, total[0])

    return fused_kernel


class ModelNew(nn.Module):
    """
    Optimized TileLang implementation of:
        Linear → MaxPool1d(kernel_size, stride=kernel_size) → sum → scale
    Output shape: (batch_size,)
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int, scale_factor: float):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.kernel_size = int(kernel_size)
        self.scale_factor = float(scale_factor)

        # ---- Parameters (identical initialization to nn.Linear) ----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---- Kernel cache keyed by (batch_size, dtype) ----
        self._kernel_cache: Dict[Tuple[int, torch.dtype], tilelang.PrimFunc] = {}

    # ------------------------------------------------------------ #
    # Kernel retrieval / compilation
    # ------------------------------------------------------------ #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            kern = _build_fused_kernel(
                batch_size=batch_size,
                in_features=self.in_features,
                out_features=self.out_features,
                kernel_size=self.kernel_size,
                scale_factor=self.scale_factor,
                in_dtype="float16",
                accum_dtype="float",
            )
            self._kernel_cache[key] = kern
        return self._kernel_cache[key]

    # ------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # prepare tensors
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        batch_size = x_fp16.shape[0]

        kernel = self._get_kernel(batch_size, x_fp16.dtype)
        out_fp16 = kernel(x_fp16, w_fp16, b_fp16)  # (B,)

        return out_fp16.to(orig_dtype)