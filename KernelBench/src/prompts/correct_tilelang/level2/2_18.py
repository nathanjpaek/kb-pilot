"""
Problem Name: 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0583 runtime_stats={'mean': 0.0583, 'std': 0.00165, 'min': 0.0564, 'max': 0.0671, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.143, 'std': 0.00458, 'min': 0.138, 'max': 0.169, 'num_trials': 100}, 'speedup_ratio': 2.45}}
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
    block_threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Kernel that outputs: bias_sum + X @ weight_sum
    Result shape: (batch_size, 1)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        Wsum: T.Tensor((in_features,), dtype),      # pre-computed Σ_j W[j, k]
        Bsum: T.Tensor((1,), dtype),                # pre-computed Σ_j bias_j
        Out: T.Tensor((batch_size, 1), dtype),
    ):
        grid = T.ceildiv(batch_size, block_threads)
        with T.Kernel(grid, threads=block_threads) as bx:
            tx = T.get_thread_binding(0)
            row = bx * block_threads + tx

            if row < batch_size:
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = Bsum[0].astype(accum_dtype)

                for k in range(in_features):
                    acc[0] += (
                        X[row, k].astype(accum_dtype)
                        * Wsum[k].astype(accum_dtype)
                    )

                Out[row, 0] = T.Cast(dtype, acc[0])

    return fused_kernel


class ModelNew(nn.Module):
    """
    Optimized model that fuses Linear → Sum → Max → Mean → LogSumExp → LogSumExp
    into a single TileLang kernel computing bias_sum + X @ weight_sum.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # ---- Parameter initialization identical to nn.Linear ----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.empty(out_features))
        with torch.no_grad():
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
                dtype="float16",
                accum_dtype="float",
            )
            self._kernel_cache[key] = kern
        return self._kernel_cache[key]

    # ------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch_size, in_features)  – any dtype / device
        Returns:
            (batch_size, 1) tensor – same dtype as input
        """
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        # Prepare summed parameters
        weight_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        bias_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        weight_sum_fp16 = weight_fp16.sum(dim=0, keepdim=False).contiguous()
        bias_sum_fp16 = bias_fp16.sum().unsqueeze(0).contiguous()

        batch_size = x_fp16.shape[0]
        kernel = self._get_kernel(batch_size, x_fp16.dtype)

        out_fp16 = kernel(x_fp16, weight_sum_fp16, bias_sum_fp16)

        return out_fp16.to(orig_dtype)