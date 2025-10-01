"""
Problem Name: 81_Gemm_Swish_Divide_Clamp_Tanh_Clamp
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0555 runtime_stats={'mean': 0.0555, 'std': 0.00108, 'min': 0.0536, 'max': 0.059, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0859, 'std': 0.0198, 'min': 0.0795, 'max': 0.278, 'num_trials': 100}, 'speedup_ratio': 1.55}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        Out: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_features, block_N),
            T.ceildiv(batch_size, block_M),
            threads=128,
        ) as (bx, by):
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)  # W tile in (N,K) form
            bias_s = T.alloc_shared((block_N,), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_loc)

            k_tiles = T.ceildiv(in_features, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], X_s)
                T.copy(W[bx * block_N, ko * block_K], W_s)
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            T.copy(B[bx * block_N], bias_s)

            for i, j in T.Parallel(block_M, block_N):
                row = by * block_M + i
                col = bx * block_N + j
                if (row < batch_size) and (col < out_features):
                    v = C_loc[i, j] + bias_s[j]
                    v = v * T.sigmoid(v)     # Swish
                    v = v / 2.0
                    v = T.min(T.max(v, -1.0), 1.0)
                    v = T.tanh(v)
                    v = T.min(T.max(v, -1.0), 1.0)
                    Out[row, col] = T.Cast(dtype, v)

    return kernel


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self._kernel_cache = {}

    def _fetch_kernel(self, batch_size: int):
        key = batch_size
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_kernel(
                batch_size=batch_size,
                in_features=self.in_features,
                out_features=self.out_features,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16)
        w = self.weight.to(device="cuda", dtype=torch.float16)
        b = (
            self.bias.to(device="cuda", dtype=torch.float16)
            if self.bias is not None
            else torch.zeros(self.out_features, device="cuda", dtype=torch.float16)
        )

        kernel = self._fetch_kernel(x.size(0))
        return kernel(x, w, b)