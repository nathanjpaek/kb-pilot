"""
Problem Name: 63_Gemm_ReLU_Divide
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0409 runtime_stats={'mean': 0.0409, 'std': 0.00307, 'min': 0.0385, 'max': 0.0544, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0582, 'std': 0.0334, 'min': 0.0473, 'max': 0.356, 'num_trials': 100}, 'speedup_ratio': 1.42}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_linear_relu_div_kernel(
    M: int,
    K: int,
    N: int,
    divisor: float,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float",
):
    div_const = float(divisor)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), in_dtype),          # (batch, in_features)
        W: T.Tensor((N, K), in_dtype),          # (out_features, in_features)
        B: T.Tensor((N,), in_dtype),            # bias
        Out: T.Tensor((M, N), in_dtype),        # output created by TileLang
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),              # grid.x
            T.ceildiv(M, block_M),              # grid.y
            threads=threads,                    # threads per block
        ) as (bx, by):
            # Shared-memory tiles
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            # Local accumulator in registers
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)

            # Pipelined reduction over K
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                # Copy current tiles from global to shared
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], B_s)
                # GEMM: A_s (M x K) @ B_s^T (K x N)
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # Bias + ReLU + divide, then store
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, B[gn])
                    val = T.max(val, T.Cast(accum_dtype, 0))   # ReLU
                    val = val / div_const                      # divide
                    Out[gm, gn] = T.Cast(in_dtype, val)

    return kernel


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, divisor: float):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.divisor = float(divisor)

        # Initialize parameters identically to nn.Linear defaults
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache:  {(batch_size, dtype): compiled_kernel}
        self._kernel_cache = {}

    # -------- kernel retrieval -------- #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_relu_div_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
                divisor=self.divisor,
            )
        return self._kernel_cache[key]

    # -------- forward pass -------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        kernel = self._get_kernel(x_fp16.shape[0], x_fp16.dtype)
        y_fp16 = kernel(x_fp16, w_fp16, b_fp16)

        return y_fp16.to(orig_dtype)