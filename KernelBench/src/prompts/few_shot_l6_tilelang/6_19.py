"""
Problem Name: 19_Gemm_Sigmoid_Scaling_ResidualAdd
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0586 runtime_stats={'mean': 0.0586, 'std': 0.0302, 'min': 0.0424, 'max': 0.169, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0838, 'std': 0.0589, 'min': 0.0506, 'max': 0.358, 'num_trials': 100}, 'speedup_ratio': 1.43}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_gemm_sigmoid_residual_kernel(
    M: int,
    N: int,
    K: int,
    scaling_factor: float,
    *,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    scale_const = float(scaling_factor)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),       # [N, K]   row-major
        B: T.Tensor((N,), dtype),         # bias
        Out: T.Tensor((M, N), dtype),     # output (allocated by TileLang)
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            # Shared-memory tiles
            X_s   = T.alloc_shared((block_M, block_K), dtype)
            W_s   = T.alloc_shared((block_N, block_K), dtype)
            Bias_s = T.alloc_shared((block_N,), dtype)

            # Fragment accumulator
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            # Load bias tile once per block
            T.copy(B[bx * block_N], Bias_s)

            # Reduction over K dimension
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], X_s)
                T.copy(W[bx * block_N, ko * block_K], W_s)
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            # Epilogue: bias add, sigmoid, scale, residual add, store
            one_acc = T.Cast(accum_dtype, 1.0)
            scale_acc = T.Cast(accum_dtype, scale_const)

            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    v = C_loc[i, j] + T.Cast(accum_dtype, Bias_s[j])
                    sig = one_acc / (one_acc + T.exp(-v))
                    out_val = v + scale_acc * sig
                    Out[gm, gn] = T.Cast(dtype, out_val)

    return kernel


class ModelNew(nn.Module):
    """
    Optimized model implementing Gemm → Sigmoid → Scaling → ResidualAdd
    using a fused TileLang kernel.
    """

    def __init__(self, input_size: int, hidden_size: int, scaling_factor: float):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.scaling_factor = float(scaling_factor)

        # Parameters (same initialization as nn.Linear)
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(input_size)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache keyed by (batch_size, dtype)
        self._kern_cache = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_gemm_sigmoid_residual_kernel(
                M=batch_size,
                N=self.hidden_size,
                K=self.input_size,
                scaling_factor=self.scaling_factor,
                dtype="float16",
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(x_f16.shape[0], x_f16.dtype)
        out_f16 = kernel(x_f16, w_f16, b_f16)

        return out_f16.to(orig_dtype)