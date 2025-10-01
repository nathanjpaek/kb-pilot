"""
Problem Name: 59_Matmul_Swish_Scaling
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0479 runtime_stats={'mean': 0.0479, 'std': 0.000874, 'min': 0.0465, 'max': 0.0526, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0577, 'std': 0.00211, 'min': 0.0556, 'max': 0.0703, 'num_trials': 100}, 'speedup_ratio': 1.2}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------------- #
#                              TileLang kernel                                  #
# ----------------------------------------------------------------------------- #
def _build_fused_kernel(
    M: int,
    K: int,
    N: int,
    scaling_value: float,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Kernel computes:
        Out = scaling_value * Swish( X @ W.T + Bias )
    Shapes:
        X   [M, K]   fp16
        W   [N, K]   fp16   (not transposed, we set transpose_B=True)
        Bias[N]      fp16
        Out [M, N]   fp16
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused(
        X: T.Tensor((M, K), in_dtype),
        W: T.Tensor((N, K), in_dtype),
        Bias: T.Tensor((N,), in_dtype),
        Out: T.Tensor((M, N), in_dtype),  # created by TileLang
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),          # grid-x
            T.ceildiv(M, block_M),          # grid-y
            threads=threads,                # 128 threads / block
        ) as (bx, by):
            # ---------------------------------------------------------------- #
            #                    Shared / register allocations                  #
            # ---------------------------------------------------------------- #
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            Bias_s = T.alloc_shared((block_N,), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Pre-load bias slice once per block
            T.copy(Bias[bx * block_N], Bias_s)

            # Clear accumulator
            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                # Copy next K-slice tiles
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], B_s)
                # GEMM: A_s (block_M, block_K) × B_sᵀ (block_K, block_N)
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # ---------------------------------------------------------------- #
            #            Bias add + Swish + scaling and write-back             #
            # ---------------------------------------------------------------- #
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, Bias_s[j])
                    val = val * T.sigmoid(val)           # Swish
                    val = val * scaling_value            # final scaling
                    Out[gi, gj] = T.Cast(in_dtype, val)

    return fused


# ----------------------------------------------------------------------------- #
#                                 PyTorch module                                #
# ----------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimised implementation of:
        Y = scaling_factor * Swish( X @ W.T + B )
    """

    def __init__(self, in_features: int, out_features: int, scaling_factor: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = float(scaling_factor)

        # ---- Parameters initialised identically to nn.Linear ----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.empty(out_features))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache keyed by (batch_size, dtype)
        self._kernel_cache = {}

    # --------------------------- kernel retrieval --------------------------- #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            kernel = _build_fused_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
                scaling_value=self.scaling_factor,
            )
            self._kernel_cache[key] = kernel
        return self._kernel_cache[key]

    # -------------------------------- forward ------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16, non_blocking=True)
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16)

        kernel = self._get_kernel(x_f16.shape[0], x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, b_f16)

        return y_f16.to(orig_dtype)