"""
Problem Name: 99_Matmul_GELU_Softmax
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.117 runtime_stats={'mean': 0.117, 'std': 0.00921, 'min': 0.111, 'max': 0.198, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0822, 'std': 0.0155, 'min': 0.0758, 'max': 0.23, 'num_trials': 100}, 'speedup_ratio': 0.703}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_linear_gelu_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Out = GELU( X @ W.T + B )
    GELU uses tanh approximation:
        0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
    """

    c0 = 0.044715
    c1 = 0.7978845608028654  # sqrt(2/pi)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), in_dtype),      # (batch, in_features)
        W: T.Tensor((N, K), in_dtype),      # (out_features, in_features)
        B: T.Tensor((N,), in_dtype),        # (out_features,)
        Out: T.Tensor((M, N), in_dtype),    # created by TileLang
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),          # grid.x
            T.ceildiv(M, block_M),          # grid.y
            threads=threads,                # threads per block
        ) as (bx, by):

            # Shared-memory tiles and accumulators
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            Bias_s = T.alloc_shared((block_N,), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Load bias slice once per block
            T.copy(B[bx * block_N], Bias_s)

            # Clear local accumulators
            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)

            # Pipelined K-reduction
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], B_s)
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # Bias + GELU and store
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    v = C_loc[i, j] + T.Cast(accum_dtype, Bias_s[j])

                    # GELU (tanh approximation, fp32 math)
                    v_cub = v * v * v
                    inner = v + T.Cast(accum_dtype, c0) * v_cub
                    t = T.tanh(T.Cast(accum_dtype, c1) * inner)
                    v = T.Cast(accum_dtype, 0.5) * v * (
                        T.Cast(accum_dtype, 1.0) + t
                    )

                    Out[gi, gj] = T.Cast(in_dtype, v)

    return kernel


class ModelNew(nn.Module):
    """
    Linear ➔ GELU (TileLang)
    ➔ Softmax (PyTorch)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # --- identical to nn.Linear default init ---
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # kernel cache {(batch_size, dtype): compiled_kernel}
        self._kernel_cache = {}

    # ------------ kernel retrieval ------------
    def _get_kernel(self, batch: int, dtype: torch.dtype):
        key = (batch, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_gelu_kernel(
                M=batch,
                K=self.in_features,
                N=self.out_features,
            )
        return self._kernel_cache[key]

    # ----------------- forward -----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = torch.device("cuda")

        x_f16 = x.to(device=device, dtype=torch.float16, non_blocking=True)
        w_f16 = self.weight.to(device=device, dtype=torch.float16)
        b_f16 = self.bias.to(device=device, dtype=torch.float16)

        kernel = self._get_kernel(x_f16.shape[0], x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, b_f16)

        # Softmax along feature dim
        y = torch.nn.functional.softmax(y_f16.to(orig_dtype), dim=1)
        return y