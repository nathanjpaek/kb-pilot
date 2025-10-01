"""
Problem Name: 12_Matmul_Scaling_ResidualAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0349 runtime_stats={'mean': 0.0349, 'std': 0.00541, 'min': 0.0296, 'max': 0.0633, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0631, 'std': 0.00883, 'min': 0.0528, 'max': 0.0929, 'num_trials': 100}, 'speedup_ratio': 1.81}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                        TileLang kernel factory                              #
# --------------------------------------------------------------------------- #
def _build_linear_scale_kernel(
    M: int,
    N: int,
    K: int,
    scale_const: float,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float",
):
    """Kernel:  Out = (X @ Wᵀ + B) * scale_const"""
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), in_dtype),
        W: T.Tensor((N, K), in_dtype),         # not transposed
        B: T.Tensor((N,), in_dtype),
        Out: T.Tensor((M, N), in_dtype),       # allocated by TileLang
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=2):
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], B_s)
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # bias add + scaling + write-back
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, B[gj])
                    val *= scale_const
                    Out[gi, gj] = T.Cast(in_dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
#                               PyTorch Module                                #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimised TileLang implementation of:
        out = (Linear(x) * scaling_factor) + Linear(x)
    which equals Linear(x) * (1 + scaling_factor)
    """

    def __init__(self, in_features: int, out_features: int, scaling_factor: float):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scale_const = 1.0 + float(scaling_factor)

        # Parameters initialised identically to nn.Linear defaults
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Cache of compiled kernels keyed by (batch_size, dtype)
        self._kernel_cache = {}

    # --------------------------- kernel retrieval --------------------------- #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            kern = _build_linear_scale_kernel(
                M=batch_size,
                N=self.out_features,
                K=self.in_features,
                scale_const=self.scale_const,
                in_dtype="float16",
                accum_dtype="float",
            )
            self._kernel_cache[key] = kern
        return self._kernel_cache[key]

    # -------------------------------- forward ------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Move to CUDA / fp16
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16)

        # Compile / fetch kernel and run
        kernel = self._get_kernel(x_f16.shape[0], x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, b_f16)

        return y_f16.to(orig_dtype)