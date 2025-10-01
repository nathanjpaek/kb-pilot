"""
Problem Name: 14_Matmul_for_upper_triangular_matrices
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.374 runtime_stats={'mean': 0.374, 'std': 0.0203, 'min': 0.365, 'max': 0.566, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.248, 'std': 0.0163, 'min': 0.237, 'max': 0.398, 'num_trials': 100}, 'speedup_ratio': 0.663}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """Optimized matmul for upper-triangular matrices:  C = triu(A @ B)."""

    def __init__(self):
        super().__init__()
        self._kernel_cache = {}

    # ---------------------------------------------------------------------
    # Kernel factory ------------------------------------------------------
    # ---------------------------------------------------------------------
    @staticmethod
    def _build_triu_kernel(N: int, dtype: str = "float16",
                           block_M: int = 128,
                           block_N: int = 128,
                           block_K: int = 32,
                           num_stages: int = 3,
                           accum_dtype: str = "float32"):
        """Returns a compiled TileLang kernel computing the upper-triangular
        part of A @ B for (N, N) square matrices."""

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            A: T.Tensor((N, N), dtype),
            B: T.Tensor((N, N), dtype),
            C: T.Tensor((N, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N),
                          T.ceildiv(N, block_M),
                          threads=128) as (bx, by):
                # Shared and local buffers ------------------------------------------------
                A_s = T.alloc_shared((block_M, block_K), dtype)
                B_s = T.alloc_shared((block_K, block_N), dtype)
                C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

                # Clear accumulators -------------------------------------------------------
                T.clear(C_frag)

                # Pipeline over K dimension -----------------------------------------------
                for ko in T.Pipelined(T.ceildiv(N, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_s)
                    T.copy(B[ko * block_K, bx * block_N], B_s)
                    T.gemm(A_s, B_s, C_frag)

                # Write back only upper-triangular elements --------------------------------
                for i, j in T.Parallel(block_M, block_N):
                    gi = by * block_M + i
                    gj = bx * block_N + j
                    if (gi < N) and (gj < N) and (gi <= gj):
                        C[gi, gj] = C_frag[i, j]

        return kernel

    # ---------------------------------------------------------------------
    # Kernel cache helper --------------------------------------------------
    # ---------------------------------------------------------------------
    def _get_kernel(self, N: int, dtype: str):
        key = (N, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self._build_triu_kernel(N, dtype)
        return self._kernel_cache[key]

    # ---------------------------------------------------------------------
    # Forward --------------------------------------------------------------
    # ---------------------------------------------------------------------
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA / fp16 inputs
        A_fp16 = A.to(device="cuda", dtype=torch.float16, copy=False)
        B_fp16 = B.to(device="cuda", dtype=torch.float16, copy=False)
        N = A_fp16.shape[0]

        kernel = self._get_kernel(N, "float16")
        C_fp16 = kernel(A_fp16, B_fp16)
        return C_fp16