"""
Problem Name: 10_3D_tensor_matrix_multiplication
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.196 runtime_stats={'mean': 0.196, 'std': 0.0046, 'min': 0.19, 'max': 0.216, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.132, 'std': 0.0232, 'min': 0.124, 'max': 0.357, 'num_trials': 100}, 'speedup_ratio': 0.673}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_matmul_kernel(
    M: int,
    N: int,
    K: int,
    *,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_frag)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_frag)

            T.copy(C_frag, C[by * block_M, bx * block_N])

    return kernel


class ModelNew(nn.Module):
    """
    TileLang-optimized implementation of torch.matmul for a 3-D tensor:
        out[n, m, l] = sum_k A[n, m, k] * B[k, l]
    """

    def __init__(self):
        super().__init__()
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, dtype="float16"):
        key = (M, N, K, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_matmul_kernel(M, N, K, dtype=dtype)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Move to CUDA and fp16
        A_fp16 = A.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        B_fp16 = B.to(device="cuda", dtype=torch.float16, copy=False).contiguous()

        N_dim, M_dim, K_dim = A_fp16.shape
        Kb, L_dim = B_fp16.shape
        assert K_dim == Kb, "Shape mismatch in matmul"

        # Flatten A to 2-D
        A_flat = A_fp16.view(N_dim * M_dim, K_dim)

        kernel = self._get_kernel(A_flat.shape[0], L_dim, K_dim, "float16")
        C_flat = kernel(A_flat, B_fp16)

        return C_flat.view(N_dim, M_dim, L_dim)