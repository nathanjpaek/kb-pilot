"""
Problem Name: 3_Batched_matrix_multiplication
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0944 runtime_stats={'mean': 0.0944, 'std': 0.00346, 'min': 0.09, 'max': 0.108, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0449, 'std': 0.00392, 'min': 0.0412, 'max': 0.0682, 'num_trials': 100}, 'speedup_ratio': 0.476}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_batched_matmul_kernel(
    B: int,
    M: int,
    N: int,
    K: int,
    *,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float32",
    num_stages: int = 3,
):
    num_n_blocks = (N + block_N - 1) // block_N
    grid_x = B * num_n_blocks
    grid_y = (M + block_M - 1) // block_M

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def batched_gemm(
        A: T.Tensor((B, M, K), dtype),
        Bm: T.Tensor((B, K, N), dtype),
        C: T.Tensor((B, M, N), dtype),
    ):
        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            batch_id = bx // num_n_blocks
            n_blk_id = bx % num_n_blocks

            m_base = by * block_M
            n_base = n_blk_id * block_N

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_frag   = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_frag)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                k_base = ko * block_K

                # Load A tile with boundary checks
                for i, k in T.Parallel(block_M, block_K):
                    g_m = m_base + i
                    g_k = k_base + k
                    A_shared[i, k] = T.if_then_else(
                        (g_m < M) & (g_k < K),
                        A[batch_id, g_m, g_k],
                        0,
                    )

                # Load B tile with boundary checks
                for k, j in T.Parallel(block_K, block_N):
                    g_k = k_base + k
                    g_n = n_base + j
                    B_shared[k, j] = T.if_then_else(
                        (g_k < K) & (g_n < N),
                        Bm[batch_id, g_k, g_n],
                        0,
                    )

                # Compute GEMM on the loaded tiles
                T.gemm(A_shared, B_shared, C_frag)

            # Write results back with boundary checks
            for i, j in T.Parallel(block_M, block_N):
                g_m = m_base + i
                g_n = n_base + j
                if (g_m < M) & (g_n < N):
                    C[batch_id, g_m, g_n] = C_frag[i, j].astype(dtype)

    return batched_gemm


class ModelNew(nn.Module):
    """
    TileLang-optimized batched matrix multiplication.
    Computes C = torch.bmm(A, B) for inputs
        A : (B, M, K)
        B : (B, K, N)
    """

    def __init__(self):
        super().__init__()
        self._kernel_cache = {}

    def _get_kernel(self, B, M, N, K, dtype_str="float16"):
        key = (B, M, N, K, dtype_str)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_batched_matmul_kernel(
                B, M, N, K, dtype=dtype_str
            )
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, Bm: torch.Tensor) -> torch.Tensor:
        orig_dtype = A.dtype
        A_fp16 = A.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        B_fp16 = Bm.to(device="cuda", dtype=torch.float16, copy=False).contiguous()

        Bsz, M, K = A_fp16.shape
        Bk, Kb, N = B_fp16.shape[0], B_fp16.shape[1], B_fp16.shape[2]
        assert Bsz == Bk and K == Kb, "Dimension mismatch for batched matmul"

        kernel = self._get_kernel(Bsz, M, N, K, "float16")
        C_fp16 = kernel(A_fp16, B_fp16)
        return C_fp16.to(orig_dtype)