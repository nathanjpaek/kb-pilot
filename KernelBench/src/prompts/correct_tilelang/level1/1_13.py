"""
Problem Name: 13_Matmul_for_symmetric_matrices
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.385 runtime_stats={'mean': 0.385, 'std': 0.00446, 'min': 0.376, 'max': 0.405, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.213, 'std': 0.0131, 'min': 0.206, 'max': 0.336, 'num_trials': 100}, 'speedup_ratio': 0.553}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_matmul_kernel(
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((N, N), dtype),
        B: T.Tensor((N, N), dtype),
        C: T.Tensor((N, N), dtype),
    ):
        grid_x = T.ceildiv(N, block_N)
        grid_y = T.ceildiv(N, block_M)

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_frag)

            k_tiles = T.ceildiv(N, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_frag)

            T.copy(C_frag, C[by * block_M, bx * block_N])

    return matmul_kernel


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel_cache = {}  # keyed by (N, dtype)

    def _get_kernel(self, N: int, dtype: torch.dtype):
        key = (N, dtype)
        if key not in self._kernel_cache:
            tl_dtype = "float16" if dtype == torch.float16 else "bfloat16"
            self._kernel_cache[key] = _build_matmul_kernel(N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        orig_dtype = A.dtype
        A_fp16 = A.to(device="cuda", dtype=torch.float16, copy=False)
        B_fp16 = B.to(device="cuda", dtype=torch.float16, copy=False)

        N, _ = A_fp16.shape
        kernel = self._get_kernel(N, A_fp16.dtype)
        C_fp16 = kernel(A_fp16, B_fp16)

        return C_fp16.to(orig_dtype)