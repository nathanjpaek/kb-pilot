"""
Problem Name: 16_Matmul_with_transposed_A
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.127 runtime_stats={'mean': 0.127, 'std': 0.0059, 'min': 0.122, 'max': 0.15, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0721, 'std': 0.00546, 'min': 0.067, 'max': 0.106, 'num_trials': 100}, 'speedup_ratio': 0.568}}
"""

import torch
import tilelang
import tilelang.language as T


def _build_gemm_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """Return a compiled TileLang kernel computing C = A @ B.
    A: (M, K), B: (K, N), C: (M, N)"""

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gemm_kernel(
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

    return gemm_kernel


class ModelNew(torch.nn.Module):
    """Optimized model computing torch.matmul(A.T, B) with TileLang."""

    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int):
        key = (M, N, K)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_gemm_kernel(M, N, K)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        orig_dtype = A.dtype

        # Prepare tensors
        A_fp16 = A.to(device="cuda", dtype=torch.float16).t().contiguous()
        B_fp16 = B.to(device="cuda", dtype=torch.float16).contiguous()

        M, K = A_fp16.shape
        Kb, N = B_fp16.shape
        assert K == Kb, "Inner dimensions must match"

        kernel = self._get_kernel(M, N, K)
        C_fp16 = kernel(A_fp16, B_fp16)
        return C_fp16.to(orig_dtype)