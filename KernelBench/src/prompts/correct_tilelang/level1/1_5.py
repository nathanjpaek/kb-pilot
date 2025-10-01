"""
Problem Name: 5_Matrix_scalar_multiplication
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.104 runtime_stats={'mean': 0.104, 'std': 0.00533, 'min': 0.0992, 'max': 0.127, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0923, 'std': 0.00537, 'min': 0.0891, 'max': 0.141, 'num_trials': 100}, 'speedup_ratio': 0.887}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _scalar_mul_kernel(M: int,
                       N: int,
                       block_M: int = 128,
                       block_N: int = 128,
                       threads: int = 256,
                       dtype: str = "float16"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A:    T.Tensor((M, N), dtype),
        s_t:  T.Tensor((1,),     dtype),
        Out:  T.Tensor((M, N),   dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N),
                      T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):

            # load scalar into register fragment once per block
            s_val = T.alloc_fragment((1,), dtype)
            s_val[0] = s_t[0]

            # element-wise multiply
            for ly, lx in T.Parallel(block_M, block_N):
                y = by * block_M + ly
                x = bx * block_N + lx
                if (y < M) and (x < N):
                    Out[y, x] = A[y, x] * s_val[0]

    return kernel


class ModelNew(nn.Module):
    """
    TileLang-accelerated matrix-scalar multiplication:  C = A * s
    """

    def __init__(self):
        super().__init__()
        self._kernel_cache = {}    # keyed by (M, N, dtype)

    def _get_kernel(self, M: int, N: int, dtype: torch.dtype):
        key = (M, N, dtype)
        if key not in self._kernel_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._kernel_cache[key] = _scalar_mul_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        orig_dtype = A.dtype
        A_fp16 = A.to(device="cuda", dtype=torch.float16, copy=False)
        s_tensor = torch.tensor([s], dtype=torch.float16, device="cuda")

        M, N = A_fp16.shape
        kernel = self._get_kernel(M, N, A_fp16.dtype)
        C_fp16 = kernel(A_fp16, s_tensor)

        return C_fp16.to(orig_dtype)