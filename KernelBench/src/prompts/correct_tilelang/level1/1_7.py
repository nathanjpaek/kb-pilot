"""
Problem Name: 7_Matmul_with_small_K_dimension_
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.598 runtime_stats={'mean': 0.598, 'std': 0.00272, 'min': 0.593, 'max': 0.608, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.227, 'std': 0.0203, 'min': 0.218, 'max': 0.376, 'num_trials': 100}, 'speedup_ratio': 0.38}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def matmul(
    M,
    N,
    K,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
    num_stages: int = 2,
    threads: int = 128,
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            # Allocate shared memory tiles for A and B
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            # Accumulator fragment in registers
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Initialize accumulation buffer
            T.clear(C_local)

            # Loop over the K dimension with pipelining
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Load a tile of A into shared memory
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # Matrix multiply the tiles into the local accumulator
                T.gemm(A_shared, B_shared, C_local)

            # Write the accumulated result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    """
    Optimized model using TileLang for matrix multiplication (C = A * B)
    with very large M, N and small K.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

        # Static shapes based on problem statement
        self.M = 16384
        self.N = 16384
        self.K = 32

        # Compile the TileLang kernel once during module initialization
        self.matmul_kernel = matmul(self.M, self.N, self.K)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Tensor of shape (M, K)
            B: Tensor of shape (K, N)

        Returns:
            Tensor of shape (M, N)
        """
        # TileLang kernels currently run on CUDA and expect fp16 inputs
        A_fp16 = A.to(device="cuda", dtype=torch.float16)
        B_fp16 = B.to(device="cuda", dtype=torch.float16)

        # Execute the compiled TileLang kernel
        C_fp16 = self.matmul_kernel(A_fp16, B_fp16)

        return C_fp16