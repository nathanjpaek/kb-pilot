"""
Problem Name: 4_Matrix_vector_multiplication_
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=20.7 runtime_stats={'mean': 20.7, 'std': 0.0592, 'min': 20.6, 'max': 20.8, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0517, 'std': 0.0107, 'min': 0.0488, 'max': 0.156, 'num_trials': 100}, 'speedup_ratio': 0.0025}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def gemv_kernel(M, K, BLOCK_M=128, BLOCK_K=128, dtype="float16", accum_dtype="float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, 1), dtype),
        C: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M)) as bm:
            tm = T.get_thread_binding(0)  # tm = threadIdx.x
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_K,), dtype)
            C_reg = T.alloc_local((1,), accum_dtype)
            T.clear(C_reg)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for tk in T.serial(BLOCK_K):
                    # Each thread loads its own A row, all threads load same B elements
                    A_shared[tm, tk] = A[bm * BLOCK_M + tm, bk * BLOCK_K + tk]
                    B_shared[tk] = B[bk * BLOCK_K + tk, 0]
                for tk in T.serial(BLOCK_K):
                    C_reg[0] += A_shared[tm, tk].astype(accum_dtype) * B_shared[tk].astype(accum_dtype)
            C[bm * BLOCK_M + tm, 0] = C_reg[0]

    return main

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.compiled_kernels = {}
    
    def forward(self, A, B):
        A = A.to(device="cuda", dtype=torch.float16)
        B = B.to(device="cuda", dtype=torch.float16)
        
        M, K = A.shape
        
        shape_key = (M, K)
        if shape_key not in self.compiled_kernels:
            kernel_func = gemv_kernel(M, K)
            self.compiled_kernels[shape_key] = kernel_func
        
        return self.compiled_kernels[shape_key](A, B).to(torch.float32)