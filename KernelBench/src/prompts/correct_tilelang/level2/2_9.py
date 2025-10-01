"""
Problem Name: 9_Matmul_Subtract_Multiply_ReLU
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0846 runtime_stats={'mean': 0.0846, 'std': 0.00365, 'min': 0.0784, 'max': 0.0971, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0809, 'std': 0.0285, 'min': 0.0731, 'max': 0.36, 'num_trials': 100}, 'speedup_ratio': 0.956}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------
#  Kernel factory
# ---------------------------------------------------------------

def _build_linear_sub_mul_relu_kernel(
    M: int,
    K: int,
    N: int,
    subtract_const: float,
    multiply_const: float,
    *,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    sub_c = float(subtract_const)
    mul_c = float(multiply_const)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), in_dtype),          # (batch, in_features)
        W: T.Tensor((N, K), in_dtype),          # (out_features, in_features)
        B: T.Tensor((N,), in_dtype),            # bias
        Out: T.Tensor((M, N), in_dtype),        # output (allocated by TileLang)
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),              # grid.x
            T.ceildiv(M, block_M),              # grid.y
            threads=threads,
        ) as (bx, by):
            # Shared tiles
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            # Accumulator in registers (fp32)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_frag)

            k_tiles = T.ceildiv(K, block_K)

            # Pipelined K loop
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                # Load global tiles into shared
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], B_s)
                # GEMM  A_s (M×K)  *  B_s^T (K×N)  ->  C_frag
                T.gemm(A_s, B_s, C_frag, transpose_B=True)

            # Epilogue: bias add, subtract const, multiply const, ReLU, store
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    val = C_frag[i, j] + T.Cast(accum_dtype, B[gn])
                    val = (val - sub_c) * mul_c
                    val = T.max(val, T.Cast(accum_dtype, 0))
                    Out[gm, gn] = T.Cast(in_dtype, val)

    return kernel


# ---------------------------------------------------------------
#  PyTorch wrapper  (ModelNew)
# ---------------------------------------------------------------

class ModelNew(nn.Module):
    """Fused Linear – Subtract – Multiply – ReLU using TileLang"""

    def __init__(self, in_features: int, out_features: int, subtract_value: float, multiply_value: float):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.subtract_value = float(subtract_value)
        self.multiply_value = float(multiply_value)

        # ---- Parameters (identical init to nn.Linear) ----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache  {(batch_size, dtype): kernel}
        self._kernel_cache = {}

    # ---------------------------------------------------
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_sub_mul_relu_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
                subtract_const=self.subtract_value,
                multiply_const=self.multiply_value,
            )
        return self._kernel_cache[key]

    # ---------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16)
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16)

        kernel = self._get_kernel(x_f16.shape[0], x_f16.dtype)
        out_f16 = kernel(x_f16, w_f16, b_f16)

        return out_f16.to(orig_dtype)