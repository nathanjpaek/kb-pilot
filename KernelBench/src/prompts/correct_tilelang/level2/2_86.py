"""
Problem Name: 86_Matmul_Divide_GELU
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0408 runtime_stats={'mean': 0.0408, 'std': 0.00127, 'min': 0.0393, 'max': 0.0509, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0508, 'std': 0.00225, 'min': 0.0488, 'max': 0.0637, 'num_trials': 100}, 'speedup_ratio': 1.25}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                          Kernel factory (Linear/Div/GELU)                  #
# --------------------------------------------------------------------------- #
def _build_linear_div_gelu_kernel(
    M: int,
    K: int,
    N: int,
    divisor: float,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    div_const = float(divisor)
    inv_sqrt2 = 0.7071067811865476  # 1/√2

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), in_dtype),      # (batch, in_features)
        W: T.Tensor((N, K), in_dtype),      # (out_features, in_features)
        B: T.Tensor((N,), in_dtype),        # bias
        Out: T.Tensor((M, N), in_dtype),    # auto-allocated output
    ):
        half_f      = T.Cast(accum_dtype, 0.5)
        inv_s2_f    = T.Cast(accum_dtype, inv_sqrt2)
        div_const_f = T.Cast(accum_dtype, div_const)

        with T.Kernel(
            T.ceildiv(N, block_N),          # grid.x
            T.ceildiv(M, block_M),          # grid.y
            threads=threads,                # threads per block
        ) as (bx, by):
            # ---------------- shared / fragment allocations ---------------- #
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)

            # -------------------- Pipelined GEMM loop ---------------------- #
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(
                    X[by * block_M : (by + 1) * block_M,
                      ko * block_K : (ko + 1) * block_K],
                    A_s,
                )
                T.copy(
                    W[bx * block_N : (bx + 1) * block_N,
                      ko * block_K : (ko + 1) * block_K],
                    B_s,
                )
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # ------------------- Bias, divide, GELU, store ------------------ #
            for i, j in T.Parallel(block_M, block_N):
                g_row = by * block_M + i
                g_col = bx * block_N + j
                if (g_row < M) and (g_col < N):
                    v = C_loc[i, j] + T.Cast(accum_dtype, B[g_col])
                    v = v / div_const_f
                    gelu = half_f * v * (T.Cast(accum_dtype, 1.0) + T.erf(v * inv_s2_f))
                    Out[g_row, g_col] = T.Cast(in_dtype, gelu)

    return kernel


# --------------------------------------------------------------------------- #
#                               PyTorch wrapper                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Linear → divide(by scalar) → GELU  implemented with a fused TileLang kernel.
    """

    def __init__(self, input_size: int, output_size: int, divisor: float):
        super().__init__()
        self.in_features = int(input_size)
        self.out_features = int(output_size)
        self.divisor = float(divisor)

        # ------- Parameters initialised exactly like nn.Linear -------------
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache keyed by (batch_size, dtype)
        self._kernel_cache = {}

    # ------------------------- kernel retrieval --------------------------- #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_div_gelu_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
                divisor=self.divisor,
            )
        return self._kernel_cache[key]

    # ------------------------------ forward ------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(x_fp16.shape[0], x_fp16.dtype)
        y_fp16 = kernel(x_fp16, w_fp16, b_fp16)

        return y_fp16.to(orig_dtype)