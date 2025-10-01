"""
Problem Name: 19_Gemm_Sigmoid_Scaling_ResidualAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0446 runtime_stats={'mean': 0.0446, 'std': 0.00142, 'min': 0.0433, 'max': 0.0521, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0568, 'std': 0.00244, 'min': 0.0547, 'max': 0.0709, 'num_trials': 100}, 'speedup_ratio': 1.27}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------
# Kernel factory (fused Linear + Sigmoid + Scale + ResidualAdd)
# ----------------------------------------------------------------------
def _build_fused_kernel(
    M: int,          # batch size
    K: int,          # input_size
    N: int,          # hidden_size
    scale_val: float,
    *,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 64,
    num_stages: int = 2,
    dtype_in: str = "float16",
    accum_dtype: str = "float32",
):
    one_f32 = 1.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype_in),
        W: T.Tensor((N, K), dtype_in),
        B: T.Tensor((N,), dtype_in),
        Out: T.Tensor((M, N), dtype_in),        # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),              # grid-x over columns
            T.ceildiv(M, block_M),              # grid-y over rows
            threads=128,
        ) as (bx, by):

            # Shared tiles
            A_s = T.alloc_shared((block_M, block_K), dtype_in)
            B_s = T.alloc_shared((block_N, block_K), dtype_in)
            Bias_s = T.alloc_shared((block_N,), dtype_in)

            # Local accumulator
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            # Load bias slice once per block
            T.copy(B[bx * block_N : (bx + 1) * block_N], Bias_s)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                # Copy tiles
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
                # GEMM (transpose_B = True)
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # Epilogue: bias, sigmoid, scale, residual add, store
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    val = C_loc[i, j] + Bias_s[j].astype(accum_dtype)
                    sig = one_f32 / (one_f32 + T.exp(-val))
                    out_val = val + scale_val * sig
                    Out[gm, gn] = T.Cast(dtype_in, out_val)

    return kernel


# ----------------------------------------------------------------------
# PyTorch wrapper
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized implementation of:
        out = (X @ Wᵀ + B) + scale * sigmoid(X @ Wᵀ + B)
    """

    def __init__(self, input_size: int, hidden_size: int, scaling_factor: float):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.scale_val = float(scaling_factor)

        # ----- Parameters (match nn.Linear defaults) -----
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(input_size)
            nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache keyed by (batch_size, dtype)
        self._kernel_cache: dict[tuple[int, str], callable] = {}

    # ------------------------------------------------------------------
    # Kernel getter
    # ------------------------------------------------------------------
    def _get_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_fused_kernel(
                batch_size,
                self.input_size,
                self.hidden_size,
                self.scale_val,
                dtype_in=dtype,
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        B = x_fp16.shape[0]
        kernel = self._get_kernel(B, "float16")
        out_fp16 = kernel(x_fp16, w_fp16, b_fp16)
        return out_fp16.to(orig_dtype)