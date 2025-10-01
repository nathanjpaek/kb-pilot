"""
Problem Name: 59_Gemm_Gemm_GlobalAvgPool_GlobalAvgPool_Add_AvgPool_Divide
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.146 runtime_stats={'mean': 0.146, 'std': 0.0211, 'min': 0.131, 'max': 0.313, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.122, 'std': 0.0209, 'min': 0.107, 'max': 0.295, 'num_trials': 100}, 'speedup_ratio': 0.836}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                     TileLang GEMM (Linear) kernel factory                   #
# --------------------------------------------------------------------------- #
def _make_linear_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Computes Y = X @ Wᵀ + B          (fp16 I/O, fp32 accumulate)

        X : [M, K]    fp16
        W : [N, K]    fp16   (row-major, NOT transposed)
        B : [N]       fp16
        Y : [M, N]    fp16   (created by TileLang, returned)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_kernel(
        X: T.Tensor((M, K), in_dtype),
        W: T.Tensor((N, K), in_dtype),
        B: T.Tensor((N,), in_dtype),
        Y: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),          # grid-x
            T.ceildiv(M, block_M),          # grid-y
            threads=threads,
        ) as (bx, by):
            # ------------------- shared / fragment allocations ------------------- #
            X_s = T.alloc_shared((block_M, block_K), in_dtype)
            W_s = T.alloc_shared((block_N, block_K), in_dtype)
            B_s = T.alloc_shared((block_N,), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Copy bias slice once per block
            T.copy(B[bx * block_N:(bx + 1) * block_N], B_s)

            # Clear accumulators
            T.clear(C_loc)

            # Main K-loop with 2-stage pipeline
            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(
                    X[by * block_M:(by + 1) * block_M,
                      ko * block_K:(ko + 1) * block_K],
                    X_s,
                )
                T.copy(
                    W[bx * block_N:(bx + 1) * block_N,
                      ko * block_K:(ko + 1) * block_K],
                    W_s,
                )
                # GEMM :  (block_M×block_K) · (block_N×block_K)ᵀ
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            # Epilogue: add bias & store
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, B_s[j])
                    Y[gi, gj] = T.Cast(in_dtype, val)

    return linear_kernel


# --------------------------------------------------------------------------- #
#                             Optimised PyTorch Module                        #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Replacement for the reference model.

    Re-implements the two identical Linear layers with a high-performance
    TileLang GEMM kernel (fp16 I/O, fp32 accumulate, fused bias).
    Remaining inexpensive ops are kept in PyTorch.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,      # kept for interface parity (unused)
        pool_size: int,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.pool_size = int(pool_size)

        # ------------------------ Linear parameters ------------------------- #
        self.weight = nn.Parameter(torch.empty(hidden_features, in_features))
        self.bias = nn.Parameter(torch.empty(hidden_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Extra parameter present in original code (but unused)
        self.add_weight = nn.Parameter(torch.randn(hidden_features))

        # AvgPool2d module (only kernel size matters, no params)
        self.avg_pool = nn.AvgPool2d(pool_size)

        # ------------- kernel cache keyed by (batch_size, dtype) ------------ #
        self._kernel_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # --------------------------------------------------------------------- #
    #                   Retrieve / compile TileLang kernel                  #
    # --------------------------------------------------------------------- #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _make_linear_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.hidden_features,
            )
        return self._kernel_cache[key]

    # --------------------------------------------------------------------- #
    #                                forward                                #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, y : [batch_size, in_features]
        Returns:
            Tensor shaped (1, ‑1) after pooling & normalisation
        """
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # ---------- Move inputs & params to CUDA / fp16 -------------------- #
        x_f16 = x.to(device=device, dtype=torch.float16).contiguous()
        y_f16 = y.to(device=device, dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device=device, dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device=device, dtype=torch.float16).contiguous()

        batch_size = x_f16.shape[0]
        kernel = self._get_kernel(batch_size, x_f16.dtype)

        # --------------------- Linear layers via TileLang ------------------ #
        x_fc = kernel(x_f16, w_f16, b_f16)          # [B, H]  fp16
        y_fc = kernel(y_f16, w_f16, b_f16)          # [B, H]  fp16

        # ------------------- Global average over batch --------------------- #
        x_mean = x_fc.to(torch.float32).mean(dim=0, keepdim=True)   # [1, H]
        y_mean = y_fc.to(torch.float32).mean(dim=0, keepdim=True)   # [1, H]

        # ---------------- Reshape to square images ------------------------- #
        side = int(math.isqrt(self.hidden_features))
        x_img = x_mean.view(1, side, side)          # 3-D tensor
        y_img = y_mean.view(1, side, side)

        # -------------------- Add, AvgPool, Divide ------------------------- #
        added = x_img + y_img                       # [1, side, side]
        pooled = self.avg_pool(added)               # [1, side/ps, side/ps]
        denom = pooled.sum() + 1e-5
        output = pooled / denom

        return output.reshape(1, -1).to(orig_dtype)