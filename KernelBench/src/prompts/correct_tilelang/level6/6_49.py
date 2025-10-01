"""
Problem Name: 49_BatchNorm_Clamp_LeakyReLU_BMM_GroupNorm_Clamp
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.351 runtime_stats={'mean': 0.351, 'std': 0.0131, 'min': 0.343, 'max': 0.462, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.273, 'std': 0.00631, 'min': 0.265, 'max': 0.301, 'num_trials': 100}, 'speedup_ratio': 0.778}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                         TileLang kernel factory (fused)                     #
# --------------------------------------------------------------------------- #
def _build_fused_bmm_kernel(
    B: int,
    M: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Compute   O[b] = LeakyReLU(clamp(X[b], -1, 1)) @ Y[b]
        X : [B, M, K]   (fp16)
        Y : [B, K, M]   (fp16)
        O : [B, M, M]   (fp16)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused_bmm_kernel(
        X: T.Tensor((B, M, K), in_dtype),
        Y: T.Tensor((B, K, M), in_dtype),
        O: T.Tensor((B, M, M), in_dtype),
    ):
        cmin  = T.Cast(accum_dtype, -1.0)
        cmax  = T.Cast(accum_dtype,  1.0)
        alpha = T.Cast(accum_dtype,  0.1)
        zero  = T.Cast(accum_dtype,  0.0)

        grid_x = T.ceildiv(M, block_N)          # tiles along N-dim
        grid_y = T.ceildiv(M, block_M)          # tiles along M-dim

        with T.Kernel(grid_x, grid_y, B, threads=threads) as (bx, by, bb):
            X_s = T.alloc_shared((block_M, block_K), in_dtype)
            Y_s = T.alloc_shared((block_K, block_N), in_dtype)
            C   = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C)

            row_base = by * block_M
            col_base = bx * block_N
            k_tiles  = T.ceildiv(K, block_K)

            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                k_base = ko * block_K

                # ---------------- load X tile with clamp + LeakyReLU --------
                for i, j in T.Parallel(block_M, block_K):
                    gm = row_base + i
                    gk = k_base  + j
                    if (gm < M) and (gk < K):
                        val = X[bb, gm, gk].astype(accum_dtype)
                        val = T.max(val, cmin)
                        val = T.min(val, cmax)
                        # LeakyReLU: max(0,x) + α*min(0,x)
                        pos = T.max(val, zero)
                        neg = T.min(val, zero) * alpha
                        fused = pos + neg
                        X_s[i, j] = T.Cast(in_dtype, fused)
                    else:
                        X_s[i, j] = T.Cast(in_dtype, 0)

                # ---------------- load Y tile ------------------------------
                T.copy(
                    Y[bb, k_base, col_base],
                    Y_s,
                )

                # ---------------- GEMM -------------------------------------
                T.gemm(X_s, Y_s, C)

            # ---------------- store results -------------------------------
            for i, j in T.Parallel(block_M, block_N):
                gm = row_base + i
                gn = col_base + j
                if (gm < M) and (gn < M):
                    O[bb, gm, gn] = T.Cast(in_dtype, C[i, j])

    return fused_bmm_kernel


# --------------------------------------------------------------------------- #
#                               PyTorch wrapper                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    BatchNorm1d → clamp[-1,1] → LeakyReLU(0.1) → BMM  (fused TileLang)
                  → GroupNorm → clamp[-0.5,0.5]
    """

    def __init__(self, num_features: int, num_groups: int, bmm_dim: int):
        super().__init__()
        self.num_features = int(num_features)
        self.bmm_dim = int(bmm_dim)

        self.batch_norm = nn.BatchNorm1d(num_features)
        self.group_norm = nn.GroupNorm(num_groups, num_features)

        # Kernel cache  {(batch_size, dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_bmm_kernel(
                B=batch_size,
                M=self.num_features,
                K=self.bmm_dim,
                in_dtype="float16",
                accum_dtype="float",
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # ---------------- BatchNorm (PyTorch) ---------------------------- #
        x = self.batch_norm(x)

        # ---------------- Prepare tensors for kernel --------------------- #
        x_fp16 = x.to(device=device, dtype=torch.float16, non_blocking=True).contiguous()
        y_fp16 = y.to(device=device, dtype=torch.float16, non_blocking=True).contiguous()

        B = x_fp16.shape[0]
        kernel = self._get_kernel(B, x_fp16.dtype)

        out_fp16 = kernel(x_fp16, y_fp16)          # (B, F, F)

        # ---------------- GroupNorm + final clamp ------------------------ #
        out = out_fp16.to(orig_dtype)
        out = self.group_norm(out)
        out = torch.clamp(out, min=-0.5, max=0.5)

        return out