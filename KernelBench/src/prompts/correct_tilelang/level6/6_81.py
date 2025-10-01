"""
Problem Name: 81_BatchNorm_ReLU_MaxPool_Matmul_Max
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.21 runtime_stats={'mean': 1.21, 'std': 0.0118, 'min': 1.2, 'max': 1.31, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.16, 'std': 0.0145, 'min': 1.15, 'max': 1.28, 'num_trials': 100}, 'speedup_ratio': 0.959}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                         TileLang GEMM kernel factory                        #
# --------------------------------------------------------------------------- #
def _build_matmul_kernel(
    B: int,
    F: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Computes  O = X @ Y.T
        X : (B, F)  in_dtype
        Y : (B, F)  in_dtype
        O : (B, B)  in_dtype  (auto-allocated by TileLang)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gemm(
        X: T.Tensor((B, F), in_dtype),
        Y: T.Tensor((B, F), in_dtype),
        O: T.Tensor((B, B), in_dtype),         # auto-allocated
    ):
        grid_x = T.ceildiv(B, block_N)         # tiles along N (cols)
        grid_y = T.ceildiv(B, block_M)         # tiles along M (rows)

        with T.Kernel(grid_x, grid_y, threads=threads) as (bx, by):
            # Shared‐memory tiles
            X_s = T.alloc_shared((block_M, block_K), in_dtype)
            Y_s = T.alloc_shared((block_K, block_N), in_dtype)

            # Fragment accumulator
            C   = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C)

            row_base = by * block_M
            col_base = bx * block_N
            k_tiles  = T.ceildiv(F, block_K)

            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                k_base = ko * block_K

                # ----- load X tile (rows of X) --------------------------------
                for mi, ki in T.Parallel(block_M, block_K):
                    gm = row_base + mi
                    gk = k_base  + ki
                    if (gm < B) and (gk < F):
                        X_s[mi, ki] = X[gm, gk]
                    else:
                        X_s[mi, ki] = T.Cast(in_dtype, 0)

                # ----- load Y tile, **transposed** ----------------------------
                for ki, nj in T.Parallel(block_K, block_N):
                    gk = k_base  + ki          # original feature dim
                    gn = col_base + nj         # batch index (column in O)
                    if (gn < B) and (gk < F):
                        Y_s[ki, nj] = Y[gn, gk]    # note the transpose
                    else:
                        Y_s[ki, nj] = T.Cast(in_dtype, 0)

                # ----- matrix multiply ----------------------------------------
                T.gemm(X_s, Y_s, C)

            # ----- store result tile ------------------------------------------
            for mi, nj in T.Parallel(block_M, block_N):
                gm = row_base + mi
                gn = col_base + nj
                if (gm < B) and (gn < B):
                    O[gm, gn] = T.Cast(in_dtype, C[mi, nj])

    return gemm


# --------------------------------------------------------------------------- #
#                               PyTorch wrapper                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    BN2d → ReLU → MaxPool2d(k=2,s=2) → flatten → Linear
           → TileLang GEMM ( X @ Y.T ) → max()
    """

    def __init__(self, num_features: int, num_channels: int, height: int, width: int):
        super().__init__()

        # ---------------- PyTorch layers (identical init) -----------------
        self.bn = nn.BatchNorm2d(num_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_feat = num_channels * (height // 2) * (width // 2)
        self.linear = nn.Linear(in_feat, num_features)
        # (nn.Linear already initialises weight & bias exactly as required)

        # Static dims
        self.num_features = int(num_features)
        self.num_channels = int(num_channels)
        self.height = int(height)
        self.width  = int(width)

        # Kernel cache :  {(B,F,dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, torch.dtype], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, B: int, F: int, dtype: torch.dtype):
        key = (B, F, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_matmul_kernel(B, F)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args
            x : (B, C, H, W)
            y : (B, F)
        Returns
            scalar tensor (same dtype as input)
        """
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # ---------------- pre-processing on GPU --------------------------
        x = x.to(device=device, dtype=torch.float16)
        y_fp16 = y.to(device=device, dtype=torch.float16).contiguous()

        self.bn = self.bn.to(device)
        self.linear = self.linear.to(device, dtype=torch.float16)

        x = self.bn(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        B = x.shape[0]
        x_flat = x.view(B, -1)
        x_fp16 = F.linear(
            x_flat,
            self.linear.weight,
            self.linear.bias,
        )  # (B, F) ‑- already fp16

        # ---------------- TileLang GEMM ---------------------------------
        kernel = self._get_kernel(B, self.num_features, x_fp16.dtype)
        out_mat = kernel(x_fp16.contiguous(), y_fp16)   # (B, B) fp16

        # ---------------- final reduction -------------------------------
        scalar = out_mat.max().to(orig_dtype)
        return scalar