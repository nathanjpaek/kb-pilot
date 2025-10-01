"""
Problem Name: 95_Gemm_Add_GlobalAvgPool_AvgPool_Divide
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0726 runtime_stats={'mean': 0.0726, 'std': 0.018, 'min': 0.0646, 'max': 0.216, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.125, 'std': 0.0249, 'min': 0.115, 'max': 0.352, 'num_trials': 100}, 'speedup_ratio': 1.72}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                       GEMM + external-add TileLang kernel                   #
# --------------------------------------------------------------------------- #
def _make_linear_add_kernel(
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
    Computes OUT = X @ Wᵀ + B + Y0          (fp16 I/O, fp32 accumulate)

        X  : [M, K] fp16
        W  : [N, K] fp16   (row-major)
        B  : [N]    fp16
        Y0 : [M, N] fp16   (tensor added element-wise)
        OUT: [M, N] fp16   (created by TileLang, returned)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:  T.Tensor((M, K), in_dtype),
        W:  T.Tensor((N, K), in_dtype),
        B:  T.Tensor((N,),    in_dtype),
        Y0: T.Tensor((M, N),  in_dtype),
        O:  T.Tensor((M, N),  in_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),          # grid-x
            T.ceildiv(M, block_M),          # grid-y
            threads=threads,
        ) as (bx, by):
            # ------------- shared / fragment allocations ------------------ #
            X_s   = T.alloc_shared((block_M, block_K), in_dtype)
            W_s   = T.alloc_shared((block_N, block_K), in_dtype)
            B_s   = T.alloc_shared((block_N,), in_dtype)
            Yadd  = T.alloc_shared((block_M, block_N), in_dtype)
            Acc   = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Copy bias slice (once per block)
            T.copy(B[bx * block_N:(bx + 1) * block_N], B_s)

            # Copy Y0 slice that corresponds to the tile
            T.copy(
                Y0[by * block_M:(by + 1) * block_M,
                   bx * block_N:(bx + 1) * block_N],
                Yadd,
            )

            # Clear accumulators
            T.clear(Acc)

            k_tiles = T.ceildiv(K, block_K)

            # Main K loop, 2-stage pipeline
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
                T.gemm(X_s, W_s, Acc, transpose_B=True)

            # Epilogue: add bias and external tensor, store
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    val = (
                        Acc[i, j]
                        + T.Cast(accum_dtype, B_s[j])
                        + T.Cast(accum_dtype, Yadd[i, j])
                    )
                    O[gi, gj] = T.Cast(in_dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
#                               PyTorch module                                #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Replaces the Linear layer and following addition with a fused TileLang
    kernel. Remaining ops (reshape, AvgPool2d, divide) stay in PyTorch.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.kernel_size = int(kernel_size)

        # ---- Linear parameters (exact same initialisation as nn.Linear) ----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # AvgPool2d (no parameters)
        self.avgpool = nn.AvgPool2d(kernel_size)

        # Kernel cache keyed by (batch_size, dtype)
        self._kern_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # --------------------------- kernel getter --------------------------- #
    def _get_kernel(self, batch: int, dtype: torch.dtype):
        key = (batch, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _make_linear_add_kernel(
                M=batch,
                K=self.in_features,
                N=self.out_features,
            )
        return self._kern_cache[key]

    # ------------------------------- forward ----------------------------- #
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_features)
        y : (B, out_features)  – added after GEMM
        """
        orig_dtype = x.dtype
        device = torch.device("cuda")

        x_f16 = x.to(device=device, dtype=torch.float16).contiguous()
        y_f16 = y.to(device=device, dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device=device, dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device=device, dtype=torch.float16).contiguous()

        B = x_f16.shape[0]
        kernel = self._get_kernel(B, x_f16.dtype)

        # Fused TileLang kernel
        out_f16 = kernel(x_f16, w_f16, b_f16, y_f16)   # (B, out_features)

        # ---------------- Post-processing ----------------- #
        side = int(math.isqrt(self.out_features))       # expect perfect square
        img = out_f16.view(B, side, side).unsqueeze(1)  # (B,1,H,W)
        pooled = self.avgpool(img)                      # k×k avg pool
        normed = pooled / pooled.mean()                 # divide by global mean

        return normed.to(orig_dtype)