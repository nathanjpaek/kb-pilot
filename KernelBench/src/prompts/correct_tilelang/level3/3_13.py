"""
Problem Name: 13_DenseNet121TransitionLayer
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.07 runtime_stats={'mean': 1.07, 'std': 0.0043, 'min': 1.06, 'max': 1.09, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.701, 'std': 0.01, 'min': 0.693, 'max': 0.784, 'num_trials': 100}, 'speedup_ratio': 0.655}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory for 1×1-Conv (GEMM)                                 #
# --------------------------------------------------------------------------- #
def _build_conv1x1_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Compute  Y[M,N] = A[M,K] @ B[K,N]   (B is weight-T)
    """
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv1x1(
        A: T.Tensor((M, K), dtype),       # input (flattened)
        B: T.Tensor((K, N), dtype),       # weight.T
        Y: T.Tensor((M, N), dtype),       # output
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),        # grid-x : columns
            T.ceildiv(M, block_M),        # grid-y : rows
            threads=128,
        ) as (bx, by):

            As = T.alloc_shared((block_M, block_K), dtype)
            Bs = T.alloc_shared((block_K, block_N), dtype)
            Cf = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(Cf)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy( A[by * block_M, ko * block_K] , As )
                T.copy( B[ko * block_K, bx * block_N] , Bs )
                T.gemm(As, Bs, Cf)

            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if (m < M) and (n < N):
                    Y[m, n] = T.Cast(dtype, Cf[i, j])

    return conv1x1


# --------------------------------------------------------------------------- #
# PyTorch module with TileLang acceleration                                   #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Transition block: BatchNorm2d → ReLU → (TileLang 1×1 Conv) → AvgPool2d
    """
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()

        # ---------- PyTorch BN + ReLU (unchanged) ---------------------------
        self.bn   = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)

        # ---------- 1×1-Conv parameters (bias = False) ----------------------
        w_shape = (num_output_features, num_input_features)   # (C_out, C_in)
        self.weight = nn.Parameter(torch.empty(w_shape))

        # identical initialisation to nn.Conv2d --------------------------------
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # ---------- AvgPool parameters ---------------------------------------
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # ---------- Kernel cache ---------------------------------------------
        self.K = num_input_features
        self.N = num_output_features
        self._kern_cache: Dict[Tuple[int, str], callable] = {}

    # --------------------------------------------------------------------- #
    def _get_kernel(self, M: int, dtype: str):
        key = (M, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_conv1x1_kernel(
                M, self.K, self.N, dtype=dtype
            )
        return self._kern_cache[key]

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(x))               # BN → ReLU

        # -------- prepare data for TileLang GEMM ----------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        N_b, C_in, H, W = x_fp16.shape
        M = N_b * H * W

        # (N, C_in, H, W) → (M, K)  with channels last
        A_mat = (
            x_fp16
            .permute(0, 2, 3, 1)                # N H W C
            .contiguous()
            .view(M, C_in)
        )

        # Weight.T  → shape (K, N)
        B_mat = self.weight.t().contiguous().to(device="cuda", dtype=torch.float16)

        # -------- obtain / launch kernel ------------------------------------
        kernel = self._get_kernel(M, "float16")
        Y_mat  = kernel(A_mat, B_mat)           # (M, C_out)

        # -------- reshape back to NCHW & pool -------------------------------
        Y_nhwc = Y_mat.view(N_b, H, W, self.N)
        Y_nchw = Y_nhwc.permute(0, 3, 1, 2).contiguous()

        out = self.pool(Y_nchw)

        return out.to(dtype=x.dtype)