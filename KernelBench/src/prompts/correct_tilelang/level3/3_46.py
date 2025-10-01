"""
Problem Name: 46_NetVladWithGhostClusters
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.273 runtime_stats={'mean': 0.273, 'std': 0.00836, 'min': 0.263, 'max': 0.32, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.264, 'std': 0.00959, 'min': 0.251, 'max': 0.333, 'num_trials': 100}, 'speedup_ratio': 0.967}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# GEMM kernel factory : (M,K) @ (K,N) = (M,N)                                 #
# --------------------------------------------------------------------------- #
def _build_gemm_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),    # (M,K)  ←  x   (B·N , D)
        B: T.Tensor((K, N), dtype),    # (K,N)  ←  clusters (D , K+G)
        C: T.Tensor((M, N), dtype),    # output (auto-alloc)
    ):
        grid_x = T.ceildiv(N, block_N)
        grid_y = T.ceildiv(M, block_M)

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            As = T.alloc_shared((block_M, block_K), dtype)
            Bs = T.alloc_shared((block_K, block_N), dtype)
            Cl = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(Cl)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                # copy tiles
                T.copy(
                    A[by * block_M : (by + 1) * block_M,
                      ko * block_K : (ko + 1) * block_K],
                    As,
                )
                T.copy(
                    B[ko * block_K : (ko + 1) * block_K,
                      bx * block_N : (bx + 1) * block_N],
                    Bs,
                )
                # GEMM
                T.gemm(As, Bs, Cl)

            # store with bounds check
            for i, j in T.Parallel(block_M, block_N):
                glo_i = by * block_M + i
                glo_j = bx * block_N + j
                if (glo_i < M) and (glo_j < N):
                    C[glo_i, glo_j] = T.Cast(dtype, Cl[i, j])

    return kernel


# --------------------------------------------------------------------------- #
# Optimised NetVLAD-style aggregation                                         #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    TileLang-accelerated version of the original model.
    Replaces the first (B·N,D)×(D,K+G) matmul with an optimised TileLang GEMM.
    """

    def __init__(self, cluster_size: int, feature_size: int, ghost_clusters: int):
        super().__init__()

        self.feature_size = int(feature_size)      #  D
        self.cluster_size = int(cluster_size)      #  K
        self.ghost_clusters = int(ghost_clusters)  #  G
        self.total_clusters = self.cluster_size + self.ghost_clusters  # K+G

        init_sc = 1.0 / math.sqrt(self.feature_size)

        # -------- parameters (identical initialisation) --------
        self.clusters = nn.Parameter(
            init_sc * torch.randn(self.feature_size, self.total_clusters)
        )
        self.batch_norm = nn.BatchNorm1d(self.total_clusters)
        self.clusters2 = nn.Parameter(
            init_sc * torch.randn(1, self.feature_size, self.cluster_size)
        )

        self.out_dim = self.cluster_size * self.feature_size

        # -------- kernel cache --------
        self._gemm_cache: Dict[Tuple[int, str], callable] = {}

        # batch-norm needs fp16 weights once moved to GPU
        self._bn_fp16_ready = False

    # ------------------------------------------------------ #
    def _get_kernel(self, M: int, dtype: str = "float16"):
        key = (M, dtype)
        if key not in self._gemm_cache:
            self._gemm_cache[key] = _build_gemm_kernel(
                M,
                self.feature_size,
                self.total_clusters,
                dtype=dtype,
            )
        return self._gemm_cache[key]

    # ------------------------------------------------------ #
    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x : (B , N , D)
        Returns:
            (B , D·K)
        """
        B, N, D = x.shape
        assert D == self.feature_size

        # --- prepare tensors on CUDA / fp16 ---
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        W_fp16 = self.clusters.to(device="cuda", dtype=torch.float16).contiguous()

        M = B * N  # flattened rows
        gemm = self._get_kernel(M, "float16")
        assignment_fp16 = gemm(x_fp16.view(M, D), W_fp16)  # (M , K+G)

        # ensure BatchNorm module is on GPU / fp16 exactly once
        if not self._bn_fp16_ready:
            self.batch_norm = self.batch_norm.to(device="cuda", dtype=torch.float16)
            self._bn_fp16_ready = True

        assignment_bn = self.batch_norm(assignment_fp16)
        assignment_sm = F.softmax(assignment_bn, dim=1)

        # remove ghost clusters, reshape  (M,K) -> (B,N,K)
        assignment_sm = assignment_sm[:, : self.cluster_size]
        assignment_sm = assignment_sm.view(B, N, self.cluster_size)

        # a_sum * clusters2
        a_sum = assignment_sm.sum(dim=1, keepdim=True)            # (B,1,K)
        a = a_sum * self.clusters2.to(device="cuda", dtype=torch.float16)  # (B,D,K)

        # assignment^T @ x
        assignment_t = assignment_sm.transpose(1, 2)  # (B,K,N)
        vlad = torch.matmul(assignment_t, x_fp16).transpose(1, 2)  # (B,D,K)
        vlad = vlad - a

        # intra-normalisation, flatten, L2 normalise
        vlad = F.normalize(vlad, dim=1)
        vlad = vlad.reshape(B, -1)
        vlad = F.normalize(vlad, dim=1)

        return vlad.to(x.dtype)  # restore original dtype