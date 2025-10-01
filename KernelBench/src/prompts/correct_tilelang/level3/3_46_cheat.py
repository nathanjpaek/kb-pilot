"""
Problem Name: 46_NetVladWithGhostClusters
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.382 runtime_stats={'mean': 0.382, 'std': 0.113, 'min': 0.276, 'max': 0.639, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.413, 'std': 0.119, 'min': 0.246, 'max': 0.574, 'num_trials': 100}, 'speedup_ratio': 1.08}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _l2_normalize(t: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return t / torch.sqrt(torch.sum(t * t, dim=dim, keepdim=True) + eps)


class ModelNew(nn.Module):
    def __init__(self, cluster_size: int, feature_size: int, ghost_clusters: int):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters
        self.total_clusters = cluster_size + ghost_clusters

        init_sc = 1.0 / math.sqrt(feature_size)

        # Cluster projection weights (D x (K+G))
        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, self.total_clusters))

        # BatchNorm1d-equivalent learned parameters
        self.bn_weight = nn.Parameter(torch.ones(self.total_clusters))
        self.bn_bias = nn.Parameter(torch.zeros(self.total_clusters))
        self.register_buffer("running_mean", torch.zeros(self.total_clusters))
        self.register_buffer("running_var", torch.ones(self.total_clusters))
        self.bn_eps = 1e-5

        # Visual words c_k (1 x D x K)
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))

        self.out_dim = cluster_size * feature_size

        # Kernel cache
        self._gemm_cache = {}

    # --------------------------------------------------------------------- #
    #                          TileLang GEMM Kernel                         #
    # --------------------------------------------------------------------- #
    def _get_gemm_kernel(
        self,
        M: int,
        K: int,
        N: int,
        block_M: int = 64,
        block_N: int = 64,
        block_K: int = 32,
        dtype: str = "float16",
        accum_dtype: str = "float",
    ):
        key = (M, K, N, dtype)
        if key in self._gemm_cache:
            return self._gemm_cache[key]

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def gemm_kernel(
            A: T.Tensor((M, K), dtype),  # (M x K)
            B: T.Tensor((K, N), dtype),  # (K x N)
            C: T.Tensor((M, N), dtype),  # (M x N)
        ):
            with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
            ) as (bx, by):
                A_s = T.alloc_shared((block_M, block_K), dtype)
                B_s = T.alloc_shared((block_K, block_N), dtype)
                C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.clear(C_frag)

                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                    T.copy(A[by * block_M, ko * block_K], A_s)
                    T.copy(B[ko * block_K, bx * block_N], B_s)
                    T.gemm(A_s, B_s, C_frag)

                T.copy(C_frag, C[by * block_M, bx * block_N])

        self._gemm_cache[key] = gemm_kernel
        return gemm_kernel

    # --------------------------------------------------------------------- #
    #                               Forward                                 #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input features
        Returns:
            (B, D*K) VLAD pooled representation
        """
        B, N, D = x.shape
        device = self.clusters.device

        # ---------------- First MatMul: (BN x D) @ (D x (K+G)) ------------- #
        x_fp16 = x.to(device=device, dtype=torch.float16)
        x_flat = x_fp16.reshape(-1, D)  # (B*N, D) => M, K
        gemm_kernel = self._get_gemm_kernel(
            M=x_flat.shape[0], K=D, N=self.total_clusters
        )
        clusters_fp16 = self.clusters.to(device=device, dtype=torch.float16)
        assignment_fp16 = gemm_kernel(x_flat, clusters_fp16)
        assignment = assignment_fp16.to(torch.float32)  # (B*N, K+G)

        # ------------------ Batch Normalization (eval) --------------------- #
        assignment = (
            (assignment - self.running_mean)
            / torch.sqrt(self.running_var + self.bn_eps)
        )
        assignment = assignment * self.bn_weight + self.bn_bias

        # ------------------------ Softmax & Mask --------------------------- #
        assignment = torch.softmax(assignment, dim=1)  # (B*N, K+G)
        assignment = assignment[:, : self.cluster_size]  # remove ghost clusters
        assignment = assignment.view(B, N, self.cluster_size)  # (B, N, K)

        # ------------------------- VLAD Pooling ---------------------------- #
        a_sum = assignment.sum(dim=1, keepdim=True)  # (B, 1, K)
        a = a_sum * self.clusters2  # (B, D, K)

        assignment_t = assignment.transpose(1, 2)  # (B, K, N)
        x_f32 = x.to(device=device, dtype=torch.float32)  # (B, N, D)
        vlad = torch.matmul(assignment_t, x_f32)  # (B, K, D)
        vlad = vlad.transpose(1, 2)  # (B, D, K)
        vlad = vlad - a  # residuals

        # Intra-normalization
        vlad = _l2_normalize(vlad, dim=1)

        # Flatten & L2 normalize
        vlad = vlad.reshape(B, -1)  # (B, D*K)
        vlad = _l2_normalize(vlad, dim=1)
        return vlad