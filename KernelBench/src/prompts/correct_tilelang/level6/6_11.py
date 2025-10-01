"""
Problem Name: 11_Gemm_GroupNorm_Hardtanh
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0918 runtime_stats={'mean': 0.0918, 'std': 0.0233, 'min': 0.0811, 'max': 0.283, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0848, 'std': 0.0099, 'min': 0.074, 'max': 0.126, 'num_trials': 100}, 'speedup_ratio': 0.924}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """
    Optimised replacement for:
        Linear → GroupNorm → HardTanh
    """

    # --------------------------------------------------------------------- #
    #                                INIT                                   #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        hardtanh_min: float,
        hardtanh_max: float,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_groups = int(num_groups)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.eps = float(eps)

        # -------- Linear parameters (same init as nn.Linear) --------------- #
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

        # -------- GroupNorm affine params ---------------------------------- #
        self.gn_weight = nn.Parameter(torch.ones(out_features))
        self.gn_bias = nn.Parameter(torch.zeros(out_features))

        # -------- Kernel caches keyed by (batch_size, dtype) --------------- #
        self._gemm_kernels: Dict[Tuple[int, torch.dtype], callable] = {}
        self._gn_kernels: Dict[Tuple[int, torch.dtype], callable] = {}

    # --------------------------------------------------------------------- #
    #                       GEMM kernel factory (private)                   #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _build_gemm_kernel(
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
        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            X: T.Tensor((M, K), in_dtype),
            W: T.Tensor((N, K), in_dtype),      # row-major weight
            B: T.Tensor((N,), in_dtype),        # bias
            Out: T.Tensor((M, N), in_dtype),
        ):
            with T.Kernel(
                T.ceildiv(N, block_N),
                T.ceildiv(M, block_M),
                threads=threads,
            ) as (bx, by):
                A_s = T.alloc_shared((block_M, block_K), in_dtype)
                W_s = T.alloc_shared((block_N, block_K), in_dtype)
                Bias_s = T.alloc_shared((block_N,), in_dtype)
                C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

                # Load bias slice once per block
                T.copy(B[bx * block_N:(bx + 1) * block_N], Bias_s)

                T.clear(C_loc)

                k_tiles = T.ceildiv(K, block_K)
                for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                    T.copy(
                        X[by * block_M:(by + 1) * block_M,
                          ko * block_K:(ko + 1) * block_K],
                        A_s,
                    )
                    T.copy(
                        W[bx * block_N:(bx + 1) * block_N,
                          ko * block_K:(ko + 1) * block_K],
                        W_s,
                    )
                    T.gemm(A_s, W_s, C_loc, transpose_B=True)

                # Epilogue: add bias + store
                for i, j in T.Parallel(block_M, block_N):
                    gm = by * block_M + i
                    gn = bx * block_N + j
                    if (gm < M) and (gn < N):
                        val = C_loc[i, j] + T.Cast(accum_dtype, Bias_s[j])
                        Out[gm, gn] = T.Cast(in_dtype, val)

        return kernel

    # --------------------------------------------------------------------- #
    #                   GroupNorm + HardTanh kernel factory                 #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _build_gn_kernel(
        batch_size: int,
        num_channels: int,
        num_groups: int,
        eps_const: float,
        ht_min: float,
        ht_max: float,
        in_dtype: str = "float16",
        accum_dtype: str = "float",
    ):
        group_size = num_channels // num_groups

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            X: T.Tensor((batch_size, num_channels), in_dtype),
            gamma: T.Tensor((num_channels,), in_dtype),
            beta: T.Tensor((num_channels,), in_dtype),
            Y: T.Tensor((batch_size, num_channels), in_dtype),
        ):
            with T.Kernel(batch_size, threads=1) as bx:
                for g in range(num_groups):
                    mean = T.alloc_local((1,), accum_dtype)
                    var = T.alloc_local((1,), accum_dtype)
                    mean[0] = T.Cast(accum_dtype, 0)
                    var[0] = T.Cast(accum_dtype, 0)

                    # First pass: mean / variance
                    for c in range(group_size):
                        idx = g * group_size + c
                        v = T.Cast(accum_dtype, X[bx, idx])
                        mean[0] += v
                        var[0] += v * v

                    denom = T.Cast(accum_dtype, group_size)
                    mean[0] = mean[0] / denom
                    var[0] = var[0] / denom - mean[0] * mean[0]
                    inv_std = T.Cast(accum_dtype, 1.0) / T.sqrt(
                        var[0] + T.Cast(accum_dtype, eps_const)
                    )

                    # Second pass: normalize + affine + HardTanh
                    for c in range(group_size):
                        idx = g * group_size + c
                        v = T.Cast(accum_dtype, X[bx, idx])
                        v = (v - mean[0]) * inv_std
                        v = (
                            v * T.Cast(accum_dtype, gamma[idx])
                            + T.Cast(accum_dtype, beta[idx])
                        )
                        v = T.min(
                            T.max(v, T.Cast(accum_dtype, ht_min)),
                            T.Cast(accum_dtype, ht_max),
                        )
                        Y[bx, idx] = T.Cast(in_dtype, v)

        return kernel

    # --------------------------------------------------------------------- #
    #                       kernel retrieval helpers                        #
    # --------------------------------------------------------------------- #
    def _get_gemm_kernel(self, batch: int, dtype: torch.dtype):
        key = (batch, dtype)
        if key not in self._gemm_kernels:
            self._gemm_kernels[key] = self._build_gemm_kernel(
                M=batch,
                K=self.in_features,
                N=self.out_features,
            )
        return self._gemm_kernels[key]

    def _get_gn_kernel(self, batch: int, dtype: torch.dtype):
        key = (batch, dtype)
        if key not in self._gn_kernels:
            self._gn_kernels[key] = self._build_gn_kernel(
                batch_size=batch,
                num_channels=self.out_features,
                num_groups=self.num_groups,
                eps_const=self.eps,
                ht_min=self.hardtanh_min,
                ht_max=self.hardtanh_max,
            )
        return self._gn_kernels[key]

    # --------------------------------------------------------------------- #
    #                               FORWARD                                 #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # Cast inputs & params to FP16 on CUDA
        x_f16 = x.to(device=device, dtype=torch.float16, non_blocking=True)
        w_f16 = self.weight.to(device=device, dtype=torch.float16, non_blocking=True)
        b_f16 = self.bias.to(device=device, dtype=torch.float16, non_blocking=True)
        gamma_f16 = self.gn_weight.to(device=device, dtype=torch.float16, non_blocking=True)
        beta_f16 = self.gn_bias.to(device=device, dtype=torch.float16, non_blocking=True)

        B = x_f16.shape[0]

        # -------- GEMM -------- #
        gemm_kernel = self._get_gemm_kernel(B, x_f16.dtype)
        y_f16 = gemm_kernel(x_f16, w_f16, b_f16)

        # -------- GroupNorm + HardTanh -------- #
        gn_kernel = self._get_gn_kernel(B, x_f16.dtype)
        out_f16 = gn_kernel(y_f16, gamma_f16, beta_f16)

        return out_f16.to(dtype=orig_dtype)