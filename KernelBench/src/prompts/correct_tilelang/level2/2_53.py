"""
Problem Name: 53_Gemm_Scaling_Hardtanh_GELU
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0706 runtime_stats={'mean': 0.0706, 'std': 0.0227, 'min': 0.0603, 'max': 0.255, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.101, 'std': 0.04, 'min': 0.0889, 'max': 0.48, 'num_trials': 100}, 'speedup_ratio': 1.43}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scaling_factor: float,
        hardtanh_min: float,
        hardtanh_max: float,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = float(scaling_factor)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

        # Parameters (identical initialization to nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache for different batch sizes / dtypes
        self._kernel_cache = {}

    # ------------------------------------------------------------------
    # Kernel factory
    # ------------------------------------------------------------------
    def _build_kernel(self, batch_size: int, dtype: str = "float16"):
        M, K, N = batch_size, self.in_features, self.out_features

        # Tile configuration
        block_M, block_N, block_K = 128, 128, 64
        num_stages, threads = 2, 256
        scale_val = self.scaling_factor
        ht_min, ht_max = self.hardtanh_min, self.hardtanh_max

        accum_dtype = "float"

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def fused_kernel(
            X: T.Tensor((M, K), dtype),
            W: T.Tensor((N, K), dtype),
            B: T.Tensor((N,), dtype),
            Out: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                # Buffers
                X_s = T.alloc_shared((block_M, block_K), dtype)
                W_s = T.alloc_shared((block_N, block_K), dtype)  # Transposed within gemm
                C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.clear(C_loc)

                # K reduction loop
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(X[by * block_M, ko * block_K], X_s)
                    T.copy(W[bx * block_N, ko * block_K], W_s)
                    T.gemm(X_s, W_s, C_loc, transpose_B=True)

                # Epilogue: bias, scaling, hardtanh, GELU and store
                for i, j in T.Parallel(block_M, block_N):
                    gm = by * block_M + i
                    gn = bx * block_N + j
                    if (gm < M) and (gn < N):
                        val = C_loc[i, j] + T.Cast(accum_dtype, B[gn])
                        val *= scale_val
                        val = T.min(T.max(val, ht_min), ht_max)
                        tmp = val
                        val = (
                            0.5
                            * tmp
                            * (
                                1
                                + T.tanh(
                                    0.7978845608028654
                                    * (tmp + 0.044715 * tmp * tmp * tmp)
                                )
                            )
                        )
                        Out[gm, gn] = T.Cast(dtype, val)

        return fused_kernel

    def _get_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self._build_kernel(batch_size, dtype)
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16)
        W = self.weight.to(device="cuda", dtype=torch.float16)
        B = self.bias.to(device="cuda", dtype=torch.float16)

        batch_size = x.shape[0]
        kernel = self._get_kernel(batch_size, "float16")
        out = kernel(x, W, B)
        return out.to(torch.float32)