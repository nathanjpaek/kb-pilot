"""
Problem Name: 11_Gemm_GroupNorm_Hardtanh
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0971 runtime_stats={'mean': 0.0971, 'std': 0.0043, 'min': 0.0884, 'max': 0.113, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0977, 'std': 0.00534, 'min': 0.09, 'max': 0.116, 'num_trials': 100}, 'speedup_ratio': 1.01}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# GEMM + bias kernel factory
# ---------------------------------------------------------------------------
def _build_gemm_bias_kernel(
    M: int,
    N: int,
    K: int,
    *,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    threads: int = 128,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),         # (batch, in_features)
        W: T.Tensor((N, K), dtype),         # (out_features, in_features), row-major
        B: T.Tensor((N,), dtype),           # bias
        Out: T.Tensor((M, N), dtype),       # created by TileLang
    ):
        grid_x = T.ceildiv(N, block_N)
        grid_y = T.ceildiv(M, block_M)

        with T.Kernel(grid_x, grid_y, threads=threads) as (bx, by):
            # Shared tiles
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)

            # Accumulator fragment
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            # Reduction loop over K
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Tile coords in global memory
                x_row = by * block_M
                x_col = ko * block_K
                w_row = bx * block_N
                w_col = ko * block_K

                # Load tiles
                T.copy(X[x_row, x_col], X_s)
                T.copy(W[w_row, w_col], W_s)

                # GEMM – note: W_s is (block_N, block_K) so transpose_B=True
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            # Add bias and write back
            for li, lj in T.Parallel(block_M, block_N):
                gm = by * block_M + li
                gn = bx * block_N + lj
                if (gm < M) and (gn < N):
                    val = C_loc[li, lj] + T.Cast(accum_dtype, B[gn])
                    Out[gm, gn] = T.Cast(dtype, val)

    return kernel


# ---------------------------------------------------------------------------
# GroupNorm + HardTanh kernel factory
# ---------------------------------------------------------------------------
def _build_groupnorm_hardtanh_kernel(
    M: int,
    N: int,
    num_groups: int,
    *,
    eps: float = 1e-5,
    min_val: float = -2.0,
    max_val: float = 2.0,
    threads: int = 1,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    assert N % num_groups == 0, "num_channels must be divisible by num_groups"
    group_size = N // num_groups
    eps_const = float(eps)
    min_const = float(min_val)
    max_const = float(max_val)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        In:    T.Tensor((M, N), dtype),     # input from GEMM
        Gamma: T.Tensor((N,),  dtype),      # scale (weight)
        Beta:  T.Tensor((N,),  dtype),      # shift (bias)
        Out:   T.Tensor((M, N), dtype),     # output tensor
    ):
        with T.Kernel(M, threads=threads) as bx:
            # Allocate local buffers for statistics
            mean   = T.alloc_local((num_groups,), accum_dtype)
            var    = T.alloc_local((num_groups,), accum_dtype)

            # ------------------------------------------------------------------
            # Compute mean per (row, group)
            # ------------------------------------------------------------------
            for g in T.serial(num_groups):
                acc = T.alloc_local((1,), accum_dtype)
                T.clear(acc)
                for c in T.serial(group_size):
                    idx = g * group_size + c
                    acc[0] += In[bx, idx].astype(accum_dtype)
                mean[g] = acc[0] / group_size

            # ------------------------------------------------------------------
            # Compute variance per (row, group)
            # ------------------------------------------------------------------
            for g in T.serial(num_groups):
                acc = T.alloc_local((1,), accum_dtype)
                T.clear(acc)
                mval = mean[g]
                for c in T.serial(group_size):
                    idx = g * group_size + c
                    diff = In[bx, idx].astype(accum_dtype) - mval
                    acc[0] += diff * diff
                var[g] = acc[0] / group_size

            # ------------------------------------------------------------------
            # Normalize, scale/bias, HardTanh
            # ------------------------------------------------------------------
            for idx in T.serial(N):
                g = idx // group_size
                mval = mean[g]
                vval = var[g]
                inv_std = T.rsqrt(vval + eps_const)
                val = In[bx, idx].astype(accum_dtype)
                val = (val - mval) * inv_std
                val = val * Gamma[idx].astype(accum_dtype) + Beta[idx].astype(accum_dtype)
                val = T.clamp(val, min_const, max_const)
                Out[bx, idx] = T.Cast(dtype, val)

    return kernel


# ---------------------------------------------------------------------------
# PyTorch wrapper module
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Linear → GroupNorm → HardTanh implemented with TileLang kernels.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        hardtanh_min: float,
        hardtanh_max: float,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_groups = int(num_groups)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.eps = 1e-5

        # -------- Linear parameters (identical init to nn.Linear) ------------
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # -------- GroupNorm parameters (gamma/beta) --------------------------
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))

        # ---------------------------------------------------------------------
        self._gemm_cache = {}       # keyed by (M, dtype)
        self._gn_cache = {}         # keyed by (M, dtype)

    # -------------------------------------------------------------------------
    # helpers to fetch / compile kernels
    # -------------------------------------------------------------------------
    def _get_gemm_kernel(self, M: int, dtype: torch.dtype):
        key = (M, dtype)
        if key not in self._gemm_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._gemm_cache[key] = _build_gemm_bias_kernel(
                M, self.out_features, self.in_features, dtype=tl_dtype
            )
        return self._gemm_cache[key]

    def _get_gn_kernel(self, M: int, dtype: torch.dtype):
        key = (M, dtype)
        if key not in self._gn_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._gn_cache[key] = _build_groupnorm_hardtanh_kernel(
                M,
                self.out_features,
                self.num_groups,
                eps=self.eps,
                min_val=self.hardtanh_min,
                max_val=self.hardtanh_max,
                dtype=tl_dtype,
            )
        return self._gn_cache[key]

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Move to CUDA and fp16
        x_f16 = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16, copy=False)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16, copy=False)
        gamma_f16 = self.gamma.to(device="cuda", dtype=torch.float16, copy=False)
        beta_f16 = self.beta.to(device="cuda", dtype=torch.float16, copy=False)

        M = x_f16.shape[0]

        # GEMM + bias
        gemm_kernel = self._get_gemm_kernel(M, x_f16.dtype)
        y_f16 = gemm_kernel(x_f16, w_f16, b_f16)

        # GroupNorm + HardTanh
        gn_kernel = self._get_gn_kernel(M, y_f16.dtype)
        out_f16 = gn_kernel(y_f16, gamma_f16, beta_f16)

        return out_f16.to(orig_dtype)