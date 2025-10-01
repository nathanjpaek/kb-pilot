"""
Problem Name: 94_Clamp_BMM_BatchNorm_GroupNorm_LeakyReLU_Clamp
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.98 runtime_stats={'mean': 3.98, 'std': 0.0428, 'min': 3.92, 'max': 4.12, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.35, 'std': 0.0457, 'min': 1.29, 'max': 1.58, 'num_trials': 100}, 'speedup_ratio': 0.339}}
"""

import torch
from torch import nn
import math

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# TileLang kernel factory for Batched Matrix-Multiply:  C[b] = Y[b] @ X[b]
#    X : (B, K, F)
#    Y : (B, F, K)
#    C : (B, F, F)
# ---------------------------------------------------------------------------

def _build_bmm_kernel(
    B: int,
    F: int,
    K: int,
    *,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    rows_per_block = block_M
    cols_per_block = block_N
    ks_per_block   = block_K

    n_row_tiles = math.ceil(F / rows_per_block)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((B, K, F), dtype),   # (B, K, F)  ─ RHS
        Y:   T.Tensor((B, F, K), dtype),   # (B, F, K)  ─ LHS
        Out: T.Tensor((B, F, F), dtype),   # (B, F, F)  ─ result
    ):
        grid_x = T.ceildiv(F, cols_per_block)     # column tiles for N-dimension
        grid_y = B * T.ceildiv(F, rows_per_block) # batch fused with row tiles

        with T.Kernel(grid_x, grid_y, threads=threads) as (bx, by):
            # --- figure out coordinates ---------------------------------------------------
            batch_idx   = by // n_row_tiles
            row_tile_id = by %  n_row_tiles

            row_base = row_tile_id * rows_per_block
            col_base = bx          * cols_per_block

            # --- shared & local storage ---------------------------------------------------
            Y_s = T.alloc_shared((rows_per_block, ks_per_block), dtype)
            X_s = T.alloc_shared((ks_per_block, cols_per_block), dtype)

            C_loc = T.alloc_fragment((rows_per_block, cols_per_block), accum_dtype)
            T.clear(C_loc)

            # --- main reduction loop ------------------------------------------------------
            for ko in T.Pipelined(T.ceildiv(K, ks_per_block), num_stages=num_stages):
                # load Y-tile  (rows_per_block, ks_per_block)
                T.copy(
                    Y[batch_idx, row_base, ko * ks_per_block],
                    Y_s,
                )
                # load X-tile  (ks_per_block, cols_per_block)
                T.copy(
                    X[batch_idx, ko * ks_per_block, col_base],
                    X_s,
                )
                # GEMM
                T.gemm(Y_s, X_s, C_loc)

            # --- write back ---------------------------------------------------------------
            for i, j in T.Parallel(rows_per_block, cols_per_block):
                global_i = row_base + i
                global_j = col_base + j
                if (global_i < F) and (global_j < F):
                    Out[batch_idx, global_i, global_j] = T.Cast(dtype, C_loc[i, j])

    return kernel


# ---------------------------------------------------------------------------
#   PyTorch wrapper module
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Optimized version of the given architecture using a TileLang BMM kernel.
    Remaining element-wise ops (BatchNorm, GroupNorm, LeakyReLU, Clamp) are
    implemented with tensor arithmetic (they are memory-bound and small).
    """

    # ------------------------------------------------------------------
    def __init__(self, num_features: int, num_groups: int, bmm_dim: int):
        super().__init__()

        self.num_features = int(num_features)    # F
        self.num_groups   = int(num_groups)
        self.bmm_dim      = int(bmm_dim)         # K
        assert self.num_features % self.num_groups == 0, "features must be divisible by groups"

        # BatchNorm parameters (as in nn.BatchNorm1d defaults)
        self.bn_weight = nn.Parameter(torch.ones(num_features))
        self.bn_bias   = nn.Parameter(torch.zeros(num_features))
        self.bn_eps    = 1e-5
        self.bn_momentum = 0.1
        # running stats (buffers, not Parameters)
        self.register_buffer("bn_running_mean", torch.zeros(num_features))
        self.register_buffer("bn_running_var",  torch.ones(num_features))

        # GroupNorm parameters (as in nn.GroupNorm defaults)
        self.gn_weight = nn.Parameter(torch.ones(num_features))
        self.gn_bias   = nn.Parameter(torch.zeros(num_features))
        self.gn_eps    = 1e-5

        # TileLang kernel cache  (keyed by (B, dtype))
        self._kern_cache = {}

    # ------------------------------------------------------------------
    def _get_bmm_kernel(self, B: int, dtype: torch.dtype):
        key = (B, dtype)
        if key not in self._kern_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._kern_cache[key] = _build_bmm_kernel(
                B=B,
                F=self.num_features,
                K=self.bmm_dim,
                dtype=tl_dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, K, F)
            y : (B, F, K)
        Returns:
            (B, F, F)
        """
        assert x.ndim == 3 and y.ndim == 3
        B, K, F_in = x.shape
        F = self.num_features
        assert (K == self.bmm_dim) and (F_in == F) and (y.shape == (B, F, K))

        orig_dtype = x.dtype

        # 1 ▸ clamp inputs & move to GPU (fp16)
        x_f16 = torch.clamp(x, -1.0, 1.0).to(device="cuda", dtype=torch.float16).contiguous()
        y_f16 = y.to(device="cuda", dtype=torch.float16).contiguous()

        # 2 ▸ BMM via TileLang
        bmm_kernel = self._get_bmm_kernel(B, x_f16.dtype)
        bmm_out = bmm_kernel(x_f16, y_f16)      # (B, F, F)  fp16

        # 3 ▸ convert to fp32 for numerics
        h = bmm_out.to(dtype=torch.float32)

        # 4 ▸ BatchNorm (training-style using batch statistics)
        mean = h.mean(dim=(0, 2), keepdim=True)             # (1, F, 1)
        var  = h.var(dim=(0, 2), unbiased=False, keepdim=True)
        inv_std = torch.rsqrt(var + self.bn_eps)
        h_hat = (h - mean) * inv_std
        h = h_hat * self.bn_weight.view(1, F, 1) + self.bn_bias.view(1, F, 1)

        # (optional) update running stats if training
        if self.training:
            momentum = self.bn_momentum
            self.bn_running_mean.mul_(1 - momentum).add_(momentum * mean.squeeze())
            self.bn_running_var.mul_(1 - momentum).add_(momentum * var.squeeze())

        # 5 ▸ GroupNorm
        G = self.num_groups
        group_size = F // G
        h_reshaped = h.view(B, G, group_size, F)            # (B, G, Cg, L)
        g_mean = h_reshaped.mean(dim=(2, 3), keepdim=True)
        g_var  = h_reshaped.var(dim=(2, 3), unbiased=False, keepdim=True)
        g_inv_std = torch.rsqrt(g_var + self.gn_eps)
        h_norm = (h_reshaped - g_mean) * g_inv_std
        h = h_norm.view(B, F, F)
        h = h * self.gn_weight.view(1, F, 1) + self.gn_bias.view(1, F, 1)

        # 6 ▸ LeakyReLU + final clamp
        neg_slope = 0.01
        h = torch.where(h >= 0, h, h * neg_slope)
        h = torch.clamp(h, 0.0, 2.0)

        # 7 ▸ cast back to original dtype and return
        return h.to(orig_dtype)