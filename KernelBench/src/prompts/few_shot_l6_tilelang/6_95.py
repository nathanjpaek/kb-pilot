"""
Problem Name: 95_Gemm_Add_GlobalAvgPool_AvgPool_Divide
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.2 runtime_stats={'mean': 1.2, 'std': 0.0602, 'min': 1.06, 'max': 1.31, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.282, 'std': 0.082, 'min': 0.121, 'max': 0.385, 'num_trials': 100}, 'speedup_ratio': 0.235}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# -----------------------------------------------------------------------------
# GEMM + Bias + Add-Y ----------------------------------------------------------
# -----------------------------------------------------------------------------
def _build_gemm_add_kernel(
    M: int,
    K: int,
    N: int,
    *,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)  # last arg is output
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),      # input
        W: T.Tensor((N, K), dtype),      # weight (row-major)
        B: T.Tensor((N,),   dtype),      # bias
        Y: T.Tensor((M, N), dtype),      # tensor to be added
        Out: T.Tensor((M, N), dtype),    # generated output
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

            # Reduction over K
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], X_s)
                T.copy(W[bx * block_N, ko * block_K], W_s)
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            # Epilogue :  +bias  +Y
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    v = C_loc[i, j] + T.Cast(accum_dtype, B[gn])
                    v += T.Cast(accum_dtype, Y[gm, gn])
                    Out[gm, gn] = T.Cast(dtype, v)

    return kernel


# -----------------------------------------------------------------------------
# AvgPool2d (kernel k) + global sum (for mean) --------------------------------
# -----------------------------------------------------------------------------
def _build_avgpool_sum_kernel(
    B: int,
    S: int,                 # input spatial size (height == width)
    k: int,                 # pool kernel_size (= stride)
    dtype: str = "float16",
    accum_dtype: str = "float",
    block_size: int = 256,
):
    S_out = S // k
    TOT   = B * S_out * S_out

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((B, S, S), dtype),          # input feature map
        Sum: T.Tensor((1,), accum_dtype),       # running sum (atomic)
        Out: T.Tensor((B, S_out, S_out), dtype) # pooled output
    ):
        grid = T.ceildiv(TOT, block_size)

        with T.Kernel(grid, threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx          # linear index into Out

            if idx < TOT:
                b      = idx // (S_out * S_out)
                rem    = idx %  (S_out * S_out)
                oh     = rem // S_out
                ow     = rem %  S_out

                acc = T.Cast(accum_dtype, 0.0)
                for dy in range(k):
                    for dx in range(k):
                        ih = oh * k + dy
                        iw = ow * k + dx
                        acc += T.Cast(accum_dtype, X[b, ih, iw])

                avg_val = acc / (k * k)
                Out[b, oh, ow] = T.Cast(dtype, avg_val)
                T.atomic_add(Sum[0], avg_val)

    return kernel, S_out


# -----------------------------------------------------------------------------
# Divide by scalar -------------------------------------------------------------
# -----------------------------------------------------------------------------
def _build_divide_kernel(
    B: int,
    S_out: int,
    dtype: str = "float16",
    block_size: int = 256,
):
    TOT = B * S_out * S_out

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:      T.Tensor((B, S_out, S_out), dtype),
        Mean_t: T.Tensor((1,), "float32"),
        Out:    T.Tensor((B, S_out, S_out), dtype),
    ):
        grid = T.ceildiv(TOT, block_size)

        with T.Kernel(grid, threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOT:
                b   = idx // (S_out * S_out)
                rem = idx %  (S_out * S_out)
                i   = rem // S_out
                j   = rem %  S_out
                Out[b, i, j] = X[b, i, j] / T.Cast(dtype, Mean_t[0])

    return kernel


# -----------------------------------------------------------------------------
# PyTorch wrapper --------------------------------------------------------------
# -----------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    TileLang-accelerated replacement for original Model.
    Performs Linear → Add → reshape → AvgPool2d → divide by global mean.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        super().__init__()
        self.in_features  = int(in_features)
        self.out_features = int(out_features)
        self.kernel_size  = int(kernel_size)

        # ─── ensure out_features is perfect square ────────────────────────────
        self.S = int(math.isqrt(self.out_features))
        assert self.S * self.S == self.out_features, "out_features must be a square number"

        # ─── parameters (match nn.Linear defaults) ────────────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # ─── kernel caches ────────────────────────────────────────────────────
        self._gemm_cache  = {}   # keyed by (batch, dtype)
        self._pool_cache  = {}   # keyed by (batch, dtype)
        self._div_cache   = {}   # keyed by (batch, dtype)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _get_gemm_kernel(self, B: int, tl_dtype: str):
        key = (B, tl_dtype)
        if key not in self._gemm_cache:
            self._gemm_cache[key] = _build_gemm_add_kernel(
                M=B,
                K=self.in_features,
                N=self.out_features,
                dtype=tl_dtype,
            )
        return self._gemm_cache[key]

    def _get_pool_kernel(self, B: int, tl_dtype: str):
        key = (B, tl_dtype)
        if key not in self._pool_cache:
            kernel, S_out = _build_avgpool_sum_kernel(
                B=B,
                S=self.S,
                k=self.kernel_size,
                dtype=tl_dtype,
            )
            self._pool_cache[key] = (kernel, S_out)
        return self._pool_cache[key]

    def _get_div_kernel(self, B: int, S_out: int, tl_dtype: str):
        key = (B, S_out, tl_dtype)
        if key not in self._div_cache:
            self._div_cache[key] = _build_divide_kernel(
                B=B,
                S_out=S_out,
                dtype=tl_dtype,
            )
        return self._div_cache[key]

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_features)
            y : (B, out_features)
        Returns:
            (B, 1, S_out, S_out) tensor
        """
        B = x.shape[0]
        orig_dtype = x.dtype

        # Move to GPU / fp16 for computation
        x_f16 = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        y_f16 = y.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16, copy=False)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16, copy=False)

        # ─── GEMM + Add-Y ────────────────────────────────────────────────────
        gemm_kernel = self._get_gemm_kernel(B, "float16")
        out_lin = gemm_kernel(x_f16, w_f16, b_f16, y_f16)  # (B, out_features)

        # reshape to (B, S, S)
        feat_map = out_lin.view(B, self.S, self.S).contiguous()

        # ─── AvgPool + global sum ────────────────────────────────────────────
        pool_kernel, S_out = self._get_pool_kernel(B, "float16")
        sum_buf = torch.zeros(1, dtype=torch.float32, device="cuda")
        pooled   = pool_kernel(feat_map, sum_buf)          # (B, S_out, S_out)

        # compute mean on host
        total_elems = B * S_out * S_out
        mean_val = (sum_buf.item() / total_elems) if total_elems > 0 else 0.0
        mean_t   = torch.tensor([mean_val], dtype=torch.float32, device="cuda")

        # ─── Division by mean ───────────────────────────────────────────────
        div_kernel = self._get_div_kernel(B, S_out, "float16")
        normed = div_kernel(pooled, mean_t)                 # (B, S_out, S_out)

        # unsqueeze channel dim & cast back
        normed = normed.unsqueeze(1).to(orig_dtype)
        return normed