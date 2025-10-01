"""
Problem Name: 36_Matmul_LogSumExp_HardSwish_ResidualAdd_Hardtanh
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=17.9 runtime_stats={'mean': 17.9, 'std': 0.0514, 'min': 17.9, 'max': 18.1, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.349, 'std': 0.0845, 'min': 0.162, 'max': 0.685, 'num_trials': 100}, 'speedup_ratio': 0.0195}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# Kernel 1: computes
#   lse_hswish[i] = hardswish(logsumexp_j  (X[i] · Y[j]))
# where  X, Y ∈ ℝ^{B×F}.   Output shape is (B, 1)
# ---------------------------------------------------------------------------


def _build_lse_hswish_kernel(
    B: int,
    F: int,
    *,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((B, F), dtype),
        Y: T.Tensor((B, F), dtype),
        Out: T.Tensor((B, 1), dtype),
    ):
        # One thread-block per row of X
        with T.Kernel(B, threads=threads) as bx:
            tx = T.get_thread_binding(0)

            # ------------------------------------------------------------------
            # per-thread local running (max, sum_exp) using single-pass algorithm
            # ------------------------------------------------------------------
            local_max = T.alloc_local((1,), accum_dtype)
            local_sum = T.alloc_local((1,), accum_dtype)
            local_max[0] = T.Cast(accum_dtype, -3.4e38)   # −∞
            local_sum[0] = T.Cast(accum_dtype, 0.0)

            for jb in T.serial(T.ceildiv(B, threads)):
                j = jb * threads + tx
                if j < B:
                    # dot = ⟨X[i], Y[j]⟩
                    dot_acc = T.alloc_local((1,), accum_dtype)
                    dot_acc[0] = T.Cast(accum_dtype, 0.0)
                    for k in T.serial(F):
                        dot_acc[0] += (
                            T.Cast(accum_dtype, X[bx, k])
                            * T.Cast(accum_dtype, Y[j, k])
                        )

                    v = dot_acc[0]

                    if v > local_max[0]:
                        local_sum[0] = (
                            local_sum[0] * T.exp(local_max[0] - v)
                            + T.Cast(accum_dtype, 1.0)
                        )
                        local_max[0] = v
                    else:
                        local_sum[0] += T.exp(v - local_max[0])

            # ------------------------------------------------------------------
            # share (max, sum) across threads and reduce in thread 0
            # ------------------------------------------------------------------
            sh_max = T.alloc_shared((threads,), accum_dtype)
            sh_sum = T.alloc_shared((threads,), accum_dtype)
            sh_max[tx] = local_max[0]
            sh_sum[tx] = local_sum[0]
            T.tvm_storage_sync("shared")

            if tx == 0:
                g_max = sh_max[0]
                g_sum = sh_sum[0]
                for t in T.serial(1, threads):
                    m = sh_max[t]
                    s = sh_sum[t]
                    if m > g_max:
                        g_sum = g_sum * T.exp(g_max - m) + s
                        g_max = m
                    else:
                        g_sum = g_sum + s * T.exp(m - g_max)

                lse = g_max + T.log(g_sum)             # logsumexp
                # hardswish:   x * relu6(x+3) / 6
                relu6 = T.min(T.max(lse + 3.0, 0.0), 6.0)
                hsw  = lse * relu6 / 6.0
                Out[bx, 0] = T.Cast(dtype, hsw)

    return kernel


# ---------------------------------------------------------------------------
# Kernel 2:  C = hardtanh( Y @ Wᵀ + bias + add_vec_broadcast )
# ---------------------------------------------------------------------------


def _build_gemm_residual_hardtanh_kernel(
    M: int,
    F: int,
    N: int,
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
        Y: T.Tensor((M, F), dtype),
        W: T.Tensor((N, F), dtype),        # row-major
        B: T.Tensor((N,), dtype),
        Add: T.Tensor((M,), dtype),        # broadcast vector (from kernel 1)
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            Y_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            for ko in T.Pipelined(T.ceildiv(F, block_K), num_stages=num_stages):
                T.copy(Y[by * block_M, ko * block_K], Y_s)
                T.copy(W[bx * block_N, ko * block_K], W_s)
                T.gemm(Y_s, W_s, C_loc, transpose_B=True)

            # ───── post-processing & store ────────────────────────────────────
            for i, j in T.Parallel(block_M, block_N):
                g_m = by * block_M + i
                g_n = bx * block_N + j
                if (g_m < M) and (g_n < N):
                    v = C_loc[i, j]
                    v += T.Cast(accum_dtype, B[g_n])
                    v += T.Cast(accum_dtype, Add[g_m])
                    v = T.max(T.min(v, 1.0), -1.0)      # hardtanh clip to [−1,1]
                    Out[g_m, g_n] = T.Cast(dtype, v)

    return kernel


# ---------------------------------------------------------------------------
# PyTorch wrapper module
# ---------------------------------------------------------------------------


class ModelNew(nn.Module):
    """
    TileLang-optimized version of the original model.
    """

    def __init__(self, feature_size: int, hidden_size: int):
        super().__init__()
        self.feature_size = int(feature_size)
        self.hidden_size = int(hidden_size)

        # Linear parameters — identical init as nn.Linear
        self.weight = nn.Parameter(torch.empty(hidden_size, feature_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(feature_size)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # kernel caches
        self._kernel1_cache = {}   # keyed by (B, dtype)
        self._kernel2_cache = {}   # keyed by (B, dtype)

    # ------------------------------------------------------------------
    # helpers to fetch / compile kernels
    # ------------------------------------------------------------------
    def _get_kernel1(self, B: int, dtype: torch.dtype):
        key = (B, dtype)
        if key not in self._kernel1_cache:
            tl_dtype = "float16"
            self._kernel1_cache[key] = _build_lse_hswish_kernel(
                B, self.feature_size, dtype=tl_dtype
            )
        return self._kernel1_cache[key]

    def _get_kernel2(self, B: int, dtype: torch.dtype):
        key = (B, dtype)
        if key not in self._kernel2_cache:
            tl_dtype = "float16"
            self._kernel2_cache[key] = _build_gemm_residual_hardtanh_kernel(
                B,
                self.feature_size,
                self.hidden_size,
                dtype=tl_dtype,
            )
        return self._kernel2_cache[key]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape and x.dim() == 2
        B, F = x.shape
        assert F == self.feature_size, "feature size mismatch"

        # to CUDA & fp16
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        y_f16 = y.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16)

        # ── kernel 1 ──
        kernel1 = self._get_kernel1(B, x_f16.dtype)
        add_vec = kernel1(x_f16, y_f16).view(-1)  # shape (B,)

        # ── kernel 2 ──
        kernel2 = self._get_kernel2(B, x_f16.dtype)
        out_f16 = kernel2(y_f16, w_f16, b_f16, add_vec)

        return out_f16.to(x.dtype)