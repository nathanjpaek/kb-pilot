"""
Problem Name: 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0822 runtime_stats={'mean': 0.0822, 'std': 0.0132, 'min': 0.0743, 'max': 0.196, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.162, 'std': 0.0307, 'min': 0.14, 'max': 0.439, 'num_trials': 100}, 'speedup_ratio': 1.97}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_linear_scale_clamp_kernel(
    M: int,
    K: int,
    N: int,
    scale_val: float,
    clamp_min: float,
    clamp_max: float,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Y = clamp( ( (X @ Wᵀ) + B ) * scale_val , clamp_min , clamp_max )
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),  # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            # Shared / fragments
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_N, block_K), dtype)
            Bias_s = T.alloc_shared((block_N,), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Load bias slice once
            T.copy(B[bx * block_N:(bx + 1) * block_N], Bias_s)

            # Clear accumulator
            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                # Load tiles
                T.copy(
                    X[by * block_M : (by + 1) * block_M,
                      ko * block_K : (ko + 1) * block_K],
                    A_s,
                )
                T.copy(
                    W[bx * block_N : (bx + 1) * block_N,
                      ko * block_K : (ko + 1) * block_K],
                    B_s,
                )
                # GEMM  (B is transposed inside gemm)
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # Scale, clamp, store
            for i, j in T.Parallel(block_M, block_N):
                g_row = by * block_M + i
                g_col = bx * block_N + j
                if (g_row < M) and (g_col < N):
                    v = C_loc[i, j] + Bias_s[j].astype(accum_dtype)
                    v = v * scale_val
                    v = T.min(
                        T.Cast(accum_dtype, clamp_max),
                        T.max(T.Cast(accum_dtype, clamp_min), v),
                    )
                    Y[g_row, g_col] = T.Cast(dtype, v)

    return kernel


def _build_logsumexp_mish_kernel(
    M: int,
    N: int,
    dtype_in: str = "float16",
    accum_dtype: str = "float32",
):
    """
    For each row r:
        z = LogSumExp(Y[r, :])
        out = z * mish(z)   (mish(z) = z * tanh(softplus(z)))
                            -> out = z^2 * tanh(softplus(z))
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        Y: T.Tensor((M, N), dtype_in),
        Out: T.Tensor((M, 1), dtype_in),
    ):
        one = T.Cast(accum_dtype, 1.0)

        with T.Kernel(M, threads=1) as bx:
            # Compute max
            max_val = T.alloc_local((1,), accum_dtype)
            max_val[0] = Y[bx, 0].astype(accum_dtype)
            for j in range(1, N):
                max_val[0] = T.max(max_val[0], Y[bx, j].astype(accum_dtype))

            # Compute sum exp
            sum_exp = T.alloc_local((1,), accum_dtype)
            sum_exp[0] = T.Cast(accum_dtype, 0)
            for j in range(N):
                sum_exp[0] += T.exp(
                    Y[bx, j].astype(accum_dtype) - max_val[0]
                )

            z = max_val[0] + T.log(sum_exp[0])

            # Mish pieces
            softplus = T.log(one + T.exp(z))
            mish = z * T.tanh(softplus)
            out_val = z * mish  # z^2 * tanh(softplus(z))

            Out[bx, 0] = T.Cast(dtype_in, out_val)

    return kernel


class ModelNew(nn.Module):
    """
    Optimized TileLang implementation of:
        Linear → scale → self-residual → clamp → LogSumExp → z * mish(z)
    Final output shape: (batch_size, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        scale_factor: float,
        clamp_min: float,
        clamp_max: float,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

        # Combined scalar (scale * 2 from the x + x step)
        self.scale_val = float(scale_factor) * 2.0
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # ----- Parameters identical to nn.Linear defaults -----
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(input_size)
            nn.init.uniform_(self.bias, -bound, bound)

        # Kernel caches
        self._k1_cache = {}  # linear+scale+clamp
        self._k2_cache = {}  # logsumexp+mish

    # ---------- Kernel getters ----------
    def _get_k1(self, batch: int, dtype: str = "float16"):
        key = (batch, dtype)
        if key not in self._k1_cache:
            self._k1_cache[key] = _build_linear_scale_clamp_kernel(
                batch,
                self.input_size,
                self.hidden_size,
                self.scale_val,
                self.clamp_min,
                self.clamp_max,
                dtype=dtype,
            )
        return self._k1_cache[key]

    def _get_k2(self, batch: int, dtype: str = "float16"):
        key = (batch, dtype)
        if key not in self._k2_cache:
            self._k2_cache[key] = _build_logsumexp_mish_kernel(
                batch, self.hidden_size, dtype_in=dtype
            )
        return self._k2_cache[key]

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Prepare tensors
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        B = x_fp16.shape[0]

        # Kernel-1 : Linear + scaling*2 + clamp
        k1 = self._get_k1(B, "float16")
        y_fp16 = k1(x_fp16, w_fp16, b_fp16)  # (B, hidden)

        # Kernel-2 : LogSumExp row-wise & z * mish(z)
        k2 = self._get_k2(B, "float16")
        out_fp16 = k2(y_fp16)  # (B,1)

        return out_fp16.to(orig_dtype)