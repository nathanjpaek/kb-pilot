"""
Problem Name: 62_Matmul_GroupNorm_LeakyReLU_Sum
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0677 runtime_stats={'mean': 0.0677, 'std': 0.0125, 'min': 0.0609, 'max': 0.169, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0798, 'std': 0.0219, 'min': 0.0693, 'max': 0.271, 'num_trials': 100}, 'speedup_ratio': 1.18}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                           Kernel 1 : GEMM                                  #
# --------------------------------------------------------------------------- #

def _build_linear_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """Y = X @ Wᵀ + Bias"""

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), in_dtype),
        W: T.Tensor((N, K), in_dtype),
        Bias: T.Tensor((N,), in_dtype),
        Y: T.Tensor((M, N), in_dtype),  # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            Bias_s = T.alloc_shared((block_N,), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # copy bias slice once per block
            T.copy(Bias[bx * block_N : (bx + 1) * block_N], Bias_s)
            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
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
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # writeback with bias add
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    val = C_loc[i, j] + Bias_s[j].astype(accum_dtype)
                    Y[gi, gj] = T.Cast(in_dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
#            Kernel 2 : GroupNorm + LeakyReLU + multiply by 2                #
# --------------------------------------------------------------------------- #

def _build_groupnorm_kernel(
    M: int,
    C: int,
    num_groups: int,
    eps_val: float,
    neg_slope: float,
    dtype_in: str = "float16",
    accum_dtype: str = "float32",
):
    group_size = C // num_groups

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        Y: T.Tensor((M, C), dtype_in),
        Gamma: T.Tensor((C,), dtype_in),
        Beta: T.Tensor((C,), dtype_in),
        Out: T.Tensor((M, C), dtype_in),
    ):
        eps_c = T.Cast(accum_dtype, eps_val)
        nslope_c = T.Cast(accum_dtype, neg_slope)
        two_c = T.Cast(accum_dtype, 2.0)

        with T.Kernel(M, threads=1) as bx:
            for g in range(num_groups):
                mean = T.alloc_local((1,), accum_dtype)
                var = T.alloc_local((1,), accum_dtype)
                mean[0] = T.Cast(accum_dtype, 0)
                for c in range(group_size):
                    idx = g * group_size + c
                    mean[0] += Y[bx, idx].astype(accum_dtype)
                mean[0] = mean[0] / group_size

                var[0] = T.Cast(accum_dtype, 0)
                for c in range(group_size):
                    idx = g * group_size + c
                    diff = Y[bx, idx].astype(accum_dtype) - mean[0]
                    var[0] += diff * diff
                var[0] = var[0] / group_size
                inv_std = T.rsqrt(var[0] + eps_c)

                for c in range(group_size):
                    idx = g * group_size + c
                    v = Y[bx, idx].astype(accum_dtype)
                    norm = (v - mean[0]) * inv_std
                    affine = norm * Gamma[idx].astype(accum_dtype) + Beta[idx].astype(accum_dtype)
                    lr = T.Select(affine > T.Cast(accum_dtype, 0.0), affine, affine * nslope_c)
                    final = lr * two_c
                    Out[bx, idx] = T.Cast(dtype_in, final)

    return kernel


# --------------------------------------------------------------------------- #
#                              PyTorch wrapper                                #
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """Linear → GroupNorm → LeakyReLU → self-add (×2) implemented with TileLang"""

    def __init__(self, input_size: int, hidden_size: int, num_groups: int, eps: float = 1e-5, negative_slope: float = 0.01):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.eps = float(eps)
        self.negative_slope = float(negative_slope)

        # ----- Linear parameters -----
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(input_size)
        nn.init.uniform_(self.bias, -bound, bound)

        # ----- GroupNorm affine -----
        self.gn_weight = nn.Parameter(torch.ones(hidden_size))
        self.gn_bias = nn.Parameter(torch.zeros(hidden_size))

        # Kernel caches
        self._k_linear: Dict[Tuple[int, torch.dtype], callable] = {}
        self._k_gn: Dict[Tuple[int, torch.dtype], callable] = {}

    # ---------------- kernel getters -----------------
    def _get_linear_kernel(self, batch: int, dtype: torch.dtype):
        key = (batch, dtype)
        if key not in self._k_linear:
            self._k_linear[key] = _build_linear_kernel(
                M=batch,
                K=self.input_size,
                N=self.hidden_size,
            )
        return self._k_linear[key]

    def _get_gn_kernel(self, batch: int, dtype: torch.dtype):
        key = (batch, dtype)
        if key not in self._k_gn:
            self._k_gn[key] = _build_groupnorm_kernel(
                M=batch,
                C=self.hidden_size,
                num_groups=self.num_groups,
                eps_val=self.eps,
                neg_slope=self.negative_slope,
            )
        return self._k_gn[key]

    # -------------------- forward --------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = torch.device("cuda")

        x_f16 = x.to(device=device, dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device=device, dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device=device, dtype=torch.float16).contiguous()
        gamma_f16 = self.gn_weight.to(device=device, dtype=torch.float16).contiguous()
        beta_f16 = self.gn_bias.to(device=device, dtype=torch.float16).contiguous()

        B = x_f16.shape[0]

        # Kernel-1 : Linear
        k1 = self._get_linear_kernel(B, x_f16.dtype)
        y_f16 = k1(x_f16, w_f16, b_f16)

        # Kernel-2 : GroupNorm + LeakyReLU ×2
        k2 = self._get_gn_kernel(B, y_f16.dtype)
        out_f16 = k2(y_f16, gamma_f16, beta_f16)

        return out_f16.to(orig_dtype)