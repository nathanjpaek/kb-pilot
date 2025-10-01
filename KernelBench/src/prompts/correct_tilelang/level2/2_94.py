"""
Problem Name: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.162 runtime_stats={'mean': 0.162, 'std': 0.027, 'min': 0.143, 'max': 0.382, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.109, 'std': 0.029, 'min': 0.0852, 'max': 0.351, 'num_trials': 100}, 'speedup_ratio': 0.673}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                           TileLang kernel factory                           #
# --------------------------------------------------------------------------- #
def _make_gemm_bias_kernel(
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
    """
    Compute  Y = X @ W.T + BiasLin + BiasExtra
        X : [M, K]  (fp16)
        W : [N, K]  (fp16)   -- row-major weight
        BiasLin   : [N]  (fp16)   (from Linear)
        BiasExtra : [N]  (fp16)   (external)
        Y : [M, N] (fp16)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gemm_bias_kernel(
        X: T.Tensor((M, K), in_dtype),
        W: T.Tensor((N, K), in_dtype),
        B_lin: T.Tensor((N,), in_dtype),
        B_ext: T.Tensor((N,), in_dtype),
        Y: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),          # grid-x
            T.ceildiv(M, block_M),          # grid-y
            threads=threads,
        ) as (bx, by):
            # ---------------- Shared / Fragment allocations ---------------- #
            X_s = T.alloc_shared((block_M, block_K), in_dtype)
            W_s = T.alloc_shared((block_N, block_K), in_dtype)
            Bl_s = T.alloc_shared((block_N,), in_dtype)
            Be_s = T.alloc_shared((block_N,), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Copy bias slices once per block
            T.copy(B_lin[bx * block_N:(bx + 1) * block_N], Bl_s)
            T.copy(B_ext[bx * block_N:(bx + 1) * block_N], Be_s)

            # Clear accumulator
            T.clear(C_loc)

            # K-loop with pipeline
            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(
                    X[by * block_M:(by + 1) * block_M,
                      ko * block_K:(ko + 1) * block_K],
                    X_s,
                )
                T.copy(
                    W[bx * block_N:(bx + 1) * block_N,
                      ko * block_K:(ko + 1) * block_K],
                    W_s,
                )
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            # Epilogue: add both biases & store
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    val = (
                        C_loc[i, j]
                        + T.Cast(accum_dtype, Bl_s[j])
                        + T.Cast(accum_dtype, Be_s[j])
                    )
                    Y[gm, gn] = T.Cast(in_dtype, val)

    return gemm_bias_kernel


# --------------------------------------------------------------------------- #
#                                PyTorch Module                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimised version:
        Linear (GEMM + two biases) → Hardtanh → Mish → GroupNorm
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_shape,
        num_groups: int,
        gn_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_groups = int(num_groups)
        self.gn_eps = float(gn_eps)

        # -------- Linear parameters (identical init to nn.Linear) --------
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_linear = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias_linear, -bound, bound)

        # -------- Extra bias (after Linear) --------
        self.bias_extra = nn.Parameter(torch.randn(bias_shape))

        # -------- GroupNorm affine params --------
        self.gn_weight = nn.Parameter(torch.ones(out_features))
        self.gn_bias = nn.Parameter(torch.zeros(out_features))

        # -------- Kernel cache keyed by (batch_size, dtype) --------
        self._kernel_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # -------------------------------- Kernel getter ------------------------------- #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _make_gemm_bias_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
            )
        return self._kernel_cache[key]

    # --------------------------------- Forward ----------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # ----- Prepare tensors in fp16 on CUDA -----
        x_f16 = x.to(device=device, dtype=torch.float16, non_blocking=True)
        w_f16 = self.weight.to(device=device, dtype=torch.float16, non_blocking=True)
        bl_f16 = self.bias_linear.to(device=device, dtype=torch.float16, non_blocking=True)
        be_f16 = self.bias_extra.to(device=device, dtype=torch.float16, non_blocking=True)

        B = x_f16.shape[0]
        kernel = self._get_kernel(B, x_f16.dtype)

        # ----- GEMM + biases kernel -----
        y_f16 = kernel(x_f16, w_f16, bl_f16, be_f16)

        # ----- Hardtanh (clamp) -----
        y_f16 = torch.clamp(y_f16, min=-1.0, max=1.0)

        # ----- Mish activation: x * tanh(softplus(x)) -----
        y_f16 = y_f16 * torch.tanh(torch.nn.functional.softplus(y_f16))

        # ----- GroupNorm (manual, fp32 for stats) -----
        B_size, C = y_f16.shape
        G = self.num_groups
        y_f32 = y_f16.to(torch.float32).view(B_size, G, C // G)

        mean = y_f32.mean(dim=2, keepdim=True)
        var = y_f32.var(dim=2, unbiased=False, keepdim=True)
        y_norm = (y_f32 - mean) / torch.sqrt(var + self.gn_eps)

        y_norm = y_norm.view(B_size, C).to(dtype=y_f16.dtype)
        y_out = y_norm * self.gn_weight.to(device=device, dtype=y_f16.dtype) + \
                self.gn_bias.to(device=device, dtype=y_f16.dtype)

        return y_out.to(orig_dtype)