"""
Problem Name: 88_Gemm_GroupNorm_Swish_Multiply_Swish
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.183 runtime_stats={'mean': 0.183, 'std': 0.00454, 'min': 0.178, 'max': 0.211, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.101, 'std': 0.00352, 'min': 0.0972, 'max': 0.126, 'num_trials': 100}, 'speedup_ratio': 0.552}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                      TileLang GEMM kernel factory                           #
# --------------------------------------------------------------------------- #
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
    """
    FP16 I/O GEMM with FP32 accumulation + bias add
    Shapes:
        X : [M, K]            (fp16)
        W : [N, K]            (fp16)  –  not transposed, we set transpose_B=True
        B : [N]               (fp16)
        Y : [M, N]            (fp16)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gemm_kernel(
        X: T.Tensor((M, K), in_dtype),
        W: T.Tensor((N, K), in_dtype),
        B: T.Tensor((N,), in_dtype),
        Y: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),          # grid-x
            T.ceildiv(M, block_M),          # grid-y
            threads=threads,                # threads / block
        ) as (bx, by):
            # ---------------------------------------------------------------- #
            #                    Shared & register allocations                  #
            # ---------------------------------------------------------------- #
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            Bias_s = T.alloc_shared((block_N,), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Clear accumulator
            T.clear(C_loc)

            # Copy bias slice once per block
            T.copy(B[bx * block_N], Bias_s)

            # Main GEMM loop over K
            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], B_s)
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # Epilogue: bias add + store
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, Bias_s[j])
                    Y[gi, gj] = T.Cast(in_dtype, val)

    return gemm_kernel


# --------------------------------------------------------------------------- #
#                                PyTorch Module                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimised version of the reference model.
    – GEMM is replaced with a TileLang kernel (FP16 I/O, FP32 accum, fused bias)
    – GroupNorm + Swish ×2 and element-wise multiply done with PyTorch tensor ops
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        multiply_weight_shape,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_groups = int(num_groups)
        self.gn_eps = float(eps)

        # ---------------- Linear parameters (same init as nn.Linear) -------- #
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

        # ---------------- GroupNorm affine parameters ----------------------- #
        self.gn_weight = nn.Parameter(torch.ones(out_features))
        self.gn_bias = nn.Parameter(torch.zeros(out_features))

        # ---------------- Element-wise multiply parameter ------------------- #
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

        # ---------------- Kernel cache -------------------------------------- #
        self._kernel_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # --------------------------------------------------------------------- #
    #                          kernel retrieval                              #
    # --------------------------------------------------------------------- #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            kernel = _build_gemm_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
            )
            self._kernel_cache[key] = kernel
        return self._kernel_cache[key]

    # --------------------------------------------------------------------- #
    #                                forward                                #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, in_features]  (any dtype, CPU or CUDA)
        Returns:
            [B, out_features]     (same dtype as input)
        """
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # ------------- Prepare tensors for GEMM kernel (fp16) -------------- #
        x_f16 = x.to(device=device, dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device=device, dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device=device, dtype=torch.float16).contiguous()

        B = x_f16.shape[0]

        # -------------------- Launch / compile GEMM kernel ------------------ #
        kernel = self._get_kernel(B, x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, b_f16)          # [B, out_features] fp16

        # -------------------------- GroupNorm -------------------------------- #
        G = self.num_groups
        C = self.out_features
        y_fp32 = y_f16.to(dtype=torch.float32)       # promote for arithmetic
        y_g = y_fp32.view(B, G, C // G)

        mean = y_g.mean(dim=2, keepdim=True)
        var = y_g.var(dim=2, unbiased=False, keepdim=True)
        y_norm = (y_g - mean) / torch.sqrt(var + self.gn_eps)
        y_norm = y_norm.view(B, C)

        y_norm = (
            y_norm
            * self.gn_weight.to(device=device, dtype=torch.float32)
            + self.gn_bias.to(device=device, dtype=torch.float32)
        )

        # ---------------- Swish → *weight → Swish --------------------------- #
        y_swish1 = y_norm * torch.sigmoid(y_norm)
        y_mul = y_swish1 * self.multiply_weight.to(device=device, dtype=torch.float32)
        y_swish2 = y_mul * torch.sigmoid(y_mul)

        out = y_swish2.to(dtype=orig_dtype)

        return out