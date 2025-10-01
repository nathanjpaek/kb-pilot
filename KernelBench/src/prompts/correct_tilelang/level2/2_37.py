"""
Problem Name: 37_Matmul_Swish_Sum_GroupNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.14 runtime_stats={'mean': 0.14, 'std': 0.0124, 'min': 0.123, 'max': 0.196, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0944, 'std': 0.00987, 'min': 0.0831, 'max': 0.13, 'num_trials': 100}, 'speedup_ratio': 0.674}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------------- #
#                           TileLang kernel factory                             #
# ----------------------------------------------------------------------------- #
def _build_fused_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Kernel:   Y = Swish( X @ W.T + bias_linear ) + bias_extra
    Shapes:   X  [M, K]         (fp16)
              W  [N, K]         (fp16)   –  not transposed, we pass transpose_B=True
              bias_linear [N]   (fp16)
              bias_extra  [N]   (fp16)
              Y  [M, N]         (fp16)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused(
        X: T.Tensor((M, K), in_dtype),
        W: T.Tensor((N, K), in_dtype),
        BiasLin: T.Tensor((N,), in_dtype),
        BiasEx: T.Tensor((N,), in_dtype),
        Out: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),          # grid-x
            T.ceildiv(M, block_M),          # grid-y
            threads=threads,                # 128 threads / block
        ) as (bx, by):
            # ---------------------------------------------------------------- #
            #                     Shared / register buffers                    #
            # ---------------------------------------------------------------- #
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            BiasLin_s = T.alloc_shared((block_N,), in_dtype)
            BiasEx_s = T.alloc_shared((block_N,), in_dtype)

            # Clear accumulator
            T.clear(C_loc)

            # Copy bias slices once per block
            T.copy(BiasLin[bx * block_N], BiasLin_s)
            T.copy(BiasEx[bx * block_N], BiasEx_s)

            # ---------------------------------------------------------------- #
            #                          GEMM main loop                           #
            # ---------------------------------------------------------------- #
            tiles_K = T.ceildiv(K, block_K)
            for ko in T.Pipelined(tiles_K, num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], B_s)
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # ---------------------------------------------------------------- #
            #             Fused Swish + extra-bias  & write-back               #
            # ---------------------------------------------------------------- #
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, BiasLin_s[j])
                    val = val * T.sigmoid(val)               # Swish
                    val = val + T.Cast(accum_dtype, BiasEx_s[j])
                    Out[gi, gj] = T.Cast(in_dtype, val)

    return fused


# ----------------------------------------------------------------------------- #
#                                 PyTorch wrapper                               #
# ----------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    MatMul + Swish + extra bias fused with TileLang, followed by GroupNorm
    (implemented with PyTorch ops for numerical ease).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        bias_shape,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.gn_eps = eps

        # ----- Linear parameters (same init as nn.Linear) -----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_lin = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias_lin, -bound, bound)

        # ----- Extra bias added after Swish -----
        self.bias_extra = nn.Parameter(torch.randn(bias_shape))

        # ----- GroupNorm affine parameters -----
        self.gn_weight = nn.Parameter(torch.ones(out_features))
        self.gn_bias = nn.Parameter(torch.zeros(out_features))

        # ----- Kernel cache -----
        self._kernel_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # --------------------------------------------------------------------- #
    #                           kernel retrieval                             #
    # --------------------------------------------------------------------- #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            kernel = _build_fused_kernel(
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
            x : [B, in_features]  (fp32)
        Returns:
            [B, out_features]     (same dtype as input)
        """
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # ---- Prepare tensors for kernel (fp16) ----
        x_f16 = x.to(device=device, dtype=torch.float16)
        w_f16 = self.weight.to(device=device, dtype=torch.float16)
        b_lin_f16 = self.bias_lin.to(device=device, dtype=torch.float16)
        b_ex_f16 = self.bias_extra.to(device=device, dtype=torch.float16)

        B = x_f16.shape[0]

        # ---- Launch / compile fused kernel ----
        kernel = self._get_kernel(B, x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, b_lin_f16, b_ex_f16)

        # ---- GroupNorm ----
        G = self.num_groups
        C = self.out_features
        y = y_f16.view(B, G, C // G).to(dtype=torch.float32)

        mean = y.mean(dim=2, keepdim=True)
        var = y.var(dim=2, unbiased=False, keepdim=True)
        y_norm = (y - mean) / torch.sqrt(var + self.gn_eps)

        y_norm = y_norm.view(B, C).to(dtype=y_f16.dtype)
        y_out = y_norm * self.gn_weight.to(device=device, dtype=y_f16.dtype) + \
                self.gn_bias.to(device=device, dtype=y_f16.dtype)

        return y_out.to(dtype=orig_dtype)