"""
Problem Name: 39_Gemm_Scale_BatchNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.156 runtime_stats={'mean': 0.156, 'std': 0.0127, 'min': 0.14, 'max': 0.197, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.103, 'std': 0.0503, 'min': 0.0861, 'max': 0.533, 'num_trials': 100}, 'speedup_ratio': 0.66}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------
#  TileLang kernel: GEMM + scale
# --------------------------------------------------------------------
def _gemm_scale_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Fused kernel that computes:
        O = (X @ W^T + bias) * scale
    returned in accum_dtype (FP32) for later BatchNorm.
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((M, K), dtype),       # input
        W: T.Tensor((N, K), dtype),       # weight (row-major)
        B: T.Tensor((N,), dtype),         # bias
        S: T.Tensor((N,), dtype),         # scale
        O: T.Tensor((M, N), accum_dtype)  # output
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], X_s)
                T.copy(W[bx * block_N, ko * block_K], W_s)
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            # bias + scale + write back
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, B[gn])
                    val = val * T.Cast(accum_dtype, S[gn])
                    O[gm, gn] = val

    return main


# --------------------------------------------------------------------
#  PyTorch wrapper module
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised implementation of Linear -> scale -> BatchNorm1d
    using TileLang for the Linear*scale part.
    """

    def __init__(self, in_features: int, out_features: int,
                 scale_shape, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.eps = float(eps)
        self.momentum = float(momentum)

        # ---- Linear parameters (same init as nn.Linear) ----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

        # ---- scale parameter ----
        self.scale = nn.Parameter(torch.randn(scale_shape))

        # ---- BatchNorm affine parameters ----
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))

        # ---- kernel cache ----
        self._cached_kernels = {}

    # ----------------------------------------------------------------
    #  kernel retrieval / compilation
    # ----------------------------------------------------------------
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._cached_kernels:
            ker = _gemm_scale_kernel(
                M=batch_size,
                N=self.out_features,
                K=self.in_features,
            )
            self._cached_kernels[key] = ker
        return self._cached_kernels[key]

    # ----------------------------------------------------------------
    #  forward
    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, in_features)  float32
        returns : (batch, out_features)  float32
        """
        device = torch.device("cuda")
        x_f16 = x.to(device=device, dtype=torch.float16)
        w_f16 = self.weight.to(device=device, dtype=torch.float16)
        b_f16 = self.bias.to(device=device, dtype=torch.float16)
        s_f16 = self.scale.to(device=device, dtype=torch.float16)

        kernel = self._get_kernel(x_f16.shape[0], x_f16.dtype)
        out_fp32 = kernel(x_f16, w_f16, b_f16, s_f16)   # (batch, out_features) float32

        # -------- BatchNorm --------
        if self.training:
            batch_mean = out_fp32.mean(dim=0)
            batch_var = out_fp32.var(dim=0, unbiased=False)

            # Update running stats
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        out_norm = (out_fp32 - mean) / torch.sqrt(var + self.eps)
        out_norm = out_norm * self.bn_weight + self.bn_bias
        return out_norm