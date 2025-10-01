"""
Problem Name: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.214 runtime_stats={'mean': 0.214, 'std': 0.0319, 'min': 0.189, 'max': 0.47, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.112, 'std': 0.0252, 'min': 0.0993, 'max': 0.333, 'num_trials': 100}, 'speedup_ratio': 0.523}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------------- #
#                            TileLang kernel factory                             #
# ----------------------------------------------------------------------------- #
def _build_linear_kernel(
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
    TileLang kernel:  Y = X @ W.T + Bias
        X : [M, K]       fp16
        W : [N, K]       fp16   (not transposed → we set transpose_B=True)
        Bias : [N]       fp16
        Y : [M, N]       fp16   (created by TileLang)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), in_dtype),
        W: T.Tensor((N, K), in_dtype),
        Bias: T.Tensor((N,), in_dtype),
        Y: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),          # grid-x
            T.ceildiv(M, block_M),          # grid-y
            threads=threads,                # threads / block
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            Bias_s = T.alloc_shared((block_N,), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Load bias slice once per block
            T.copy(Bias[bx * block_N], Bias_s)

            # Clear accumulator
            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], B_s)
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # Bias add & write-back
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, Bias_s[j])
                    Y[gi, gj] = T.Cast(in_dtype, val)

    return kernel


# ----------------------------------------------------------------------------- #
#                                 PyTorch wrapper                               #
# ----------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Linear (TileLang) ➔ BatchNorm ➔ extra-bias ➔ divide ➔ Swish
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        bias_shape=(1,),
        divide_value: float = 1.0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # ----- Linear parameters (same init as nn.Linear) -----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_lin = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias_lin, -bound, bound)

        # ----- BatchNorm parameters (like nn.BatchNorm1d) -----
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))
        self.bn_eps = float(bn_eps)
        self.bn_momentum = float(bn_momentum)

        # ----- Extra bias, divide constant -----
        self.bias_extra = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = float(divide_value)

        # ----- Kernel cache -----
        self._kernel_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # --------------------------- kernel retrieval --------------------------- #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
            )
        return self._kernel_cache[key]

    # -------------------------------- forward ------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # --- Prepare tensors for TileLang (fp16) ---
        x_fp16 = x.to(device=device, dtype=torch.float16, non_blocking=True)
        w_fp16 = self.weight.to(device=device, dtype=torch.float16)
        b_lin_fp16 = self.bias_lin.to(device=device, dtype=torch.float16)

        B = x_fp16.shape[0]
        kernel = self._get_kernel(B, x_fp16.dtype)

        y_fp16 = kernel(x_fp16, w_fp16, b_lin_fp16)       # GEMM(+bias)

        # ---- BatchNorm (fp32 math) ----
        y = y_fp16.to(torch.float32)
        if self.training:
            batch_mean = y.mean(dim=0)
            batch_var = y.var(dim=0, unbiased=False)

            with torch.no_grad():
                self.running_mean.mul_(1 - self.bn_momentum).add_(self.bn_momentum * batch_mean)
                self.running_var.mul_(1 - self.bn_momentum).add_(self.bn_momentum * batch_var)

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        y = (y - mean) / torch.sqrt(var + self.bn_eps)
        y = y * self.bn_weight.to(device) + self.bn_bias.to(device)

        # ---- Extra bias, divide, Swish ----
        y = y + self.bias_extra.to(device=y.device, dtype=y.dtype)
        y = y / self.divide_value
        y = y * torch.sigmoid(y)

        return y.to(orig_dtype)