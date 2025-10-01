"""
Problem Name: 33_Gemm_Scale_BatchNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.175 runtime_stats={'mean': 0.175, 'std': 0.0149, 'min': 0.158, 'max': 0.214, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0982, 'std': 0.0319, 'min': 0.0855, 'max': 0.393, 'num_trials': 100}, 'speedup_ratio': 0.561}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------
#  TileLang fused kernel: GEMM + Scale
# ---------------------------------------------------------------

def _build_gemm_scale_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    threads: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """Returns a TileLang kernel performing
         O = (X @ W^T + bias) * scale  (fp16 IO, fp32 accum)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), in_dtype),       # input activations
        W: T.Tensor((N, K), in_dtype),       # weight (row-major)
        B: T.Tensor((N,), in_dtype),         # bias
        S: T.Tensor((N,), in_dtype),         # scale vector
        O: T.Tensor((M, N), accum_dtype),    # output (fp32 for BatchNorm)
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            X_s = T.alloc_shared((block_M, block_K), in_dtype)
            W_s = T.alloc_shared((block_N, block_K), in_dtype)
            Acc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(Acc)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], X_s)
                T.copy(W[bx * block_N, ko * block_K], W_s)
                T.gemm(X_s, W_s, Acc, transpose_B=True)

            # epilogue: bias + scale
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    v = Acc[i, j] + T.Cast(accum_dtype, B[gn])
                    v = v * T.Cast(accum_dtype, S[gn])
                    O[gm, gn] = v

    return kernel


# ---------------------------------------------------------------
#  PyTorch wrapper
# ---------------------------------------------------------------

class ModelNew(nn.Module):
    """Optimized Linear → scale → BatchNorm module using TileLang."""

    def __init__(self, in_features: int, out_features: int, scale_shape,
                 eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.eps = float(eps)
        self.momentum = float(momentum)

        # ---- Linear parameters (identical init to nn.Linear) ----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

        # ---- scale parameter ----
        self.scale = nn.Parameter(torch.randn(scale_shape))

        # ---- BatchNorm parameters ----
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))

        # ---- kernel cache ----
        self._kernel_cache = {}

    # -----------------------------------------------------------
    #  kernel retrieval / compilation
    # -----------------------------------------------------------
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_gemm_scale_kernel(
                M=batch_size,
                N=self.out_features,
                K=self.in_features,
            )
        return self._kernel_cache[key]

    # -----------------------------------------------------------
    #  forward
    # -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = torch.device("cuda")

        # Prepare tensors for kernel (fp16 for bandwidth/TC)
        x_f16 = x.to(device=device, dtype=torch.float16, non_blocking=True)
        w_f16 = self.weight.to(device=device, dtype=torch.float16)
        b_f16 = self.bias.to(device=device, dtype=torch.float16)
        s_f16 = self.scale.to(device=device, dtype=torch.float16)

        kernel = self._get_kernel(x_f16.shape[0], x_f16.dtype)
        out_fp32 = kernel(x_f16, w_f16, b_f16, s_f16)  # (batch, out_features) fp32

        # ---------------- BatchNorm ----------------
        if self.training:
            batch_mean = out_fp32.mean(dim=0)
            batch_var = out_fp32.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        out_norm = (out_fp32 - mean) / torch.sqrt(var + self.eps)
        out_norm = out_norm * self.bn_weight.to(device) + self.bn_bias.to(device)
        return out_norm