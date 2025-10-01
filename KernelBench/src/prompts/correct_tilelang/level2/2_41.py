"""
Problem Name: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.369 runtime_stats={'mean': 0.369, 'std': 0.131, 'min': 0.243, 'max': 0.915, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.16, 'std': 0.0574, 'min': 0.123, 'max': 0.403, 'num_trials': 100}, 'speedup_ratio': 0.434}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _gemm_factory(M: int,
                  K: int,
                  N: int,
                  block_M: int = 128,
                  block_N: int = 128,
                  block_K: int = 32,
                  num_stages: int = 3,
                  dtype: str = "float16",
                  accum_dtype: str = "float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),  # Expect B already transposed (K, N)
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N),
                      T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        # Linear parameters (weight: (out_features, in_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.empty(out_features))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.bn_eps = 1e-5

        # GroupNorm parameters
        self.gn_weight = nn.Parameter(torch.ones(out_features))
        self.gn_bias = nn.Parameter(torch.zeros(out_features))
        self.gn_eps = 1e-5

        # Kernel cache
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            kernel = _gemm_factory(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
                dtype="float16",
                accum_dtype="float",
            )
            self._kernel_cache[key] = kernel
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move inputs & params to CUDA
        x = x.to(device="cuda", dtype=torch.float16)
        weight_t = self.weight.to(device="cuda", dtype=torch.float16).t().contiguous()
        bias = self.bias.to(device="cuda", dtype=torch.float16)

        batch_size = x.shape[0]
        kernel = self._get_kernel(batch_size, x.dtype)

        # GEMM (fp16 I/O, fp32 accum) -> cast to fp32 for following ops
        y = kernel(x, weight_t)

        # Bias add
        y = y + bias

        # BatchNorm (training-style statistics)
        mean_bn = y.mean(dim=0, keepdim=True)
        var_bn = y.var(dim=0, unbiased=False, keepdim=True)
        y = (y - mean_bn) / torch.sqrt(var_bn + self.bn_eps)
        y = y * self.bn_weight + self.bn_bias

        # GELU (exact)
        y = 0.5 * y * (1.0 + torch.erf(y / math.sqrt(2.0)))

        # GroupNorm
        B, C = y.shape
        G = self.num_groups
        y_reshape = y.view(B, G, C // G)
        mean_g = y_reshape.mean(dim=2, keepdim=True)
        var_g = y_reshape.var(dim=2, unbiased=False, keepdim=True)
        y_reshape = (y_reshape - mean_g) / torch.sqrt(var_g + self.gn_eps)
        y = y_reshape.view(B, C) * self.gn_weight + self.gn_bias

        # Mean across features (dim=1), keepdim=True
        y = y.mean(dim=1, keepdim=True)

        # ReLU
        y = torch.clamp(y, min=0.0)

        return y