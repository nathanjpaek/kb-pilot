"""
Problem Name: 98_Matmul_AvgPool_GELU_Scale_Max
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0703 runtime_stats={'mean': 0.0703, 'std': 0.00781, 'min': 0.0627, 'max': 0.101, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.121, 'std': 0.0813, 'min': 0.0724, 'max': 0.497, 'num_trials': 100}, 'speedup_ratio': 1.72}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_kernel(
    batch_size: int,
    in_features: int,
    out_pooled: int,
    pool_kernel: int,
    scale_factor: float,
    block_M: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
):
    block_N = out_pooled  # single-block along feature dim

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, in_features), "float16"),
        Wt: T.Tensor((in_features, out_pooled), "float16"),
        Bp: T.Tensor((out_pooled,), "float16"),
        Out: T.Tensor((batch_size,), "float"),
    ):
        with T.Kernel(1, T.ceildiv(batch_size, block_M), threads=128) as (bx, by):
            A_sh = T.alloc_shared((block_M, block_K), "float16")
            W_sh = T.alloc_shared((block_K, block_N), "float16")
            B_sh = T.alloc_shared((block_N,), "float16")

            Acc = T.alloc_fragment((block_M, block_N), "float")
            RowMax = T.alloc_fragment((block_M,), "float")

            if by == 0:
                T.copy(Bp, B_sh)

            T.clear(Acc)
            k_tiles = T.ceildiv(in_features, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], A_sh)
                T.copy(Wt[ko * block_K, 0], W_sh)
                T.gemm(A_sh, W_sh, Acc)

            # fused bias, GELU, scale
            for i, j in T.Parallel(block_M, block_N):
                v = Acc[i, j] + T.Cast("float", B_sh[j])

                v_cub = v * v
                v_cub = v_cub * v  # v^3
                u = v + v_cub * 0.044715
                gelu_val = 0.5 * v * (1.0 + T.tanh(0.7978845608028654 * u))
                Acc[i, j] = gelu_val * scale_factor

            # max over feature dimension
            T.reduce_max(Acc, RowMax, dim=1)

            # write back with bounds check
            for i in T.Parallel(block_M):
                idx = by * block_M + i
                if idx < batch_size:
                    Out[idx] = RowMax[i]

    return kernel


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int):
        key = batch_size
        if key not in self._kernel_cache:
            out_pooled = self.out_features // self.pool_kernel_size
            self._kernel_cache[key] = _build_kernel(
                batch_size,
                self.in_features,
                out_pooled,
                self.pool_kernel_size,
                self.scale_factor,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float16, device="cuda", non_blocking=True)
        batch_size = x.size(0)

        out_pooled = self.out_features // self.pool_kernel_size

        w = (
            self.weight.to(device=x.device, dtype=torch.float16)
            .view(out_pooled, self.pool_kernel_size, self.in_features)
            .mean(dim=1)
        )  # (out_pooled, K)

        b = (
            self.bias.to(device=x.device, dtype=torch.float16)
            .view(out_pooled, self.pool_kernel_size)
            .mean(dim=1)
        )  # (out_pooled,)

        w_t = w.transpose(0, 1).contiguous()  # (K, out_pooled)

        kernel = self._get_kernel(batch_size)
        out = kernel(x, w_t, b)
        return out