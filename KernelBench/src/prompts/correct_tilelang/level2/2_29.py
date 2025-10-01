"""
Problem Name: 29_Matmul_Mish_Mish
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0386 runtime_stats={'mean': 0.0386, 'std': 0.0293, 'min': 0.0312, 'max': 0.31, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0458, 'std': 0.0131, 'min': 0.0394, 'max': 0.163, 'num_trials': 100}, 'speedup_ratio': 1.19}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_linear_kernel(batch_size, in_features, out_features,
                         block_M=128, block_N=64, block_K=32,
                         in_dtype="float16", accum_dtype="float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear(
        X: T.Tensor((batch_size, in_features), in_dtype),
        W: T.Tensor((out_features, in_features), in_dtype),
        B: T.Tensor((out_features,), in_dtype),
        Y: T.Tensor((batch_size, out_features), in_dtype),
    ):
        with T.Kernel(T.ceildiv(out_features, block_N),
                      T.ceildiv(batch_size, block_M),
                      threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            W_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=3):
                for i, j in T.Parallel(block_M, block_K):
                    g_m = by * block_M + i
                    g_k = ko * block_K + j
                    in_bounds = (g_m < batch_size) and (g_k < in_features)
                    A_shared[i, j] = T.if_then_else(
                        in_bounds, X[g_m, g_k], T.Cast(in_dtype, 0)
                    )

                for n, k in T.Parallel(block_N, block_K):
                    g_n = bx * block_N + n
                    g_k = ko * block_K + k
                    in_bounds = (g_n < out_features) and (g_k < in_features)
                    W_shared[n, k] = T.if_then_else(
                        in_bounds, W[g_n, g_k], T.Cast(in_dtype, 0)
                    )

                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                g_m = by * block_M + i
                g_n = bx * block_N + j
                if (g_m < batch_size) and (g_n < out_features):
                    val = C_local[i, j] + T.Cast(accum_dtype, B[g_n])
                    
                    # first Mish
                    exp1 = T.exp(val)
                    ln1 = T.log(T.Cast(accum_dtype, 1.0) + exp1)
                    tanh1 = T.tanh(ln1)
                    tmp1 = val * tanh1
                    
                    # second Mish
                    exp2 = T.exp(tmp1)
                    ln2 = T.log(T.Cast(accum_dtype, 1.0) + exp2)
                    tanh2 = T.tanh(ln2)
                    out = tmp1 * tanh2

                    Y[g_m, g_n] = T.Cast(in_dtype, out)

    return linear


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self._linear_kernels = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        if (batch_size, x.dtype) not in self._linear_kernels:
            self._linear_kernels[(batch_size, x.dtype)] = _build_linear_kernel(
                batch_size, self.in_features, self.out_features
            )

        kernel = self._linear_kernels[(batch_size, x.dtype)]

        x_fp16 = x.to(device="cuda", dtype=torch.float16, non_blocking=True)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        y = kernel(x_fp16, w_fp16, b_fp16)
        return y