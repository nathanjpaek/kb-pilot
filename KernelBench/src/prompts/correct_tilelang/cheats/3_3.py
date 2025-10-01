"""
Problem Name: 3_DeepNarrowMLP
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.413 runtime_stats={'mean': 0.413, 'std': 0.00586, 'min': 0.402, 'max': 0.432, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.239, 'std': 0.00391, 'min': 0.232, 'max': 0.253, 'num_trials': 100}, 'speedup_ratio': 0.579}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def compile_linear_kernel(B, In, Out, fused):
    block_M, block_N, block_K = 64, 64, 32
    dtype, accum_dtype = "float16", "float"

    if fused:

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            X: T.Tensor((B, In), dtype),
            WT: T.Tensor((In, Out), dtype),
            Bias: T.Tensor((Out,), dtype),
            OutT: T.Tensor((B, Out), dtype),
        ):
            with T.Kernel(
                T.ceildiv(Out, block_N),
                T.ceildiv(B, block_M),
                threads=128,
            ) as (bx, by):
                A_s = T.alloc_shared((block_M, block_K), dtype)
                B_s = T.alloc_shared((block_K, block_N), dtype)
                C = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.clear(C)
                for ko in T.Pipelined(T.ceildiv(In, block_K), num_stages=2):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        k = ko * block_K + j
                        valid = (m < B) and (k < In)
                        A_s[i, j] = T.if_then_else(valid, X[m, k], T.Cast(dtype, 0))
                    for i, j in T.Parallel(block_K, block_N):
                        k = ko * block_K + i
                        n = bx * block_N + j
                        valid = (k < In) and (n < Out)
                        B_s[i, j] = T.if_then_else(valid, WT[k, n], T.Cast(dtype, 0))
                    T.gemm(A_s, B_s, C)

                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    if (m < B) and (n < Out):
                        val = C[i, j] + T.Cast(accum_dtype, Bias[n])
                        val = T.max(val, T.Cast(accum_dtype, 0))
                        OutT[m, n] = T.Cast(dtype, val)

    else:

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            X: T.Tensor((B, In), dtype),
            WT: T.Tensor((In, Out), dtype),
            Bias: T.Tensor((Out,), dtype),
            OutT: T.Tensor((B, Out), dtype),
        ):
            with T.Kernel(
                T.ceildiv(Out, block_N),
                T.ceildiv(B, block_M),
                threads=128,
            ) as (bx, by):
                A_s = T.alloc_shared((block_M, block_K), dtype)
                B_s = T.alloc_shared((block_K, block_N), dtype)
                C = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.clear(C)
                for ko in T.Pipelined(T.ceildiv(In, block_K), num_stages=2):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        k = ko * block_K + j
                        valid = (m < B) and (k < In)
                        A_s[i, j] = T.if_then_else(valid, X[m, k], T.Cast(dtype, 0))
                    for i, j in T.Parallel(block_K, block_N):
                        k = ko * block_K + i
                        n = bx * block_N + j
                        valid = (k < In) and (n < Out)
                        B_s[i, j] = T.if_then_else(valid, WT[k, n], T.Cast(dtype, 0))
                    T.gemm(A_s, B_s, C)

                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    if (m < B) and (n < Out):
                        val = C[i, j] + T.Cast(accum_dtype, Bias[n])
                        OutT[m, n] = T.Cast(dtype, val)

    return kernel


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()

        sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.in_sizes = sizes[:-1]
        self.out_sizes = sizes[1:]

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for in_f, out_f in zip(self.in_sizes, self.out_sizes):
            w = nn.Parameter(torch.empty(out_f, in_f))
            b = nn.Parameter(torch.empty(out_f))
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            bound = 1 / math.sqrt(in_f)
            nn.init.uniform_(b, -bound, bound)
            self.weights.append(w)
            self.biases.append(b)

        self._kernel_cache = {}

    def _get_kernel(self, batch, in_f, out_f, fused):
        key = (batch, in_f, out_f, fused)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = compile_linear_kernel(batch, in_f, out_f, fused)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.to(device="cuda", dtype=torch.float16)
        batch = x.shape[0]

        num_layers = len(self.weights)
        for idx, (w, b, in_f, out_f) in enumerate(
            zip(self.weights, self.biases, self.in_sizes, self.out_sizes)
        ):
            fused = idx < num_layers - 1
            wt = w.t().contiguous().to(dtype=torch.float16, device="cuda")
            bias = b.to(dtype=torch.float16, device="cuda")
            kernel = self._get_kernel(batch, in_f, out_f, fused)
            x = kernel(x, wt, bias)
        return x.to(torch.float32)