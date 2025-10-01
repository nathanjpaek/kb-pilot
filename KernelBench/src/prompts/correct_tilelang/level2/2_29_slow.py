"""
Problem Name: 29_Matmul_Mish_Mish
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0906 runtime_stats={'mean': 0.0906, 'std': 0.00232, 'min': 0.0868, 'max': 0.0972, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0447, 'std': 0.00201, 'min': 0.0425, 'max': 0.0567, 'num_trials': 100}, 'speedup_ratio': 0.493}}
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

            for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
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
                    Y[g_m, g_n] = T.Cast(in_dtype, val)

    return linear


def build_mish_kernel(N, block_size: int = 256, dtype: str = "float16"):
    """
    Factory that returns a TileLang kernel which applies element-wise mish
    to a flat tensor of length N.
    mish(x) = x * tanh(log(1 + exp(x)))
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def mish_kernel(
        X: T.Tensor((N,), dtype),
        Y: T.Tensor((N,), dtype),
    ):
        # 1-D grid; each block has `block_size` threads
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)  # threadIdx.x
            idx: T.int32 = bx * block_size + tx
            if idx < N:
                x_val = X[idx]
                # mish(x) = x * tanh(log(1 + exp(x)))
                exp_x = T.exp(x_val)
                log1p_exp_x = T.log(T.Cast(dtype, 1.0) + exp_x)
                tanh_val = T.tanh(log1p_exp_x)
                Y[idx] = x_val * tanh_val

    return mish_kernel


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
        self._mish_kernels = {}

    def _get_mish_kernel(self, numel: int, dtype: torch.dtype):
        key = (numel, dtype)
        if key not in self._mish_kernels:
            tl_dtype = "float16" if dtype == torch.float16 else "float"
            self._mish_kernels[key] = build_mish_kernel(numel, dtype=tl_dtype)
        return self._mish_kernels[key]

    def _apply_mish_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Mish activation to the input tensor using the compiled
        TileLang kernel.
        """
        orig_shape = x.shape
        orig_dtype = x.dtype
        # TileLang works best with fp16 on CUDA
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        flat_x = x_fp16.view(-1)

        kernel = self._get_mish_kernel(flat_x.numel(), flat_x.dtype)
        flat_out = kernel(flat_x)
        out = flat_out.view(orig_shape).to(orig_dtype)  # cast back to original dtype
        return out

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

        # Apply mish activation twice using TileLang kernel
        y = self._apply_mish_kernel(y)
        y = self._apply_mish_kernel(y)
        return y