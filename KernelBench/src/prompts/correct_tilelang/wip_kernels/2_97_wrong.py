import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _fused_kernel_factory(
    M,
    N,
    K,
    block_M=64,
    block_N=64,
    block_K=32,
    num_stages=2,
    divide_value=1.0,
    dtype="float16",
    accum_dtype="float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        linear_bias: T.Tensor((N,), dtype),
        bn_scale: T.Tensor((N,), dtype),
        bn_shift: T.Tensor((N,), dtype),
        extra_bias: T.Tensor((1,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Copy tiles into shared memory
                T.copy(X[by * block_M, ko * block_K], A_shared)
                T.copy(W[bx * block_N, ko * block_K], B_shared)
                # GEMM with B transposed
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            # Element-wise post-processing and store
            for i, j in T.Parallel(block_M, block_N):
                global_m = by * block_M + i
                global_n = bx * block_N + j 
                in_bound = (global_m < M) and (global_n < N)
                if in_bound:
                    val = C_local[i, j]
                    val += T.Cast(accum_dtype, linear_bias[global_n])
                    val = (
                        val * T.Cast(accum_dtype, bn_scale[global_n])
                        + T.Cast(accum_dtype, bn_shift[global_n])
                    )
                    val += T.Cast(accum_dtype, extra_bias[0])
                    val /= divide_value
                    val = val * T.sigmoid(val)
                    Out[global_m, global_n] = T.Cast(dtype, val)

    return fused_kernel


class ModelNew(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bn_eps=1e-5,
        bn_momentum=0.1,
        bias_shape=(1,),
        divide_value=1.0,
    ):
        super(ModelNew, self).__init__()

        # Linear parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.linear_bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.linear_bias, -bound, bound)

        # BatchNorm parameters & stats
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))
        self.eps = bn_eps
        self.momentum = bn_momentum

        # Extra bias
        self.extra_bias = nn.Parameter(torch.randn(bias_shape))

        # Scalar division constant
        self.divide_value = float(divide_value)

        # Kernel cache
        self._kernel_cache = {}

    def _get_kernel(self, M):
        key = (M, torch.float16)
        if key not in self._kernel_cache:
            kernel = _fused_kernel_factory(
                M,
                self.weight.shape[0],
                self.weight.shape[1],
                divide_value=self.divide_value,
            )
            self._kernel_cache[key] = kernel
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor):
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        W_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        linear_bias_fp16 = self.linear_bias.to(device="cuda", dtype=torch.float16)

        # BatchNorm fusion parameters
        running_var_eps = self.running_var + self.eps
        bn_scale = self.bn_weight / torch.sqrt(running_var_eps)
        bn_shift = self.bn_bias - self.running_mean * bn_scale
        bn_scale_fp16 = bn_scale.to(device="cuda", dtype=torch.float16)
        bn_shift_fp16 = bn_shift.to(device="cuda", dtype=torch.float16)

        extra_bias_fp16 = self.extra_bias.to(device="cuda", dtype=torch.float16)

        M = x_fp16.shape[0]
        kernel = self._get_kernel(M)

        out_fp16 = kernel(
            x_fp16,
            W_fp16,
            linear_bias_fp16,
            bn_scale_fp16,
            bn_shift_fp16,
            extra_bias_fp16,
        )

        return out_fp16