"""
Problem Name: 1_MLP
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.185 runtime_stats={'mean': 0.185, 'std': 0.0103, 'min': 0.173, 'max': 0.212, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.176, 'std': 0.0198, 'min': 0.164, 'max': 0.319, 'num_trials': 100}, 'speedup_ratio': 0.951}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_linear_kernel(
    M: int,
    N: int,
    K: int,
    fused_relu: bool,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """Return a TileLang kernel computing
         Y = (X @ W.T) + bias         (optionally followed by ReLU)

    X : (M, K)      – input minibatch
    W : (N, K)      – weight matrix in PyTorch layout (out, in)
    bias : (N,)     – bias vector
    Y : (M, N)      – output
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),       # (out,in)  -> will be transposed
        bias: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),            # bx – columns / output features
            T.ceildiv(M, block_M),            # by – rows / batch
            threads=128,
        ) as (bx, by):

            A_sh = T.alloc_shared((block_M, block_K), dtype)
            B_sh = T.alloc_shared((block_N, block_K), dtype)
            C_fr = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_fr)

            # Pipelined tiled GEMM  (X  @  W.T)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], A_sh)
                T.copy(W[bx * block_N, ko * block_K], B_sh)  # still (out,in)
                T.gemm(A_sh, B_sh, C_fr, transpose_B=True)   # → internally B.T

            # Add bias and (optionally) ReLU, then write out
            for ii, jj in T.Parallel(block_M, block_N):
                gi = by * block_M + ii    # global row
                gj = bx * block_N + jj    # global col / feature

                if (gi < M) and (gj < N):
                    val = C_fr[ii, jj] + T.Cast(accum_dtype, bias[gj])
                    val_fp16 = T.Cast(dtype, val)

                    if fused_relu:
                        zero = T.Cast(dtype, 0.0)
                        val_fp16 = T.max(val_fp16, zero)

                    Y[gi, gj] = val_fp16

    return linear_kernel


class FusedLinearReLU(nn.Module):
    """Linear layer (W,b) followed by ReLU – backed by a fused TileLang kernel."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Same layout as nn.Linear: (out, in)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self._kernel_cache = {}

    def _kernel(self, M: int, K: int):
        # N = out_features
        key = (M, self.weight.shape[0], K, True)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_kernel(
                M, key[1], K, fused_relu=True
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        W_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        M, K = x_fp16.shape
        kernel = self._kernel(M, K)
        y = kernel(x_fp16, W_fp16, b_fp16)  # (M, out)
        return y


class Linear(nn.Module):
    """Pure linear layer backed by a TileLang kernel (no activation)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self._kernel_cache = {}

    def _kernel(self, M: int, K: int):
        key = (M, self.weight.shape[0], K, False)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_kernel(
                M, key[1], K, fused_relu=False
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        W_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        M, K = x_fp16.shape
        kernel = self._kernel(M, K)
        y = kernel(x_fp16, W_fp16, b_fp16)  # (M, out)
        return y


import math


class ModelNew(nn.Module):
    """
    Fully-TileLang implementation of the original three-layer MLP:
        1000 → 400 → 800 → 500
    The two hidden layers use fused Linear+ReLU kernels, the final layer
    is pure Linear.
    """

    def __init__(self, input_size: int, layer_sizes, output_size: int):
        super().__init__()
        l1, l2 = layer_sizes
        self.fc1 = FusedLinearReLU(input_size, l1)
        self.fc2 = FusedLinearReLU(l1, l2)
        self.fc3 = Linear(l2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # All kernels already move data to CUDA / FP16 internally.
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x