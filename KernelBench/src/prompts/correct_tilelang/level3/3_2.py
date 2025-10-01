"""
Problem Name: 2_ShallowWideMLP
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.144 runtime_stats={'mean': 0.144, 'std': 0.00318, 'min': 0.139, 'max': 0.157, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0929, 'std': 0.00255, 'min': 0.0895, 'max': 0.108, 'num_trials': 100}, 'speedup_ratio': 0.645}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ------------------------------------------------------------
# Kernel builders
# ------------------------------------------------------------
def _build_kernel(
    B: int,
    IN: int,
    OUT: int,
    fuse_relu: bool,
    *,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Build a TileLang kernel computing:

        Y = ReLU(X @ W^T + bias)          if fuse_relu
        Y =        X @ W^T + bias         otherwise

    Shapes:
        X    : (B, IN)
        W    : (OUT, IN)
        bias : (OUT,)
        Y    : (B, OUT)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_kernel(
        X: T.Tensor((B, IN), dtype),
        W: T.Tensor((OUT, IN), dtype),
        bias: T.Tensor((OUT,), accum_dtype),  # keep bias in higher precision
        Y: T.Tensor((B, OUT), dtype),
    ):
        with T.Kernel(
            T.ceildiv(OUT, block_N), T.ceildiv(B, block_M), threads=128
        ) as (bx, by):
            X_sh = T.alloc_shared((block_M, block_K), dtype)
            W_sh = T.alloc_shared((block_N, block_K), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_frag)

            # Tiled GEMM
            for ko in T.Pipelined(T.ceildiv(IN, block_K), num_stages=num_stages):
                # Move a tile of X and W into shared memory
                T.copy(X[by * block_M, ko * block_K], X_sh)
                T.copy(W[bx * block_N, ko * block_K], W_sh)

                # Multiply-accumulate (W gets transposed inside GEMM)
                T.gemm(X_sh, W_sh, C_frag, transpose_B=True)

            # Epilogue: bias (+ ReLU)
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < B) and (gj < OUT):
                    val = C_frag[i, j] + bias[gj]
                    if fuse_relu:
                        zero = T.Cast(accum_dtype, 0)
                        val = T.max(val, zero)
                    Y[gi, gj] = T.Cast(dtype, val)

    return linear_kernel


def build_linear_relu_kernel(B: int, IN: int, OUT: int):
    return _build_kernel(B, IN, OUT, fuse_relu=True)


def build_linear_kernel(B: int, IN: int, OUT: int):
    return _build_kernel(B, IN, OUT, fuse_relu=False)


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Multi-layer perceptron implemented entirely with custom TileLang
    kernels. Linear and ReLU are fused except for the final linear layer.
    """

    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()

        # Layer dimensions
        self.sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
        self.num_layers = len(self.sizes) - 1

        # Parameters
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for in_feat, out_feat in zip(self.sizes[:-1], self.sizes[1:]):
            w = nn.Parameter(torch.empty(out_feat, in_feat))
            b = nn.Parameter(torch.empty(out_feat))
            self._init_linear(w, b, in_feat)
            self.weights.append(w)
            self.biases.append(b)

        # Kernel cache:  key = (B, IN, OUT, fused_bool)
        self._kernel_cache = {}

    # ----------------------------------------
    # Helpers
    # ----------------------------------------
    @staticmethod
    def _init_linear(w: torch.Tensor, b: torch.Tensor, fan_in: int):
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(b, -bound, bound)

    def _get_kernel(self, B: int, IN: int, OUT: int, fused: bool):
        key = (B, IN, OUT, fused)
        if key not in self._kernel_cache:
            builder = build_linear_relu_kernel if fused else build_linear_kernel
            self._kernel_cache[key] = builder(B, IN, OUT)
        return self._kernel_cache[key]

    # ----------------------------------------
    # Forward
    # ----------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B = x.shape[0]

        # Start in fp16 on GPU
        x = x.to(device="cuda", dtype=torch.float16, non_blocking=True)

        for idx in range(self.num_layers):
            W = self.weights[idx].to(device="cuda", dtype=torch.float16)
            b = self.biases[idx].to(device="cuda", dtype=torch.float32)

            IN = W.shape[1]
            OUT = W.shape[0]
            fused = idx < (self.num_layers - 1)  # final layer has no ReLU

            kernel = self._get_kernel(B, IN, OUT, fused)
            x = kernel(x, W, b)  # (B, OUT)

        return x.to(orig_dtype)