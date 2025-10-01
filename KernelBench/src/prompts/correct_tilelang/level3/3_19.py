"""
Problem Name: 19_MobileNetV1
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.1 runtime_stats={'mean': 2.1, 'std': 0.016, 'min': 2.06, 'max': 2.14, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.09, 'std': 0.0182, 'min': 2.05, 'max': 2.16, 'num_trials': 100}, 'speedup_ratio': 0.995}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# -----------------------------------------------------------------------------#
# TileLang GEMM kernel factory                                                 #
# -----------------------------------------------------------------------------#
def _build_linear_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Returns a TileLang kernel computing  Y = X @ W.T + B
    Shapes:
        X : (batch_size, in_features)   – fp16
        W : (out_features, in_features) – fp16
        B : (out_features,)             – fp16
        Y : (batch_size, out_features)  – fp16
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        Y: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_features, block_N),
            T.ceildiv(batch_size, block_M),
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)
            C_f = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_f)

            # K-reduction loop (pipelined)
            for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], W_s)
                T.gemm(A_s, W_s, C_f, transpose_B=True)

            # bias add + store
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < batch_size) and (gn < out_features):
                    val = C_f[i, j] + B[gn]
                    Y[gm, gn] = T.Cast(dtype, val)

    return kernel


# -----------------------------------------------------------------------------#
# PyTorch wrapper with TileLang-accelerated FC                                 #
# -----------------------------------------------------------------------------#
class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000, input_channels: int = 3, alpha: float = 1.0):
        super().__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        a = alpha  # shorthand
        self.features = nn.Sequential(
            conv_bn(input_channels, int(32 * a), 2),
            conv_dw(int(32 * a), int(64 * a), 1),
            conv_dw(int(64 * a), int(128 * a), 2),
            conv_dw(int(128 * a), int(128 * a), 1),
            conv_dw(int(128 * a), int(256 * a), 2),
            conv_dw(int(256 * a), int(256 * a), 1),
            conv_dw(int(256 * a), int(512 * a), 2),
            conv_dw(int(512 * a), int(512 * a), 1),
            conv_dw(int(512 * a), int(512 * a), 1),
            conv_dw(int(512 * a), int(512 * a), 1),
            conv_dw(int(512 * a), int(512 * a), 1),
            conv_dw(int(512 * a), int(512 * a), 1),
            conv_dw(int(512 * a), int(1024 * a), 2),
            conv_dw(int(1024 * a), int(1024 * a), 1),
            nn.AvgPool2d(7),
        )

        # -------- TileLang-based fully-connected parameters -----------------
        self.in_features = int(1024 * a)
        self.out_features = num_classes

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features))
        # identical to nn.Linear init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

        # kernel cache keyed by (batch_size, dtype)
        self._kernel_cache = {}

    # --------------------------------------------------------------------- #
    def _get_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_kernel(
                batch_size, self.in_features, self.out_features, dtype=dtype
            )
        return self._kernel_cache[key]

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone in original precision (usually fp32)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten

        # Prepare tensors for TileLang kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(x_fp16.shape[0], dtype="float16")
        out_fp16 = kernel(x_fp16, w_fp16, b_fp16)

        return out_fp16.to(dtype=x.dtype)