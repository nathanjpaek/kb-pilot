"""
Problem Name: 87_conv_pointwise_2D
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.753 runtime_stats={'mean': 0.753, 'std': 0.0129, 'min': 0.739, 'max': 0.862, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.13, 'std': 0.00954, 'min': 0.124, 'max': 0.217, 'num_trials': 100}, 'speedup_ratio': 0.173}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _pointwise_conv_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    F: int,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    M_TOTAL = N * H * W  # flattened spatial-batch dimension

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((N, H, W, C), dtype),          # NHWC
        W_cf: T.Tensor((C, F), dtype),             # CF
        BIAS: T.Tensor((F,), dtype),               # F
        Y: T.Tensor((N, H, W, F), dtype),          # NHWF
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),                 # grid x  – output channels
            T.ceildiv(M_TOTAL, block_M),           # grid y  – flattened pixels
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_acc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_acc)

            for ko in T.Pipelined(T.ceildiv(C, block_K), num_stages=num_stages):
                # ----- load X tile to shared -----
                for i, j in T.Parallel(block_M, block_K):
                    m_idx = by * block_M + i
                    k_idx = ko * block_K + j

                    cond = (m_idx < M_TOTAL) and (k_idx < C)
                    if cond:
                        n_idx = m_idx // (H * W)
                        hw_idx = m_idx % (H * W)
                        h_idx = hw_idx // W
                        w_idx = hw_idx % W
                        A_s[i, j] = X[n_idx, h_idx, w_idx, k_idx]
                    else:
                        A_s[i, j] = T.cast(0, dtype)

                # ----- load Weight tile to shared -----
                T.copy(
                    W_cf[ko * block_K : ko * block_K + block_K,
                         bx * block_N : bx * block_N + block_N],
                    B_s,
                )

                # ----- GEMM -----
                T.gemm(A_s, B_s, C_acc)

            # ----- store with bias -----
            for i, j in T.Parallel(block_M, block_N):
                m_idx = by * block_M + i
                f_idx = bx * block_N + j
                if (m_idx < M_TOTAL) and (f_idx < F):
                    n_idx = m_idx // (H * W)
                    hw_idx = m_idx % (H * W)
                    h_idx = hw_idx // W
                    w_idx = hw_idx % W
                    val = C_acc[i, j] + T.cast(BIAS[f_idx], accum_dtype)
                    Y[n_idx, h_idx, w_idx, f_idx] = T.cast(val, dtype)

    return main


class ModelNew(nn.Module):
    """
    Point-wise (1×1) convolution accelerated by TileLang.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_flag = bias

        # ---- parameters ----
        w = torch.empty(out_channels, in_channels, 1, 1)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = nn.Parameter(w.squeeze(-1).squeeze(-1))  # shape (F,C)

        if bias:
            bound = 1 / math.sqrt(in_channels)
            b = torch.empty(out_channels)
            nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.register_buffer("_zero_bias", torch.zeros(out_channels))

        # ---- kernel cache ----
        self._kernels = {}

    def _get_kernel(self, N: int, H: int, W: int, dtype: torch.dtype):
        key = (N, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _pointwise_conv_kernel(
                N,
                self.in_channels,
                H,
                W,
                self.out_channels,
            )
        return self._kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C_in, H, W = x_fp16.shape
        assert C_in == self.in_channels

        # weight CF layout
        w_cf = self.weight.to(device="cuda", dtype=torch.float16).contiguous().t().contiguous()  # (C,F)

        bias_fp16 = (
            self.bias.to(device="cuda", dtype=torch.float16)
            if self.bias_flag
            else self._zero_bias.to(device="cuda", dtype=torch.float16)
        )

        kernel_fn = self._get_kernel(N, H, W, x_fp16.dtype)

        x_nhwc = x_fp16.permute(0, 2, 3, 1).contiguous()
        y_nhwf = kernel_fn(x_nhwc, w_cf, bias_fp16)
        y = y_nhwf.permute(0, 3, 1, 2).contiguous()
        return y