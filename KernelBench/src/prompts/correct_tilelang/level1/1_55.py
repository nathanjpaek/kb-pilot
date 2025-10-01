"""
Problem Name: 55_conv_standard_2D__asymmetric_input__square_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.558 runtime_stats={'mean': 0.558, 'std': 0.0134, 'min': 0.548, 'max': 0.676, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.155, 'std': 0.00875, 'min': 0.151, 'max': 0.233, 'num_trials': 100}, 'speedup_ratio': 0.278}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _conv2d_kernel_factory(
    N: int,
    C: int,
    H: int,
    W: int,
    F: int,
    K: int,
    STRIDE: int,
    PAD: int,
    DIL: int,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    OH = (H + 2 * PAD - DIL * (K - 1) - 1) // STRIDE + 1
    OW = (W + 2 * PAD - DIL * (K - 1) - 1) // STRIDE + 1
    K_TOTAL = K * K * C
    M_TOTAL = N * OH * OW

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv2d_im2col(
        x: T.Tensor((N, H, W, C), dtype),
        w: T.Tensor((K, K, C, F), dtype),
        y: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N), T.ceildiv(M_TOTAL, block_M), threads=128
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_acc = T.alloc_fragment((block_M, block_N), accum_dtype)
            Y_s = T.alloc_shared((block_M, block_N), dtype)

            w_flat = T.Tensor((K_TOTAL, F), dtype, w.data)
            y_flat = T.Tensor((M_TOTAL, F), dtype, y.data)

            T.clear(C_acc)

            for ko in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # ---- load X tile (im2col) ----
                for i, j in T.Parallel(block_M, block_K):
                    m_idx = by * block_M + i
                    k_idx = ko * block_K + j

                    if (m_idx < M_TOTAL) and (k_idx < K_TOTAL):
                        n = m_idx // (OH * OW)
                        oh = (m_idx % (OH * OW)) // OW
                        ow = m_idx % OW

                        kh = k_idx // (K * C)
                        kw = (k_idx // C) % K
                        c_idx = k_idx % C

                        ih = oh * STRIDE + kh * DIL - PAD
                        iw = ow * STRIDE + kw * DIL - PAD

                        valid = (
                            (ih >= 0)
                            and (ih < H)
                            and (iw >= 0)
                            and (iw < W)
                        )
                        A_s[i, j] = T.if_then_else(
                            valid, x[n, ih, iw, c_idx], T.Cast(dtype, 0)
                        )
                    else:
                        A_s[i, j] = T.Cast(dtype, 0)

                # ---- load W tile ----
                T.copy(w_flat[ko * block_K, bx * block_N], B_s)

                # ---- GEMM ----
                T.gemm(A_s, B_s, C_acc)

            # ---- store ----
            T.copy(C_acc, Y_s)
            T.copy(Y_s, y_flat[by * block_M, bx * block_N])

    return conv2d_im2col


class ModelNew(nn.Module):
    """
    Optimised 2-D convolution (square kernel) using TileLang.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert groups == 1, "Grouped conv not supported."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = kernel_size
        self.stride = stride
        self.pad = padding
        self.dil = dilation
        self.use_bias = bias

        # ---- parameters ----
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.K, self.K)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in = in_channels * self.K * self.K
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # ---- kernel cache ----
        self._kernels = {}

    def _get_kernel(self, N, C, H, W, dtype: torch.dtype):
        key = (N, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _conv2d_kernel_factory(
                N,
                C,
                H,
                W,
                self.out_channels,
                self.K,
                self.stride,
                self.pad,
                self.dil,
            )
        return self._kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = x_fp16.shape
        kernel_fn = self._get_kernel(N, C, H, W, x_fp16.dtype)

        # ---- layout transforms ----
        x_nhwc = x_fp16.permute(0, 2, 3, 1).contiguous()  # N H W C
        w_hwcf = w_fp16.permute(2, 3, 1, 0).contiguous()  # K K C F

        y_nhwc = kernel_fn(x_nhwc, w_hwcf)
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()       # back to NCHW

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)

        return y