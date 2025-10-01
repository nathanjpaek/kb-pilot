"""
Problem Name: 1_conv_standard_2D__square_input__square_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.53 runtime_stats={'mean': 1.53, 'std': 0.0153, 'min': 1.51, 'max': 1.64, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.334, 'std': 0.00947, 'min': 0.329, 'max': 0.418, 'num_trials': 100}, 'speedup_ratio': 0.218}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------
# TileLang convolution kernel factory (square input, square kernel)
# ----------------------------------------------------------------------
def _conv2d_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    F: int,
    K: int,
    stride: int,
    pad: int,
    dil: int,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 64,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    OH = (H + 2 * pad - dil * (K - 1) - 1) // stride + 1
    OW = (W + 2 * pad - dil * (K - 1) - 1) // stride + 1
    K_TOTAL = K * K * C                # flattened K dimension after im2col
    M_TOTAL = N * OH * OW              # flattened output pixels (M dimension)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        x: T.Tensor((N, H, W, C), dtype),          # NHWC
        w: T.Tensor((K, K, C, F), dtype),          # KKCF
        y: T.Tensor((N, OH, OW, F), dtype),        # NHWF
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),                 # grid.x → output channels
            T.ceildiv(M_TOTAL, block_M),           # grid.y → output pixels
            threads=128,
        ) as (bx, by):

            # Shared/fragment buffers
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            Y_s = T.alloc_shared((block_M, block_N), dtype)

            # Flattened views (no copies)
            w_flat = T.Tensor((K_TOTAL, F), dtype, w.data)
            y_flat = T.Tensor((M_TOTAL, F), dtype, y.data)

            T.clear(C_frag)

            for ko in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # ---------------- im2col load: X → shared ----------------
                for i, j in T.Parallel(block_M, block_K):
                    m_idx = by * block_M + i
                    k_idx = ko * block_K + j

                    if (m_idx < M_TOTAL) and (k_idx < K_TOTAL):
                        n  = m_idx // (OH * OW)
                        oh = (m_idx % (OH * OW)) // OW
                        ow = m_idx % OW

                        kh = k_idx // (K * C)
                        kw = (k_idx // C) % K
                        c  = k_idx % C

                        ih = oh * stride + kh * dil - pad
                        iw = ow * stride + kw * dil - pad

                        valid = (
                            (ih >= 0) and (ih < H) and
                            (iw >= 0) and (iw < W)
                        )
                        A_s[i, j] = T.if_then_else(
                            valid,
                            x[n, ih, iw, c],
                            T.cast(0, dtype),
                        )
                    else:
                        A_s[i, j] = T.cast(0, dtype)

                # ---------------- weight load: W → shared ----------------
                T.copy(w_flat[ko * block_K, bx * block_N], B_s)

                # ---------------- GEMM ----------------
                T.gemm(A_s, B_s, C_frag)

            # ---------------- store results ----------------
            T.copy(C_frag, Y_s)
            T.copy(Y_s, y_flat[by * block_M, bx * block_N])

    return main


# ----------------------------------------------------------------------
# PyTorch wrapper that replaces nn.Conv2d
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    High-performance 2-D convolution (square kernel) using TileLang.
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
        if groups != 1:
            raise NotImplementedError("Grouped convolution is not supported.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = kernel_size
        self.stride = stride
        self.pad = padding
        self.dil = dilation
        self.use_bias = bias

        # ---------------- parameters ----------------
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.K, self.K)
        )
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in = in_channels * self.K * self.K
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            with torch.no_grad():
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # cache for compiled kernels
        self._kernel_cache = {}

    # ------------------------------------------------------------------
    # Return (and cache) a compiled TileLang kernel for current shapes
    # ------------------------------------------------------------------
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: torch.dtype):
        key = (N, C, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _conv2d_kernel(
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
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # prepare data
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = x_fp16.shape
        kernel = self._get_kernel(N, C, H, W, x_fp16.dtype)

        # layout transforms
        x_nhwc = x_fp16.permute(0, 2, 3, 1).contiguous()   # NHWC
        w_hwcf = w_fp16.permute(2, 3, 1, 0).contiguous()   # KKCF

        y_nhwc = kernel(x_nhwc, w_hwcf)                   # NHWC
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()       # NCHW

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)

        return y