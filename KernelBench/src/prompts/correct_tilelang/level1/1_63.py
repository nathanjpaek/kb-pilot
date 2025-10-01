"""
Problem Name: 63_conv_standard_2D__square_input__square_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.39 runtime_stats={'mean': 1.39, 'std': 0.0125, 'min': 1.38, 'max': 1.5, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.294, 'std': 0.00342, 'min': 0.289, 'max': 0.301, 'num_trials': 100}, 'speedup_ratio': 0.212}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


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
    K_TOTAL = K * K * C  # flattened K dimension (im2col)
    M_TOTAL = N * OH * OW

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        data: T.Tensor((N, H, W, C), dtype),
        weight: T.Tensor((K, K, C, F), dtype),
        out: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),        # grid dim x – output channels
            T.ceildiv(M_TOTAL, block_M),  # grid dim y – output pixels * batch
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_s = T.alloc_shared((block_M, block_N), dtype)

            weight_flat = T.Tensor((K_TOTAL, F), dtype, weight.data)
            out_flat = T.Tensor((M_TOTAL, F), dtype, out.data)

            T.clear(C_frag)

            for ko in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # ---------------- load im2col tile ----------------
                for i, j in T.Parallel(block_M, block_K):
                    m_idx = by * block_M + i
                    k_idx = ko * block_K + j

                    if (m_idx < M_TOTAL) and (k_idx < K_TOTAL):
                        n = m_idx // (OH * OW)
                        oh = (m_idx % (OH * OW)) // OW
                        ow = m_idx % OW

                        kh = k_idx // (K * C)
                        kw = (k_idx // C) % K
                        c  = k_idx % C

                        ih = oh * stride + kh * dil - pad
                        iw = ow * stride + kw * dil - pad

                        valid = (
                            (ih >= 0)
                            and (iw >= 0)
                            and (ih < H)
                            and (iw < W)
                        )
                        A_s[i, j] = T.if_then_else(
                            valid,
                            data[n, ih, iw, c],
                            T.Cast(dtype, 0),
                        )
                    else:
                        A_s[i, j] = T.Cast(dtype, 0)

                # ---------------- load weight tile ----------------
                T.copy(weight_flat[ko * block_K, bx * block_N], B_s)

                # ---------------- GEMM ----------------
                T.gemm(A_s, B_s, C_frag)

            # ---------------- store results ----------------
            T.copy(C_frag, out_s)
            T.copy(out_s, out_flat[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
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
            raise NotImplementedError("Grouped conv not supported.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias_flag:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            with torch.no_grad():
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self._kernel_cache = {}

    def _get_kernel(self, N, C, H, W, dtype):
        key = (N, C, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _conv2d_kernel(
                N,
                C,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16).contiguous()
        w = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = x.shape
        kernel_fn = self._get_kernel(N, C, H, W, x.dtype)

        # Layout transforms
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_hwcf = w.permute(2, 3, 1, 0).contiguous()

        out_nhwc = kernel_fn(x_nhwc, w_hwcf)
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out