"""
Problem Name: 1_conv_standard_2D__square_input__square_kernel
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.889 runtime_stats={'mean': 0.889, 'std': 0.06, 'min': 0.811, 'max': 1.05, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.327, 'std': 0.0424, 'min': 0.288, 'max': 0.478, 'num_trials': 100}, 'speedup_ratio': 0.368}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------
# TileLang kernel factory for 2-D convolution via im2col + GEMM
# ---------------------------------------------------------------------
def _build_conv2d_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    F: int,
    K: int,
    stride: int,
    padding: int,
    dilation: int,
    *,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    KH = KW = K
    OH = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1
    K_TOTAL = KH * KW * C  # im2col reduction dimension

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv2d_kernel(
        inp: T.Tensor((N, H, W, C), dtype),          # NHWC
        wgt: T.Tensor((KH, KW, C, F), dtype),        # HWCF
        out: T.Tensor((N, OH, OW, F), dtype),        # NHWC
    ):
        grid_x = T.ceildiv(F, block_N)
        grid_y = T.ceildiv(N * OH * OW, block_M)

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            # ------------------------------------------------------------------
            # Shared / fragment storage
            # ------------------------------------------------------------------
            A_sh = T.alloc_shared((block_M, block_K), dtype)      # im2col tile
            B_sh = T.alloc_shared((block_K, block_N), dtype)      # weight tile
            Acc  = T.alloc_fragment((block_M, block_N), accum_dtype)
            Out_sh = T.alloc_shared((block_M, block_N), dtype)    # staging

            T.clear(Acc)

            # ------------------------------------------------------------------
            # Flatten helper tensors (reuse memory)
            # ------------------------------------------------------------------
            w_flat  = T.Tensor((K_TOTAL, F), dtype, wgt.data)
            o_flat  = T.Tensor((N * OH * OW, F), dtype, out.data)

            # ------------------------------------------------------------------
            # Reduction over K_TOTAL using pipelined stages
            # ------------------------------------------------------------------
            for ko in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # ---- load im2col tile ---------------------------------------------------
                for ly, lx in T.Parallel(block_M, block_K):
                    m_idx = by * block_M + ly      # global row id in im2col matrix
                    k_idx = ko * block_K + lx      # global col id

                    batch = m_idx // (OH * OW)
                    oh    = (m_idx % (OH * OW)) // OW
                    ow    = m_idx % OW

                    kh  = (k_idx // (KW * C))
                    kw  = (k_idx // C) % KW
                    ci  = k_idx % C

                    ih = oh * stride + kh * dilation - padding
                    iw = ow * stride + kw * dilation - padding

                    in_bounds = (
                        (m_idx < N * OH * OW)
                        and (k_idx < K_TOTAL)
                        and (ih >= 0)
                        and (iw >= 0)
                        and (ih < H)
                        and (iw < W)
                    )
                    A_sh[ly, lx] = T.if_then_else(
                        in_bounds,
                        inp[batch, ih, iw, ci],
                        T.cast(0, dtype),
                    )

                # ---- load weight tile ---------------------------------------------------
                T.copy(w_flat[ko * block_K, bx * block_N], B_sh)

                # ---- barrier to ensure all data ready -----------------------------------
                T.tvm_storage_sync("shared")

                # ---- GEMM ---------------------------------------------------------------
                T.gemm(A_sh, B_sh, Acc)

            # ------------------------------------------------------------------
            # Store results back to global memory
            # ------------------------------------------------------------------
            T.copy(Acc, Out_sh)
            T.copy(Out_sh, o_flat[by * block_M, bx * block_N])

    return conv2d_kernel


# ---------------------------------------------------------------------
# PyTorch wrapper module
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    TileLang-accelerated Conv2D (square kernel, groups = 1).
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
        assert groups == 1, "Grouped convolution not supported in this optimized kernel"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias

        # ---------------------------------------------------------------
        # Parameter initialization identical to nn.Conv2d defaults
        # ---------------------------------------------------------------
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias_flag:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # kernel cache keyed by dynamic input shapes & dtype
        self._kern_cache = {}

    # -----------------------------------------------------------------
    # get / compile kernel
    # -----------------------------------------------------------------
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: torch.dtype):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.bfloat16, torch.float32) else "float32"
            self._kern_cache[key] = _build_conv2d_kernel(
                N,
                C,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                dtype=tl_dtype,
            )
        return self._kern_cache[key]

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = x_f16.shape
        kernel = self._get_kernel(N, C, H, W, x_f16.dtype)

        # Reorder to NHWC / HWCF for kernel expectations
        x_nhwc = x_f16.permute(0, 2, 3, 1).contiguous()     # N H W C
        w_hwcf = w_f16.permute(2, 3, 1, 0).contiguous()      # H W C F

        out_nhwc = kernel(x_nhwc, w_hwcf)                    # returns NHWC
        out_nchw = out_nhwc.permute(0, 3, 1, 2).contiguous()

        if self.bias is not None:
            out_nchw = out_nchw + self.bias.view(1, -1, 1, 1)

        return out_nchw