"""
Problem Name: 2_conv_standard_3D__square_input__asymmetric_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=17.7 runtime_stats={'mean': 17.7, 'std': 0.0249, 'min': 17.6, 'max': 17.8, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 3.59, 'std': 0.0212, 'min': 3.58, 'max': 3.8, 'num_trials': 100}, 'speedup_ratio': 0.203}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# -----------------------------------------------------------------
#  TileLang kernel factory
# -----------------------------------------------------------------
def _conv3d_kernel_factory(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    F: int,
    KD: int,
    KH: int,
    KW: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dil_d: int,
    dil_h: int,
    dil_w: int,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    OD = (D + 2 * pad_d - dil_d * (KD - 1) - 1) // stride_d + 1
    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    M_TOTAL = N * OD * OH * OW
    K_TOTAL = KD * KH * KW * C

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        x: T.Tensor((N, D, H, W, C), dtype),        # NDHWC
        w: T.Tensor((KD, KH, KW, C, F), dtype),     # KD KH KW C F
        y: T.Tensor((N, OD, OH, OW, F), dtype),     # NODHWF
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),
            T.ceildiv(M_TOTAL, block_M),
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            Y_s = T.alloc_shared((block_M, block_N), dtype)

            w_flat = T.Tensor((K_TOTAL, F), dtype, w.data)
            y_flat = T.Tensor((M_TOTAL, F), dtype, y.data)

            T.clear(C_frag)

            for ko in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # -------------------------------------------------
                #  Load X-tile (im2col) into shared memory
                # -------------------------------------------------
                for i, j in T.Parallel(block_M, block_K):
                    m = by * block_M + i
                    k = ko * block_K + j

                    if (m < M_TOTAL) and (k < K_TOTAL):
                        n_idx = m // (OD * OH * OW)
                        rem_m = m % (OD * OH * OW)
                        od_idx = rem_m // (OH * OW)
                        rem_m2 = rem_m % (OH * OW)
                        oh_idx = rem_m2 // OW
                        ow_idx = rem_m2 % OW

                        kd_idx = k // (KH * KW * C)
                        rem_k1 = k % (KH * KW * C)
                        kh_idx = rem_k1 // (KW * C)
                        rem_k2 = rem_k1 % (KW * C)
                        kw_idx = rem_k2 // C
                        c_idx  = rem_k2 % C

                        d_in = od_idx * stride_d + kd_idx * dil_d - pad_d
                        h_in = oh_idx * stride_h + kh_idx * dil_h - pad_h
                        w_in = ow_idx * stride_w + kw_idx * dil_w - pad_w

                        in_bounds = (
                            (d_in >= 0) and (d_in < D) and
                            (h_in >= 0) and (h_in < H) and
                            (w_in >= 0) and (w_in < W)
                        )
                        A_s[i, j] = T.if_then_else(
                            in_bounds,
                            x[n_idx, d_in, h_in, w_in, c_idx],
                            T.Cast(dtype, 0),
                        )
                    else:
                        A_s[i, j] = T.Cast(dtype, 0)

                # -------------------------------------------------
                #  Load W-tile into shared memory
                # -------------------------------------------------
                T.copy(w_flat[ko * block_K, bx * block_N], B_s)

                # -------------------------------------------------
                #  GEMM
                # -------------------------------------------------
                T.gemm(A_s, B_s, C_frag)

            # -----------------------------------------------------
            #  Store results
            # -----------------------------------------------------
            T.copy(C_frag, Y_s)
            T.copy(Y_s, y_flat[by * block_M, bx * block_N])

    return main


# -----------------------------------------------------------------
#  PyTorch wrapper
# -----------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if groups != 1:
            raise NotImplementedError("Grouped convolution is not supported.")

        # Unpack 3-D params ---------------------------------------------------
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kd, self.kh, self.kw = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else kernel_size
        )

        self.sd, self.sh, self.sw = (
            (stride, stride, stride) if isinstance(stride, int) else stride
        )
        self.pd, self.ph, self.pw = (
            (padding, padding, padding) if isinstance(padding, int) else padding
        )
        self.dd, self.dh, self.dw = (
            (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        )

        self.use_bias = bias

        # Parameters ----------------------------------------------------------
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                self.kd,
                self.kh,
                self.kw,
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = in_channels * self.kd * self.kh * self.kw
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # ---------------------------------------------------------------------
        self._kernel_cache = {}

    # -------------------------------------------------------------------------
    #  Kernel retrieval / compilation
    # -------------------------------------------------------------------------
    def _get_kernel(self, N, D, H, W, dtype):
        key = (N, D, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _conv3d_kernel_factory(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.kd,
                self.kh,
                self.kw,
                self.sd,
                self.sh,
                self.sw,
                self.pd,
                self.ph,
                self.pw,
                self.dd,
                self.dh,
                self.dw,
            )
        return self._kernel_cache[key]

    # -------------------------------------------------------------------------
    #  Forward
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input as N C D H W (PyTorch default, even if comment said width first)
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, D, H, W = x_fp16.shape
        kernel_fn = self._get_kernel(N, D, H, W, x_fp16.dtype)

        # Layout transforms ----------------------------------------------------
        x_ndhwc = x_fp16.permute(0, 2, 3, 4, 1).contiguous()          # NDHWC
        w_kd_kh_kw_c_f = w_fp16.permute(2, 3, 4, 1, 0).contiguous()   # KD KH KW C F

        y_ndhwf = kernel_fn(x_ndhwc, w_kd_kh_kw_c_f)
        y_ncdhw = y_ndhwf.permute(0, 4, 1, 2, 3).contiguous()         # N C_out D H W

        if self.bias is not None:
            y_ncdhw = y_ncdhw + self.bias.view(1, -1, 1, 1, 1)

        return y_ncdhw