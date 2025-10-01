"""
Problem Name: 57_conv_transposed_2D__square_input__square_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.358 runtime_stats={'mean': 0.358, 'std': 0.00416, 'min': 0.351, 'max': 0.368, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.12, 'std': 0.00987, 'min': 0.114, 'max': 0.211, 'num_trials': 100}, 'speedup_ratio': 0.335}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_deconv2d_kernel(
    N: int,
    CI: int,
    HI: int,
    WI: int,
    CO: int,
    K: int,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    HO = HI + K - 1  # stride=1 , padding=0 , output_padding=0
    WO = WI + K - 1
    K_TOTAL = CI * K * K

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv2d_kernel(
        X: T.Tensor((N, HI, WI, CI), dtype),          # NHWC
        Wt: T.Tensor((K, K, CI, CO), dtype),          # HWCI
        Out: T.Tensor((N, HO, WO, CO), dtype),        # NHWC
    ):
        with T.Kernel(
            T.ceildiv(CO, block_N),                  # grid x : output channel tiles
            T.ceildiv(N * HO * WO, block_M),         # grid y : output pixels tiles
            threads=128,
        ) as (bx, by):
            A_sh = T.alloc_shared((block_M, block_K), dtype)
            B_sh = T.alloc_shared((block_K, block_N), dtype)
            C_fr = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_fr)

            W_flat = T.Tensor((K_TOTAL, CO), dtype, Wt.data)
            for kk in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # -------- load X tile (im2col) ----------
                for mi, ki in T.Parallel(block_M, block_K):
                    m_global = by * block_M + mi
                    k_global = kk * block_K + ki
                    if (m_global < N * HO * WO) and (k_global < K_TOTAL):
                        n_idx   = m_global // (HO * WO)
                        rem     = m_global % (HO * WO)
                        h_out   = rem // WO
                        w_out   = rem % WO

                        kh_idx  = k_global // (K * CI)
                        kw_idx  = (k_global // CI) % K
                        c_idx   = k_global % CI

                        h_in = h_out - kh_idx
                        w_in = w_out - kw_idx
                        cond = (
                            (h_in >= 0) and (h_in < HI) and
                            (w_in >= 0) and (w_in < WI)
                        )
                        A_sh[mi, ki] = T.if_then_else(
                            cond,
                            X[n_idx, h_in, w_in, c_idx],
                            T.Cast(dtype, 0),
                        )
                    else:
                        A_sh[mi, ki] = T.Cast(dtype, 0)

                # -------- load W tile ----------
                T.copy(
                    W_flat[kk * block_K, bx * block_N],
                    B_sh,
                )

                # -------- GEMM ----------
                T.gemm(A_sh, B_sh, C_fr)

            # -------- store ----------
            for mi, ni in T.Parallel(block_M, block_N):
                m_global = by * block_M + mi
                n_global = bx * block_N + ni
                if (m_global < N * HO * WO) and (n_global < CO):
                    n_idx   = m_global // (HO * WO)
                    rem     = m_global % (HO * WO)
                    h_out   = rem // WO
                    w_out   = rem % WO
                    Out[n_idx, h_out, w_out, n_global] = T.Cast(dtype, C_fr[mi, ni])

    return deconv2d_kernel


class ModelNew(nn.Module):
    """
    TileLang-optimised ConvTranspose2d (square kernel, stride = 1, padding = 0,
    output_padding = 0, groups = 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert (
            stride == 1 and padding == 0 and output_padding == 0 and groups == 1
        ), "Only stride=1, padding=0, output_padding=0, groups=1 supported."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = kernel_size
        self.use_bias = bias

        # Parameters (same initialisation as nn.ConvTranspose2d)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, self.K, self.K)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = in_channels * self.K * self.K
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self._kernel_cache = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_deconv2d_kernel(
                N, self.in_channels, H, W, self.out_channels, self.K
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, CI, HI, WI = x_fp16.shape

        # Reorder tensors to NHWC / HWCI
        x_nhwc = x_fp16.permute(0, 2, 3, 1).contiguous()
        w_hwci = (
            self.weight.to(device="cuda", dtype=torch.float16)
            .permute(2, 3, 0, 1)
            .contiguous()
        )

        kernel = self._get_kernel(N, HI, WI, "float16")
        out_nhwc = kernel(x_nhwc, w_hwci)

        out = out_nhwc.permute(0, 3, 1, 2).contiguous()

        if self.use_bias:
            out += self.bias.view(1, -1, 1, 1).to(out.dtype)

        return out.to(orig_dtype)