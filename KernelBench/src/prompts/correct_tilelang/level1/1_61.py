"""
Problem Name: 61_conv_transposed_3D__square_input__square_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.62 runtime_stats={'mean': 1.62, 'std': 0.032, 'min': 1.6, 'max': 1.93, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.351, 'std': 0.0109, 'min': 0.342, 'max': 0.451, 'num_trials': 100}, 'speedup_ratio': 0.217}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_deconv3d_kernel(
    N: int,
    CI: int,
    D: int,
    H: int,
    W: int,
    CO: int,
    K: int,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    DO = D + K - 1  # stride = 1 , padding = 0 , output_padding = 0
    HO = H + K - 1
    WO = W + K - 1

    K_TOTAL = CI * K * K * K  # flattened kernel dim
    M_TOTAL = N * DO * HO * WO  # flattened output spatial dim

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv3d_kernel(
        X: T.Tensor((N, D, H, W, CI), dtype),          # NDHWC
        Wt: T.Tensor((K, K, K, CI, CO), dtype),        # KK K CI CO
        Out: T.Tensor((N, DO, HO, WO, CO), dtype),     # NDHWC
    ):
        with T.Kernel(
            T.ceildiv(CO, block_N),                   # grid x : output channel tiles
            T.ceildiv(M_TOTAL, block_M),              # grid y : output position tiles
            threads=128,
        ) as (bx, by):
            A_sh = T.alloc_shared((block_M, block_K), dtype)
            B_sh = T.alloc_shared((block_K, block_N), dtype)
            C_fr = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_fr)

            W_flat = T.Tensor((K_TOTAL, CO), dtype, Wt.data)

            for kk in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # ---- load X tile (im2col for deconv) ----
                for mi, ki in T.Parallel(block_M, block_K):
                    m_global = by * block_M + mi
                    k_global = kk * block_K + ki

                    if (m_global < M_TOTAL) and (k_global < K_TOTAL):
                        # decode output indices
                        n_idx  = m_global // (DO * HO * WO)
                        rem1   = m_global % (DO * HO * WO)
                        d_out  = rem1 // (HO * WO)
                        rem2   = rem1 % (HO * WO)
                        h_out  = rem2 // WO
                        w_out  = rem2 % WO

                        # decode kernel indices
                        kd_idx = k_global // (K * K * CI)
                        remk1  = k_global % (K * K * CI)
                        kh_idx = remk1 // (K * CI)
                        remk2  = remk1 % (K * CI)
                        kw_idx = remk2 // CI
                        c_idx  = remk2 % CI

                        d_in = d_out - kd_idx
                        h_in = h_out - kh_idx
                        w_in = w_out - kw_idx

                        cond = (
                            (d_in >= 0) and (d_in < D) and
                            (h_in >= 0) and (h_in < H) and
                            (w_in >= 0) and (w_in < W)
                        )
                        A_sh[mi, ki] = T.if_then_else(
                            cond,
                            X[n_idx, d_in, h_in, w_in, c_idx],
                            T.Cast(dtype, 0),
                        )
                    else:
                        A_sh[mi, ki] = T.Cast(dtype, 0)

                # ---- load W tile ----
                T.copy(
                    W_flat[kk * block_K, bx * block_N],
                    B_sh,
                )

                # ---- GEMM ----
                T.gemm(A_sh, B_sh, C_fr)

            # ---- store results ----
            for mi, ni in T.Parallel(block_M, block_N):
                m_global = by * block_M + mi
                n_global = bx * block_N + ni
                if (m_global < M_TOTAL) and (n_global < CO):
                    n_idx  = m_global // (DO * HO * WO)
                    rem1   = m_global % (DO * HO * WO)
                    d_out  = rem1 // (HO * WO)
                    rem2   = rem1 % (HO * WO)
                    h_out  = rem2 // WO
                    w_out  = rem2 % WO
                    Out[n_idx, d_out, h_out, w_out, n_global] = T.Cast(dtype, C_fr[mi, ni])

    return deconv3d_kernel


class ModelNew(nn.Module):
    """TileLang-optimised ConvTranspose3d (square kernel, stride=1, padding=0,
    output_padding=0, groups=1)."""

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
        # Supported configuration checks
        assert (
            stride == 1 and padding == 0 and output_padding == 0 and groups == 1
        ), "Only stride=1, padding=0, output_padding=0, groups=1 supported."

        self.CI = in_channels
        self.CO = out_channels
        self.K = kernel_size
        self.use_bias = bias

        # Parameters (identical init as nn.ConvTranspose3d)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, self.K, self.K, self.K)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = in_channels * self.K ** 3
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self._kernel_cache = {}

    # --------------------------------------------------------------
    def _get_kernel(self, N: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_deconv3d_kernel(
                N, self.CI, D, H, W, self.CO, self.K
            )
        return self._kernel_cache[key]

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, CI, D, H, W = x_fp16.shape

        # Reorder input to NDHWC
        x_ndhwc = x_fp16.permute(0, 2, 3, 4, 1).contiguous()

        # Prepare weight tensor, permute to KK K CI CO
        w_perm = (
            self.weight
            .to(device="cuda", dtype=torch.float16)
            .permute(2, 3, 4, 0, 1)
            .contiguous()
        )

        kernel = self._get_kernel(N, D, H, W, "float16")
        out_ndhwc = kernel(x_ndhwc, w_perm)

        out = out_ndhwc.permute(0, 4, 1, 2, 3).contiguous()

        if self.use_bias:
            out += self.bias.view(1, -1, 1, 1, 1).to(out.dtype)

        return out.to(orig_dtype)