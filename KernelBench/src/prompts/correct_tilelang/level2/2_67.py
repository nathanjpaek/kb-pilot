"""
Problem Name: 67_Conv2d_GELU_GlobalAvgPool
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.149 runtime_stats={'mean': 0.149, 'std': 0.0418, 'min': 0.124, 'max': 0.318, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0953, 'std': 0.0568, 'min': 0.0549, 'max': 0.25, 'num_trials': 100}, 'speedup_ratio': 0.64}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_conv_gelu_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    F: int,
    KH: int,
    KW: int,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    OH = H - KH + 1
    OW = W - KW + 1
    K_TOTAL = KH * KW * C
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv_gelu(
        x: T.Tensor((N, H, W, C), dtype),
        w: T.Tensor((KH, KW, C, F), dtype),
        b: T.Tensor((F,), dtype),
        out: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),
            T.ceildiv(N * OH * OW, block_M),
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_loc)

            w_flat = T.Tensor((K_TOTAL, F), dtype, w.data)

            for ko in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=2):
                # Load A tile (im2col) to shared
                for ti, tk in T.Parallel(block_M, block_K):
                    m_idx = by * block_M + ti
                    k_idx = ko * block_K + tk

                    cond = (m_idx < N * OH * OW) and (k_idx < K_TOTAL)

                    if cond:
                        n_idx = m_idx // (OH * OW)
                        rem1 = m_idx % (OH * OW)
                        oh_idx = rem1 // OW
                        ow_idx = rem1 % OW

                        kh_idx = k_idx // (KW * C)
                        rem2 = k_idx % (KW * C)
                        kw_idx = rem2 // C
                        c_idx = rem2 % C

                        h_in = oh_idx + kh_idx
                        w_in = ow_idx + kw_idx

                        A_s[ti, tk] = x[n_idx, h_in, w_in, c_idx]
                    else:
                        A_s[ti, tk] = T.Cast(dtype, 0)

                # Load B tile (weights) to shared
                T.copy(w_flat[ko * block_K, bx * block_N], B_s)

                # GEMM
                T.gemm(A_s, B_s, C_loc)

            # Write results with bias + GELU
            for ti, tj in T.Parallel(block_M, block_N):
                m_idx = by * block_M + ti
                f_idx = bx * block_N + tj
                if (m_idx < N * OH * OW) and (f_idx < F):
                    n_idx = m_idx // (OH * OW)
                    rem1 = m_idx % (OH * OW)
                    oh_idx = rem1 // OW
                    ow_idx = rem1 % OW

                    val = C_loc[ti, tj] + b[f_idx].astype(accum_dtype)
                    gelu_val = (
                        0.5
                        * val
                        * (
                            1.0
                            + T.erf(val * T.Cast(accum_dtype, inv_sqrt2))
                        )
                    )
                    out[n_idx, oh_idx, ow_idx, f_idx] = T.Cast(dtype, gelu_val)

    return conv_gelu


def build_gap_kernel(
    N: int,
    OH: int,
    OW: int,
    F: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    inv_area = 1.0 / float(OH * OW)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gap(
        inp: T.Tensor((N, OH, OW, F), dtype),
        out: T.Tensor((N, F), dtype),
    ):
        with T.Kernel(F, N, threads=1) as (fx, nx):
            acc = T.alloc_local((1,), accum_dtype)
            acc[0] = T.Cast(accum_dtype, 0)

            for h in range(OH):
                for w in range(OW):
                    acc[0] += inp[nx, h, w, fx].astype(accum_dtype)

            out[nx, fx] = T.Cast(dtype, acc[0] * inv_area)

    return gap


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.C = in_channels
        self.F = out_channels
        self.KH = kernel_size
        self.KW = kernel_size

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(out_channels))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Kernel caches
        self._conv_cache = {}
        self._gap_cache = {}

    def _get_conv_kernel(self, N, H, W, dtype):
        key = (N, self.C, H, W, self.F, self.KH, self.KW, dtype)
        if key not in self._conv_cache:
            self._conv_cache[key] = build_conv_gelu_kernel(
                N,
                self.C,
                H,
                W,
                self.F,
                self.KH,
                self.KW,
            )
        return self._conv_cache[key]

    def _get_gap_kernel(self, N, OH, OW, dtype):
        key = (N, OH, OW, self.F, dtype)
        if key not in self._gap_cache:
            self._gap_cache[key] = build_gap_kernel(
                N,
                OH,
                OW,
                self.F,
            )
        return self._gap_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = x.shape
        assert C == self.C, "in_channels mismatch"

        # Reorder to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w_fp16 = (
            self.weight.to(device="cuda", dtype=torch.float16)
            .permute(2, 3, 1, 0)
            .contiguous()
        )
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        conv_kernel = self._get_conv_kernel(N, H, W, "float16")
        out_feat = conv_kernel(x_nhwc, w_fp16, b_fp16)

        OH = H - self.KH + 1
        OW = W - self.KW + 1

        gap_kernel = self._get_gap_kernel(N, OH, OW, "float16")
        out_pooled = gap_kernel(out_feat)

        return out_pooled.to(orig_dtype)