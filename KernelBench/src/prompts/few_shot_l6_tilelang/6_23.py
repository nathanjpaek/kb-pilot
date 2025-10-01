"""
Problem Name: 23_Conv3d_Tanh_Clamp_GELU
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=14.5 runtime_stats={'mean': 14.5, 'std': 0.0325, 'min': 14.5, 'max': 14.7, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.95, 'std': 0.0307, 'min': 1.9, 'max': 2.05, 'num_trials': 100}, 'speedup_ratio': 0.134}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_conv3d_fused_kernel(
    N,
    Cin,
    Din,
    Hin,
    Win,
    Cout,
    Kd,
    Kh,
    Kw,
    clamp_min,
    clamp_max,
    dtype="float16",
    accum_dtype="float",
    block_M=128,
    block_N=64,
    block_K=32,
    num_stages=2,
):
    Dout = Din - Kd + 1
    Hout = Hin - Kh + 1
    Wout = Win - Kw + 1
    K_TOTAL = Cin * Kd * Kh * Kw

    cmn = float(clamp_min)
    cmx = float(clamp_max)
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    gelu_coeff = 0.044715

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, Din, Hin, Win, Cin), dtype),
        W: T.Tensor((Kd, Kh, Kw, Cin, Cout), dtype),
        Out: T.Tensor((N, Dout, Hout, Wout, Cout), dtype),
    ):
        with T.Kernel(
            T.ceildiv(Cout, block_N),
            T.ceildiv(N * Dout * Hout * Wout, block_M),
            threads=128,
        ) as (bx, by):
            A_sh = T.alloc_shared((block_M, block_K), dtype)
            B_sh = T.alloc_shared((block_K, block_N), dtype)
            Acc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(Acc)

            W_flat = T.Tensor((K_TOTAL, Cout), dtype, W.data)
            Out_flat = T.Tensor((N * Dout * Hout * Wout, Cout), dtype, Out.data)

            for ko in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                for i, j in T.Parallel(block_M, block_K):
                    k = ko * block_K + j
                    m = by * block_M + i

                    batch_idx = m // (Dout * Hout * Wout)
                    rem = m % (Dout * Hout * Wout)
                    od = rem // (Hout * Wout)
                    rem2 = rem % (Hout * Wout)
                    oh = rem2 // Wout
                    ow = rem2 % Wout

                    kd_idx = k // (Kh * Kw * Cin)
                    kh_idx = (k // (Kw * Cin)) % Kh
                    kw_idx = (k // Cin) % Kw
                    c_idx = k % Cin

                    id_idx = od + kd_idx
                    ih_idx = oh + kh_idx
                    iw_idx = ow + kw_idx

                    in_bound = (
                        (k < K_TOTAL)
                        and (m < N * Dout * Hout * Wout)
                        and (id_idx < Din)
                        and (ih_idx < Hin)
                        and (iw_idx < Win)
                    )

                    A_sh[i, j] = T.if_then_else(
                        in_bound,
                        X[batch_idx, id_idx, ih_idx, iw_idx, c_idx],
                        T.cast(0, dtype),
                    )

                T.copy(W_flat[ko * block_K, bx * block_N], B_sh)
                T.gemm(A_sh, B_sh, Acc)

            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if (m < N * Dout * Hout * Wout) and (n < Cout):
                    v = Acc[i, j]
                    v = T.tanh(v)
                    v = T.clamp(v, cmn, cmx)
                    tmp = v + gelu_coeff * v * v * v
                    tmp = tmp * sqrt_2_over_pi
                    gelu = 0.5 * v * (1 + T.tanh(tmp))
                    Out_flat[m, n] = T.Cast(dtype, gelu)

    return kernel


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
            )
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * (self.kernel_size ** 3)
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(out_channels))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self._kernel_cache = {}

    def _get_kernel(self, N, D, H, W, dtype):
        key = (N, D, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_conv3d_fused_kernel(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
                self.clamp_min,
                self.clamp_max,
                dtype="float16",
            )
        return self._kernel_cache[key]

    def forward(self, x):
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_f16.shape

        w_f16 = (
            self.weight.to(device="cuda", dtype=torch.float16)
            .permute(2, 3, 4, 1, 0)
            .contiguous()
        )
        x_perm = x_f16.permute(0, 2, 3, 4, 1).contiguous()

        kernel = self._get_kernel(N, D, H, W, x_f16.dtype)
        out_perm = kernel(x_perm, w_f16)
        out_f16 = out_perm.permute(0, 4, 1, 2, 3).contiguous()

        # add bias
        out_f16 += self.bias.to(device="cuda", dtype=torch.float16).view(1, -1, 1, 1, 1)

        return out_f16.to(orig_dtype)