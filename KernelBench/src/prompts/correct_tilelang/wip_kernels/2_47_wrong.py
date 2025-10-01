import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def conv3d_kernel(
    N: int,
    Ci: int,
    D: int,
    H: int,
    W: int,
    Co: int,
    K: int,
    stride: int,
    padding: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
):
    KD, KH, KW = K, K, K
    OD = (D + 2 * padding - KD) // stride + 1
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    M_dim = N * OD * OH * OW
    K_dim = KD * KH * KW * Ci

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((N, Ci, D, H, W), dtype),
        Wt: T.Tensor((Co, Ci, KD, KH, KW), dtype),
        B: T.Tensor((Co,), dtype),
        Out: T.Tensor((N, Co, OD, OH, OW), dtype),
    ):
        with T.Kernel(
            T.ceildiv(Co, block_N), T.ceildiv(M_dim, block_M), threads=128
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            W_flat = T.Tensor((K_dim, Co), dtype, Wt.data)
            Out_flat = T.Tensor((M_dim, Co), dtype, Out.data)

            T.clear(C_local)
            for k_iter in T.Pipelined(T.ceildiv(K_dim, block_K), num_stages=2):
                for i, j in T.Parallel(block_M, block_K):
                    k_idx = k_iter * block_K + j
                    m_idx = by * block_M + i

                    ow = m_idx % OW
                    tmp1 = m_idx // OW
                    oh = tmp1 % OH
                    tmp2 = tmp1 // OH
                    od = tmp2 % OD
                    n_idx = tmp2 // OD

                    kd_idx = k_idx // (KH * KW * Ci)
                    rem1 = k_idx % (KH * KW * Ci)
                    kh_idx = rem1 // (KW * Ci)
                    rem2 = rem1 % (KW * Ci)
                    kw_idx = rem2 // Ci
                    c_idx = rem2 % Ci

                    ad = od * stride + kd_idx - padding
                    ah = oh * stride + kh_idx - padding
                    aw = ow * stride + kw_idx - padding

                    in_bound = (
                        (ad >= 0)
                        and (ah >= 0)
                        and (aw >= 0)
                        and (ad < D)
                        and (ah < H)
                        and (aw < W)
                    )
                    A_shared[i, j] = T.if_then_else(
                        in_bound, A[n_idx, c_idx, ad, ah, aw], 0
                    )

                T.copy(W_flat[k_iter * block_K, bx * block_N], W_shared)
                T.gemm(A_shared, W_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                global_j = bx * block_N + j
                C_shared[i, j] = C_local[i, j] + B[global_j]

            T.copy(C_shared, Out_flat[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(
                out_channels, in_channels, kernel_size, kernel_size, kernel_size
            )
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size ** 3
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self._cached_kernels = {}

    def _get_kernel(self, N, D, H, W, dtype):
        key = (N, D, H, W, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = conv3d_kernel(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                dtype=dtype,
            )
        return self._cached_kernels[key]

    def forward(self, x):
        x = x.to(device="cuda", dtype=torch.float16)
        w = self.weight.to(device="cuda", dtype=torch.float16)
        b = self.bias.to(device="cuda", dtype=torch.float16)

        N, Ci, D, H, W = x.shape
        kernel = self._get_kernel(N, D, H, W, dtype="float16")
        out = kernel(x, w, b)

        softplus = torch.log1p(torch.exp(out))
        mish_out = out * torch.tanh(softplus)
        return torch.tanh(mish_out)