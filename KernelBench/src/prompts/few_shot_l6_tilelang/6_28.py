"""
Problem Name: 28_Conv2d_Softmax_AvgPool_AvgPool_ResidualAdd_Tanh
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.33 runtime_stats={'mean': 4.33, 'std': 0.0302, 'min': 4.27, 'max': 4.38, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.6, 'std': 0.0316, 'min': 1.54, 'max': 1.73, 'num_trials': 100}, 'speedup_ratio': 0.37}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# -----------------------------------------------------------------------------
# Kernel factories
# -----------------------------------------------------------------------------

def _build_conv2d_naive_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    F: int,
    KH: int,
    KW: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
    block: int = 256,
):
    OH = H - KH + 1
    OW = W - KW + 1

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv_naive(
        X: T.Tensor((N, C, H, W), dtype),
        Wt: T.Tensor((F, C, KH, KW), dtype),
        B: T.Tensor((F,), dtype),
        Y: T.Tensor((N, F, OH, OW), dtype),
    ):
        total = N * F * OH * OW
        with T.Kernel(T.ceildiv(total, block), threads=block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < total:
                ow = idx % OW
                tmp = idx // OW
                oh = tmp % OH
                tmp //= OH
                f = tmp % F
                n = tmp // F

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = 0.0

                for c in range(C):
                    for kh in range(KH):
                        for kw in range(KW):
                            acc[0] += (
                                T.Cast(accum_dtype, X[n, c, oh + kh, ow + kw])
                                * T.Cast(accum_dtype, Wt[f, c, kh, kw])
                            )
                acc[0] += T.Cast(accum_dtype, B[f])
                Y[n, f, oh, ow] = T.Cast(dtype, acc[0])

    return conv_naive


def _build_channel_softmax_kernel(
    rows: int,
    cols: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def softmax_kernel(
        X: T.Tensor((rows, cols), dtype),
        Y: T.Tensor((rows, cols), dtype),
    ):
        with T.Kernel(rows, threads=1) as bx:
            max_val = T.alloc_local((1,), accum_dtype)
            sum_val = T.alloc_local((1,), accum_dtype)

            # max
            max_val[0] = T.Cast(accum_dtype, X[bx, 0])
            for j in range(1, cols):
                max_val[0] = T.max(
                    max_val[0], T.Cast(accum_dtype, X[bx, j])
                )

            # exp and sum
            sum_val[0] = 0.0
            for j in range(cols):
                e = T.exp(
                    T.Cast(accum_dtype, X[bx, j]) - max_val[0]
                )
                Y[bx, j] = T.Cast(dtype, e)
                sum_val[0] += e

            inv_sum = 1.0 / sum_val[0]

            # normalize
            for j in range(cols):
                Y[bx, j] = T.Cast(
                    dtype,
                    T.Cast(accum_dtype, Y[bx, j]) * inv_sum,
                )

    return softmax_kernel


def _build_avgpool2d_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    ps: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
    block: int = 256,
):
    Ho = H // ps
    Wo = W // ps
    total = N * C * Ho * Wo

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def avgpool_kernel(
        X: T.Tensor((N, C, H, W), dtype),
        Y: T.Tensor((N, C, Ho, Wo), dtype),
    ):
        with T.Kernel(T.ceildiv(total, block), threads=block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < total:
                wo = idx % Wo
                tmp = idx // Wo
                ho = tmp % Ho
                tmp //= Ho
                c = tmp % C
                n = tmp // C

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = 0.0
                for i in range(ps):
                    for j in range(ps):
                        acc[0] += T.Cast(
                            accum_dtype,
                            X[n, c, ho * ps + i, wo * ps + j],
                        )
                acc[0] = acc[0] / (ps * ps)
                Y[n, c, ho, wo] = T.Cast(dtype, acc[0])

    return avgpool_kernel


def _build_linear_kernel(
    M: int,
    N: int,
    K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),  # row-major
        B: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)
            acc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(acc)

            for ko in T.Pipelined(
                T.ceildiv(K, block_K), num_stages=num_stages
            ):
                T.copy(X[by * block_M, ko * block_K], X_s)
                T.copy(W[bx * block_N, ko * block_K], W_s)
                T.gemm(X_s, W_s, acc, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    v = acc[i, j] + T.Cast(accum_dtype, B[gn])
                    Y[gm, gn] = T.Cast(dtype, v)

    return linear_kernel


def _build_add_tanh_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
    block: int = 256,
):
    total = N * C * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def add_tanh_kernel(
        X: T.Tensor((N, C, H, W), dtype),
        R: T.Tensor((N, C), dtype),
        Y: T.Tensor((N, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(total, block), threads=block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < total:
                w = idx % W
                tmp = idx // W
                h = tmp % H
                tmp //= H
                c = tmp % C
                n = tmp // C

                val = (
                    T.Cast(accum_dtype, X[n, c, h, w])
                    + T.Cast(accum_dtype, R[n, c])
                )
                # tanh via exp
                two = T.Cast(accum_dtype, 2.0)
                exp2x = T.exp(val * two)
                tanh_val = (exp2x - 1.0) / (exp2x + 1.0)
                Y[n, c, h, w] = T.Cast(dtype, tanh_val)

    return add_tanh_kernel


# -----------------------------------------------------------------------------
# PyTorch wrapper
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        in_features: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = self.kernel_w = (
            kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        )
        self.pool_size = pool_size
        self.in_features = in_features

        # --- Conv parameters -------------------------------------------------
        self.conv_weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_h, self.kernel_w)
        )
        self.conv_bias = nn.Parameter(torch.empty(out_channels))

        torch.nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_channels * self.kernel_h * self.kernel_w)
        torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # --- Linear parameters ----------------------------------------------
        self.linear_weight = nn.Parameter(torch.empty(out_channels, in_features))
        self.linear_bias = nn.Parameter(torch.empty(out_channels))

        torch.nn.init.kaiming_uniform_(self.linear_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.linear_bias, -bound, bound)

        # --- kernel caches ---------------------------------------------------
        self._conv_kernels = {}
        self._softmax_kernels = {}
        self._avgpool_kernels = {}
        self._linear_kernels = {}
        self._add_tanh_kernels = {}

    # ----------------------------------------------------------------------
    # kernel getters
    # ----------------------------------------------------------------------
    def _get_conv_kernel(self, N: int, H: int, W: int):
        key = (N, H, W)
        if key not in self._conv_kernels:
            self._conv_kernels[key] = _build_conv2d_naive_kernel(
                N,
                self.in_channels,
                H,
                W,
                self.out_channels,
                self.kernel_h,
                self.kernel_w,
            )
        return self._conv_kernels[key]

    def _get_softmax_kernel(self, rows: int):
        key = (rows,)
        if key not in self._softmax_kernels:
            self._softmax_kernels[key] = _build_channel_softmax_kernel(
                rows,
                self.out_channels,
            )
        return self._softmax_kernels[key]

    def _get_avgpool_kernel(self, N: int, H: int, W: int):
        key = (N, H, W)
        if key not in self._avgpool_kernels:
            self._avgpool_kernels[key] = _build_avgpool2d_kernel(
                N,
                self.out_channels,
                H,
                W,
                self.pool_size,
            )
        return self._avgpool_kernels[key]

    def _get_linear_kernel(self, M: int):
        key = (M,)
        if key not in self._linear_kernels:
            self._linear_kernels[key] = _build_linear_kernel(
                M,
                self.out_channels,
                self.in_features,
            )
        return self._linear_kernels[key]

    def _get_add_tanh_kernel(self, N: int, H: int, W: int):
        key = (N, H, W)
        if key not in self._add_tanh_kernels:
            self._add_tanh_kernels[key] = _build_add_tanh_kernel(
                N,
                self.out_channels,
                H,
                W,
            )
        return self._add_tanh_kernels[key]

    # ----------------------------------------------------------------------
    # forward
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        N, C_in, H, W = x_f16.shape

        # ------------------------------------------------------------------
        # Conv2d -----------------------------------------------------------
        conv_kernel = self._get_conv_kernel(N, H, W)
        w_f16 = self.conv_weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.conv_bias.to(device="cuda", dtype=torch.float16)
        x_conv = conv_kernel(x_f16, w_f16, b_f16)

        # ------------------------------------------------------------------
        # Softmax (channel-wise) ------------------------------------------
        N_, C_out, H1, W1 = x_conv.shape
        rows_soft = N_ * H1 * W1
        softmax_kernel = self._get_softmax_kernel(rows_soft)

        x_conv_perm = (
            x_conv.permute(0, 2, 3, 1).contiguous().view(rows_soft, C_out)
        )
        x_soft = softmax_kernel(x_conv_perm).view(
            N_, H1, W1, C_out
        ).permute(0, 3, 1, 2).contiguous()

        # ------------------------------------------------------------------
        # AvgPool 1 --------------------------------------------------------
        pool1_kernel = self._get_avgpool_kernel(N_, H1, W1)
        x_pool1 = pool1_kernel(x_soft)

        # ------------------------------------------------------------------
        # AvgPool 2 --------------------------------------------------------
        _, _, H2, W2 = x_pool1.shape
        pool2_kernel = self._get_avgpool_kernel(N_, H2, W2)
        x_pool2 = pool2_kernel(x_pool1)

        # ------------------------------------------------------------------
        # Residual path (flatten + linear) ---------------------------------
        res_in = x_f16.view(N, -1).contiguous()
        lin_kernel = self._get_linear_kernel(N)
        lw_f16 = self.linear_weight.to(device="cuda", dtype=torch.float16)
        lb_f16 = self.linear_bias.to(device="cuda", dtype=torch.float16)
        res_vec = lin_kernel(res_in, lw_f16, lb_f16)  # (N, out_channels)

        # ------------------------------------------------------------------
        # Add & Tanh -------------------------------------------------------
        _, _, H3, W3 = x_pool2.shape
        add_tanh_kernel = self._get_add_tanh_kernel(N, H3, W3)
        out_f16 = add_tanh_kernel(x_pool2, res_vec)

        return out_f16.to(orig_dtype)