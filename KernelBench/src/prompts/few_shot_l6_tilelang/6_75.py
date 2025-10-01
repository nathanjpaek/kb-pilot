"""
Problem Name: 75_Conv3d_InstanceNorm_MaxPool_BatchNorm_LogSumExp
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=54.3 runtime_stats={'mean': 54.3, 'std': 0.17, 'min': 53.9, 'max': 54.8, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.85, 'std': 0.0213, 'min': 2.83, 'max': 3.04, 'num_trials': 100}, 'speedup_ratio': 0.0525}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ------------------------------------------------------------
# Kernel builders
# ------------------------------------------------------------
def _build_conv3d_kernel(N, Cin, Din, Hin, Win, Cout, K,
                         block: int = 128,
                         dtype: str = "float16",
                         accum_dtype: str = "float"):
    Dout = Din - K + 1
    Hout = Hin - K + 1
    Wout = Win - K + 1
    numel = N * Cout * Dout * Hin - Hin + Hin  # dummy replace to use numel later
    numel = N * Cout * Dout * Hout * Wout

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv3d_kernel(
        X: T.Tensor((N, Cin, Din, Hin, Win), dtype),
        Wt: T.Tensor((Cout, Cin, K, K, K), dtype),
        B: T.Tensor((Cout,), dtype),
        Y: T.Tensor((N, Cout, Dout, Hout, Wout), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block), threads=block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < numel:
                tmp = idx
                w_out = tmp % Wout
                tmp //= Wout
                h_out = tmp % Hout
                tmp //= Hout
                d_out = tmp % Dout
                tmp //= Dout
                c_out = tmp % Cout
                n = tmp // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = 0.0
                for ci in T.serial(Cin):
                    for kd in T.serial(K):
                        for kh in T.serial(K):
                            for kw in T.serial(K):
                                val_in = X[n, ci, d_out + kd, h_out + kh, w_out + kw].astype(accum_dtype)
                                val_w = Wt[c_out, ci, kd, kh, kw].astype(accum_dtype)
                                acc[0] += val_in * val_w
                acc[0] = acc[0] + B[c_out].astype(accum_dtype)
                Y[n, c_out, d_out, h_out, w_out] = T.Cast(dtype, acc[0])

    return conv3d_kernel


def _build_instancenorm3d_kernel(N, C, D, H, W,
                                 eps: float = 1e-5,
                                 block: int = 128,
                                 dtype: str = "float16",
                                 accum_dtype: str = "float"):
    size = D * H * W
    inv_size = 1.0 / size

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def instnorm_kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, C, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(N * C, block), threads=block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < N * C:
                n = idx // C
                c = idx % C
                sum_val = T.alloc_local((1,), accum_dtype)
                sum_sq = T.alloc_local((1,), accum_dtype)
                sum_val[0] = 0.0
                sum_sq[0] = 0.0
                for d in T.serial(D):
                    for h in T.serial(H):
                        for w in T.serial(W):
                            v = X[n, c, d, h, w].astype(accum_dtype)
                            sum_val[0] += v
                            sum_sq[0] += v * v
                mean = sum_val[0] * inv_size
                var = sum_sq[0] * inv_size - mean * mean
                inv_std = 1.0 / T.sqrt(var + eps)
                for d in T.serial(D):
                    for h in T.serial(H):
                        for w in T.serial(W):
                            v = X[n, c, d, h, w].astype(accum_dtype)
                            norm = (v - mean) * inv_std
                            Y[n, c, d, h, w] = T.Cast(dtype, norm)

    return instnorm_kernel


def _build_maxpool3d_kernel(N, C, D, H, W, P,
                            block: int = 128,
                            dtype: str = "float16"):
    Dp = (D - P) // P + 1
    Hp = (H - P) // P + 1
    Wp = (W - P) // P + 1
    numel = N * C * Dp * Hp * Wp

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def maxpool_kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, C, Dp, Hp, Wp), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block), threads=block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < numel:
                tmp = idx
                wp = tmp % Wp
                tmp //= Wp
                hp = tmp % Hp
                tmp //= Hp
                dp = tmp % Dp
                tmp //= Dp
                c = tmp % C
                n = tmp // C
                max_val = X[n, c, dp * P, hp * P, wp * P]
                for kd in T.serial(P):
                    for kh in T.serial(P):
                        for kw in T.serial(P):
                            v = X[n, c,
                                  dp * P + kd,
                                  hp * P + kh,
                                  wp * P + kw]
                            max_val = T.max(max_val, v)
                Y[n, c, dp, hp, wp] = max_val

    return maxpool_kernel


def _build_batchnorm3d_kernel(N, C, D, H, W,
                              eps: float = 1e-5,
                              block: int = 128,
                              dtype: str = "float16",
                              accum_dtype: str = "float"):
    size = N * D * H * W
    inv_size = 1.0 / size

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def bn_kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        gamma: T.Tensor((C,), dtype),
        beta: T.Tensor((C,), dtype),
        Y: T.Tensor((N, C, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(C, block), threads=block) as bx:
            tx = T.get_thread_binding(0)
            c = bx * block + tx
            if c < C:
                sum_val = T.alloc_local((1,), accum_dtype)
                sum_sq = T.alloc_local((1,), accum_dtype)
                sum_val[0] = 0.0
                sum_sq[0] = 0.0
                for n in T.serial(N):
                    for d in T.serial(D):
                        for h in T.serial(H):
                            for w in T.serial(W):
                                v = X[n, c, d, h, w].astype(accum_dtype)
                                sum_val[0] += v
                                sum_sq[0] += v * v
                mean = sum_val[0] * inv_size
                var = sum_sq[0] * inv_size - mean * mean
                inv_std = 1.0 / T.sqrt(var + eps)
                for n in T.serial(N):
                    for d in T.serial(D):
                        for h in T.serial(H):
                            for w in T.serial(W):
                                v = X[n, c, d, h, w].astype(accum_dtype)
                                norm = (v - mean) * inv_std
                                res = norm * gamma[c].astype(accum_dtype) + beta[c].astype(accum_dtype)
                                Y[n, c, d, h, w] = T.Cast(dtype, res)

    return bn_kernel


def _build_logsumexp_kernel(N, C, D, H, W,
                            block: int = 128,
                            dtype: str = "float16",
                            accum_dtype: str = "float"):
    DHW = D * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def lse_kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, C), dtype),
    ):
        with T.Kernel(T.ceildiv(N * C, block), threads=block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < N * C:
                n = idx // C
                c = idx % C
                max_val = X[n, c, 0, 0, 0].astype(accum_dtype)
                for d in T.serial(D):
                    for h in T.serial(H):
                        for w in T.serial(W):
                            v = X[n, c, d, h, w].astype(accum_dtype)
                            max_val = T.max(max_val, v)
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = 0.0
                for d in T.serial(D):
                    for h in T.serial(H):
                        for w in T.serial(W):
                            v = X[n, c, d, h, w].astype(accum_dtype)
                            sum_exp[0] += T.exp(v - max_val)
                Y[n, c] = T.Cast(dtype, max_val + T.log(sum_exp[0]))

    return lse_kernel


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel, num_features):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.k = int(kernel_size)
        self.p = int(pool_kernel)
        self.num_features = int(num_features)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.k, self.k, self.k))
        self.bias = nn.Parameter(torch.empty(out_channels))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_channels * self.k ** 3)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self.bn_weight = nn.Parameter(torch.ones(num_features))
        self.bn_bias = nn.Parameter(torch.zeros(num_features))

        self._conv_kernels = {}
        self._inst_kernels = {}
        self._pool_kernels = {}
        self._bn_kernels = {}
        self._lse_kernels = {}

    def _get_conv(self, N, D, H, W, dtype):
        key = (N, D, H, W, dtype)
        if key not in self._conv_kernels:
            self._conv_kernels[key] = _build_conv3d_kernel(
                N, self.in_channels, D, H, W,
                self.out_channels, self.k
            )
        return self._conv_kernels[key]

    def _get_inst(self, N, D, H, W, dtype):
        key = (N, D, H, W, dtype)
        if key not in self._inst_kernels:
            self._inst_kernels[key] = _build_instancenorm3d_kernel(
                N, self.out_channels, D, H, W
            )
        return self._inst_kernels[key]

    def _get_pool(self, N, D, H, W, dtype):
        key = (N, D, H, W, dtype)
        if key not in self._pool_kernels:
            self._pool_kernels[key] = _build_maxpool3d_kernel(
                N, self.out_channels, D, H, W, self.p
            )
        return self._pool_kernels[key]

    def _get_bn(self, N, D, H, W, dtype):
        key = (N, D, H, W, dtype)
        if key not in self._bn_kernels:
            self._bn_kernels[key] = _build_batchnorm3d_kernel(
                N, self.out_channels, D, H, W
            )
        return self._bn_kernels[key]

    def _get_lse(self, N, D, H, W, dtype):
        key = (N, D, H, W, dtype)
        if key not in self._lse_kernels:
            self._lse_kernels[key] = _build_logsumexp_kernel(
                N, self.out_channels, D, H, W
            )
        return self._lse_kernels[key]

    def forward(self, x):
        x = x.to(device="cuda", dtype=torch.float16).contiguous()
        w = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b = self.bias.to(device="cuda", dtype=torch.float16).contiguous()
        gamma = self.bn_weight.to(device="cuda", dtype=torch.float16).contiguous()
        beta = self.bn_bias.to(device="cuda", dtype=torch.float16).contiguous()

        N, Cin, D, H, W = x.shape

        conv_kernel = self._get_conv(N, D, H, W, x.dtype)
        y = conv_kernel(x, w, b)

        _, _, D1, H1, W1 = y.shape
        inst_kernel = self._get_inst(N, D1, H1, W1, y.dtype)
        y = inst_kernel(y)

        pool_kernel = self._get_pool(N, D1, H1, W1, y.dtype)
        y = pool_kernel(y)

        _, _, D2, H2, W2 = y.shape
        bn_kernel = self._get_bn(N, D2, H2, W2, y.dtype)
        y = bn_kernel(y, gamma, beta)

        lse_kernel = self._get_lse(N, D2, H2, W2, y.dtype)
        y = lse_kernel(y)

        return y.to(torch.float32)