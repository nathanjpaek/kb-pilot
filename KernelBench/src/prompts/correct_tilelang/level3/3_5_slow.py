"""
Problem Name: 5_AlexNet
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=28.4 runtime_stats={'mean': 28.4, 'std': 2.44, 'min': 25.4, 'max': 41.3, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.475, 'std': 0.0265, 'min': 0.429, 'max': 0.553, 'num_trials': 100}, 'speedup_ratio': 0.0167}}
"""

import math
from typing import Dict, Tuple

import torch
from torch import nn

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------------- #
#                               utility helpers                                 #
# ----------------------------------------------------------------------------- #
def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ----------------------------------------------------------------------------- #
#                 2-D convolution + bias + ReLU  (NCHW, strided)                #
# ----------------------------------------------------------------------------- #
def _build_conv_relu(
    N: int,
    C_in: int,
    C_out: int,
    H_in: int,
    W_in: int,
    k: int,
    stride: int,
    pad: int,
    dtype: str = "float16",
    accum_dtype: str = "float32",
    threads_per_block: int = 256,
):
    H_out = (H_in + 2 * pad - k) // stride + 1
    W_out = (W_in + 2 * pad - k) // stride + 1
    total = N * C_out * H_out * W_out
    grid = _ceildiv(total, threads_per_block)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv_relu(
        X: T.Tensor((N, C_in, H_in, W_in), dtype),
        W: T.Tensor((C_out, C_in, k, k), dtype),
        B: T.Tensor((C_out,), dtype),
        Y: T.Tensor((N, C_out, H_out, W_out), dtype),
    ):
        zero = T.Cast(accum_dtype, 0.0)

        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < total:
                ow = idx % W_out
                tmp1 = idx // W_out
                oh = tmp1 % H_out
                tmp2 = tmp1 // H_out
                oc = tmp2 % C_out
                n  = tmp2 // C_out

                acc = T.Cast(accum_dtype, B[oc])

                for ic in T.serial(C_in):
                    for kh in T.serial(k):
                        ih = oh * stride - pad + kh
                        if ih >= 0 and ih < H_in:
                            for kw in T.serial(k):
                                iw = ow * stride - pad + kw
                                if iw >= 0 and iw < W_in:
                                    acc += (
                                        T.Cast(accum_dtype, X[n, ic, ih, iw])
                                        * T.Cast(accum_dtype, W[oc, ic, kh, kw])
                                    )

                # ReLU
                acc = T.max(acc, zero)
                Y[n, oc, oh, ow] = T.Cast(dtype, acc)

    return conv_relu, (H_out, W_out)


# ----------------------------------------------------------------------------- #
#                          2-D max-pool (NCHW)                                  #
# ----------------------------------------------------------------------------- #
def _build_maxpool(
    N: int,
    C: int,
    H_in: int,
    W_in: int,
    k: int,
    stride: int,
    dtype: str = "float16",
    accum_dtype: str = "float32",
    threads_per_block: int = 256,
):
    H_out = (H_in - k) // stride + 1
    W_out = (W_in - k) // stride + 1
    total = N * C * H_out * W_out
    grid = _ceildiv(total, threads_per_block)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def maxpool(
        X: T.Tensor((N, C, H_in, W_in), dtype),
        Y: T.Tensor((N, C, H_out, W_out), dtype),
    ):
        minus_inf = T.Cast(accum_dtype, -1.0e30)

        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < total:
                ow = idx % W_out
                tmp1 = idx // W_out
                oh = tmp1 % H_out
                tmp2 = tmp1 // H_out
                c = tmp2 % C
                n = tmp2 // C

                best = T.alloc_local((1,), accum_dtype)
                best[0] = minus_inf

                for kh in T.serial(k):
                    ih = oh * stride + kh
                    for kw in T.serial(k):
                        iw = ow * stride + kw
                        val = T.Cast(accum_dtype, X[n, c, ih, iw])
                        if val > best[0]:
                            best[0] = val

                Y[n, c, oh, ow] = T.Cast(dtype, best[0])

    return maxpool, (H_out, W_out)


# ----------------------------------------------------------------------------- #
#                     flatten (N,C,H,W)  →  (N, C·H·W)                          #
# ----------------------------------------------------------------------------- #
def _build_flatten(
    N: int, C: int, H: int, W: int, dtype: str = "float16", threads_per_block: int = 256
):
    K = C * H * W
    total = N * K
    grid = _ceildiv(total, threads_per_block)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def flatten(
        X: T.Tensor((N, C, H, W), dtype),
        Y: T.Tensor((N, K), dtype),
    ):
        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < total:
                k = idx % K
                n = idx // K

                c = k // (H * W)
                rem = k % (H * W)
                h = rem // W
                w = rem % W
                Y[n, k] = X[n, c, h, w]

    return flatten, K


# ----------------------------------------------------------------------------- #
#              linear (GEMM) + bias + ReLU   :  (N, in) → (N, out)              #
# ----------------------------------------------------------------------------- #
def _build_linear_relu(
    N: int,
    in_features: int,
    out_features: int,
    dtype: str = "float16",
    accum_dtype: str = "float32",
    threads_per_block: int = 256,
):
    total = N * out_features
    grid = _ceildiv(total, threads_per_block)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_relu(
        X: T.Tensor((N, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        Y: T.Tensor((N, out_features), dtype),
    ):
        zero = T.Cast(accum_dtype, 0.0)

        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < total:
                o = idx % out_features
                n = idx // out_features

                acc = T.Cast(accum_dtype, B[o])
                for i in T.serial(in_features):
                    acc += (
                        T.Cast(accum_dtype, X[n, i])
                        * T.Cast(accum_dtype, W[o, i])
                    )
                acc = T.max(acc, zero)  # ReLU
                Y[n, o] = T.Cast(dtype, acc)

    return linear_relu


# ----------------------------------------------------------------------------- #
#                                ModelNew                                       #
# ----------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Fully TileLang-accelerated AlexNet-style network
    (Conv+ReLU+Pool ×3, Conv+ReLU ×2, Flatten, Linear+ReLU ×2, Linear).
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # ---------------- convolution parameters ---------------- #
        self.conv1_w = nn.Parameter(torch.empty(96, 3, 11, 11))
        self.conv1_b = nn.Parameter(torch.empty(96))
        self.conv2_w = nn.Parameter(torch.empty(256, 96, 5, 5))
        self.conv2_b = nn.Parameter(torch.empty(256))
        self.conv3_w = nn.Parameter(torch.empty(384, 256, 3, 3))
        self.conv3_b = nn.Parameter(torch.empty(384))
        self.conv4_w = nn.Parameter(torch.empty(384, 384, 3, 3))
        self.conv4_b = nn.Parameter(torch.empty(384))
        self.conv5_w = nn.Parameter(torch.empty(256, 384, 3, 3))
        self.conv5_b = nn.Parameter(torch.empty(256))

        # ---------------- linear parameters --------------------- #
        self.fc1_w = nn.Parameter(torch.empty(4096, 256 * 6 * 6))
        self.fc1_b = nn.Parameter(torch.empty(4096))
        self.fc2_w = nn.Parameter(torch.empty(4096, 4096))
        self.fc2_b = nn.Parameter(torch.empty(4096))
        self.fc3_w = nn.Parameter(torch.empty(num_classes, 4096))
        self.fc3_b = nn.Parameter(torch.empty(num_classes))

        # ---------------- parameter init (same as PyTorch) ------- #
        for w in [
            self.conv1_w,
            self.conv2_w,
            self.conv3_w,
            self.conv4_w,
            self.conv5_w,
        ]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        for b in [
            self.conv1_b,
            self.conv2_b,
            self.conv3_b,
            self.conv4_b,
            self.conv5_b,
        ]:
            fan_in = b.shape[0]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

        nn.init.kaiming_uniform_(self.fc1_w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2_w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc3_w, a=math.sqrt(5))

        for b in [self.fc1_b, self.fc2_b, self.fc3_b]:
            fan_in = b.shape[0]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

        # ---------------- kernel caches -------------------------- #
        self._conv_cache: Dict[Tuple, Tuple[callable, Tuple[int, int]]] = {}
        self._pool_cache: Dict[Tuple, Tuple[callable, Tuple[int, int]]] = {}
        self._flat_cache: Dict[Tuple, Tuple[callable, int]] = {}
        self._lin_cache: Dict[Tuple, callable] = {}

    # ----------------------------------------------------------------- #
    #                       kernel retrieval helpers                    #
    # ----------------------------------------------------------------- #
    def _get_conv(self, N, C_in, C_out, H, W, k, stride, pad, dtype):
        key = (N, C_in, C_out, H, W, k, stride, pad, dtype)
        if key not in self._conv_cache:
            self._conv_cache[key] = _build_conv_relu(
                N, C_in, C_out, H, W, k, stride, pad, dtype=dtype
            )
        return self._conv_cache[key]

    def _get_pool(self, N, C, H, W, k, stride, dtype):
        key = (N, C, H, W, k, stride, dtype)
        if key not in self._pool_cache:
            self._pool_cache[key] = _build_maxpool(
                N, C, H, W, k, stride, dtype=dtype
            )
        return self._pool_cache[key]

    def _get_flat(self, N, C, H, W, dtype):
        key = (N, C, H, W, dtype)
        if key not in self._flat_cache:
            self._flat_cache[key] = _build_flatten(N, C, H, W, dtype=dtype)
        return self._flat_cache[key]

    def _get_linear(self, N, in_feat, out_feat, dtype):
        key = (N, in_feat, out_feat, dtype)
        if key not in self._lin_cache:
            self._lin_cache[key] = _build_linear_relu(
                N, in_feat, out_feat, dtype=dtype
            )
        return self._lin_cache[key]

    # ----------------------------------------------------------------- #
    #                               forward                             #
    # ----------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, _, H, W = x.shape

        # ---------------- Conv1 + ReLU + MaxPool ---------------- #
        conv1, (H1, W1) = self._get_conv(
            N, 3, 96, H, W, 11, 4, 2, "float16"
        )
        x = conv1(
            x,
            self.conv1_w.to(device="cuda", dtype=torch.float16),
            self.conv1_b.to(device="cuda", dtype=torch.float16),
        )

        pool1, (H1p, W1p) = self._get_pool(N, 96, H1, W1, 3, 2, "float16")
        x = pool1(x)

        # ---------------- Conv2 + ReLU + MaxPool ---------------- #
        conv2, (H2, W2) = self._get_conv(
            N, 96, 256, H1p, W1p, 5, 1, 2, "float16"
        )
        x = conv2(
            x,
            self.conv2_w.to(device="cuda", dtype=torch.float16),
            self.conv2_b.to(device="cuda", dtype=torch.float16),
        )

        pool2, (H2p, W2p) = self._get_pool(N, 256, H2, W2, 3, 2, "float16")
        x = pool2(x)

        # ---------------- Conv3 + ReLU --------------------------- #
        conv3, (H3, W3) = self._get_conv(
            N, 256, 384, H2p, W2p, 3, 1, 1, "float16"
        )
        x = conv3(
            x,
            self.conv3_w.to(device="cuda", dtype=torch.float16),
            self.conv3_b.to(device="cuda", dtype=torch.float16),
        )

        # ---------------- Conv4 + ReLU --------------------------- #
        conv4, (H4, W4) = self._get_conv(
            N, 384, 384, H3, W3, 3, 1, 1, "float16"
        )
        x = conv4(
            x,
            self.conv4_w.to(device="cuda", dtype=torch.float16),
            self.conv4_b.to(device="cuda", dtype=torch.float16),
        )

        # ---------------- Conv5 + ReLU + MaxPool ---------------- #
        conv5, (H5, W5) = self._get_conv(
            N, 384, 256, H4, W4, 3, 1, 1, "float16"
        )
        x = conv5(
            x,
            self.conv5_w.to(device="cuda", dtype=torch.float16),
            self.conv5_b.to(device="cuda", dtype=torch.float16),
        )

        pool3, (H5p, W5p) = self._get_pool(N, 256, H5, W5, 3, 2, "float16")
        x = pool3(x)  # shape (N,256,6,6)

        # ---------------- Flatten -------------------------------- #
        flatten, flat_dim = self._get_flat(N, 256, H5p, W5p, "float16")
        x = flatten(x)  # (N, flat_dim = 9216)

        # ---------------- FC1 + ReLU ----------------------------- #
        fc1_kernel = self._get_linear(N, 256 * 6 * 6, 4096, "float16")
        x = fc1_kernel(
            x,
            self.fc1_w.to(device="cuda", dtype=torch.float16),
            self.fc1_b.to(device="cuda", dtype=torch.float16),
        )

        # ---------------- FC2 + ReLU ----------------------------- #
        fc2_kernel = self._get_linear(N, 4096, 4096, "float16")
        x = fc2_kernel(
            x,
            self.fc2_w.to(device="cuda", dtype=torch.float16),
            self.fc2_b.to(device="cuda", dtype=torch.float16),
        )

        # ---------------- FC3 (no activation) -------------------- #
        # Re-use linear-relu kernel but supply zero-ReLU by post-clamp
        fc3_total = N * self.fc3_b.shape[0]
        grid = _ceildiv(fc3_total, 256)

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def fc3(
            X: T.Tensor((N, 4096), "float16"),
            W: T.Tensor((self.fc3_b.shape[0], 4096), "float16"),
            B: T.Tensor((self.fc3_b.shape[0],), "float16"),
            Y: T.Tensor((N, self.fc3_b.shape[0]), "float16"),
        ):
            with T.Kernel(grid, threads=256) as bx:
                tx = T.get_thread_binding(0)
                idx = bx * 256 + tx
                if idx < fc3_total:
                    o = idx % self.fc3_b.shape[0]
                    n = idx // self.fc3_b.shape[0]

                    acc = T.Cast("float32", B[o])
                    for i in T.serial(4096):
                        acc += (
                            T.Cast("float32", X[n, i])
                            * T.Cast("float32", W[o, i])
                        )
                    Y[n, o] = T.Cast("float16", acc)

        x = fc3(
            x,
            self.fc3_w.to(device="cuda", dtype=torch.float16),
            self.fc3_b.to(device="cuda", dtype=torch.float16),
        )

        return x.to(orig_dtype)