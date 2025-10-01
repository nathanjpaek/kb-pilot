"""
Problem Name: 24_Conv3d_Min_Softmax
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.475 runtime_stats={'mean': 0.475, 'std': 0.0077, 'min': 0.469, 'max': 0.545, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.525, 'std': 0.00857, 'min': 0.519, 'max': 0.597, 'num_trials': 100}, 'speedup_ratio': 1.11}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factories
# --------------------------------------------------------------------------- #
def _build_min_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    red_dim: int,                 # 2, 3 or 4   (D,H,W)
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Build a specialised 5-D → 4-D min-reduction kernel removing `red_dim`.
    red_dim is the dimension index in the original (N,C,D,H,W) layout.
    """
    red_dim = int(red_dim)
    assert red_dim in (2, 3, 4)

    if red_dim == 2:                                                # reduce D
        OUT1, OUT2 = H, W
        tot_out    = N * C * H * W

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            X: T.Tensor((N, C, D, H, W), dtype),
            Y: T.Tensor((N, C, OUT1, OUT2), dtype),
        ):
            big = T.Cast(accum_dtype, 1.0e9)
            with T.Kernel(T.ceildiv(tot_out, block_size), threads=block_size) as bx:
                tx  = T.get_thread_binding(0)
                idx = bx * block_size + tx
                if idx < tot_out:
                    w  = idx % W
                    tmp = idx // W
                    h  = tmp % H
                    tmp = tmp // H
                    c  = tmp % C
                    n  = tmp // C

                    m = T.alloc_local((1,), accum_dtype)
                    m[0] = big
                    for d in range(D):
                        val = X[n, c, d, h, w].astype(accum_dtype)
                        m[0] = T.min(m[0], val)
                    Y[n, c, h, w] = T.Cast(dtype, m[0])

        return kernel

    if red_dim == 3:                                                # reduce H
        OUT1, OUT2 = D, W
        tot_out    = N * C * D * W

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            X: T.Tensor((N, C, D, H, W), dtype),
            Y: T.Tensor((N, C, OUT1, OUT2), dtype),
        ):
            big = T.Cast(accum_dtype, 1.0e9)
            with T.Kernel(T.ceildiv(tot_out, block_size), threads=block_size) as bx:
                tx  = T.get_thread_binding(0)
                idx = bx * block_size + tx
                if idx < tot_out:
                    w  = idx % W
                    tmp = idx // W
                    d  = tmp % D
                    tmp = tmp // D
                    c  = tmp % C
                    n  = tmp // C

                    m = T.alloc_local((1,), accum_dtype)
                    m[0] = big
                    for h in range(H):
                        val = X[n, c, d, h, w].astype(accum_dtype)
                        m[0] = T.min(m[0], val)
                    Y[n, c, d, w] = T.Cast(dtype, m[0])

        return kernel

    # red_dim == 4   (reduce W)
    OUT1, OUT2 = D, H
    tot_out    = N * C * D * H

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, C, OUT1, OUT2), dtype),
    ):
        big = T.Cast(accum_dtype, 1.0e9)
        with T.Kernel(T.ceildiv(tot_out, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < tot_out:
                h  = idx % H
                tmp = idx // H
                d  = tmp % D
                tmp = tmp // D
                c  = tmp % C
                n  = tmp // C

                m = T.alloc_local((1,), accum_dtype)
                m[0] = big
                for w in range(W):
                    val = X[n, c, d, h, w].astype(accum_dtype)
                    m[0] = T.min(m[0], val)
                Y[n, c, d, h] = T.Cast(dtype, m[0])

    return kernel


def _build_softmax_kernel(
    N: int,
    C: int,
    S1: int,
    S2: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Softmax over channel dimension for a (N,C,S1,S2) tensor.
    """
    groups   = N * S1 * S2                                  # one softmax per group
    tot_elem = groups                                       # one thread per group

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, S1, S2), dtype),
        Y: T.Tensor((N, C, S1, S2), dtype),
    ):
        one  = T.Cast(accum_dtype, 1.0)
        with T.Kernel(T.ceildiv(tot_elem, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            gid = bx * block_size + tx
            if gid < groups:
                s2 = gid % S2
                tmp = gid // S2
                s1 = tmp % S1
                n  = tmp // S1

                # 1) find max for numerical stability
                maxv = T.alloc_local((1,), accum_dtype)
                maxv[0] = -T.Cast(accum_dtype, 1.0e9)
                for c in range(C):
                    v = X[n, c, s1, s2].astype(accum_dtype)
                    maxv[0] = T.max(maxv[0], v)

                # 2) compute sum of exp
                sumv = T.alloc_local((1,), accum_dtype)
                sumv[0] = 0.0
                for c in range(C):
                    v = X[n, c, s1, s2].astype(accum_dtype)
                    e = T.exp(v - maxv[0])
                    sumv[0] += e

                # 3) write normalised values
                denom = one / sumv[0]
                for c in range(C):
                    v = X[n, c, s1, s2].astype(accum_dtype)
                    e = T.exp(v - maxv[0]) * denom
                    Y[n, c, s1, s2] = T.Cast(dtype, e)

    return kernel


# --------------------------------------------------------------------------- #
# Optimised PyTorch Module
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d → min-reduction (along given dim) → channel-softmax, both on TileLang.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.red_dim = int(dim)              # 2/3/4 supported

        # kernel cache  :  {(shape,dtype,tag) : kernel}
        self._kern_cache = {}

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _key(self, *vals):
        return tuple(vals)

    def _get_min_kernel(self, shape, dtype):
        N, C, D, H, W = shape
        key = self._key("min", *shape, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_min_kernel(
                N, C, D, H, W, self.red_dim, dtype=dtype
            )
        return self._kern_cache[key]

    def _get_softmax_kernel(self, out_shape, dtype):
        N, C, S1, S2 = out_shape
        key = self._key("softmax", *out_shape, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_softmax_kernel(
                N, C, S1, S2, dtype=dtype
            )
        return self._kern_cache[key]

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # 1. Conv3d           (cuDNN, float32 for accuracy)
        x = self.conv(x)

        # 2. to CUDA / fp16
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, D, H, W = x_fp16.shape

        # 3. min-reduction kernel
        min_kernel = self._get_min_kernel((N, C, D, H, W), "float16")
        if self.red_dim == 2:
            y_min = min_kernel(x_fp16)                   # (N,C,H,W)
            S1, S2 = H, W
        elif self.red_dim == 3:
            y_min = min_kernel(x_fp16)                   # (N,C,D,W)
            S1, S2 = D, W
        else:  # self.red_dim == 4
            y_min = min_kernel(x_fp16)                   # (N,C,D,H)
            S1, S2 = D, H

        # 4. soft-max kernel
        soft_kernel = self._get_softmax_kernel((N, C, S1, S2), "float16")
        y_soft = soft_kernel(y_min)

        return y_soft.to(orig_dtype)