"""
Problem Name: 45_Average_Pooling_2D
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0648 runtime_stats={'mean': 0.0648, 'std': 0.00951, 'min': 0.0589, 'max': 0.149, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.118, 'std': 0.0122, 'min': 0.114, 'max': 0.237, 'num_trials': 100}, 'speedup_ratio': 1.82}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory
# --------------------------------------------------------------------------- #
def _avg_pool2d_kernel(
    N: int,
    C: int,
    H_in: int,
    W_in: int,
    kernel_size: int,
    stride: int,
    padding: int,
    block_elems: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    K = kernel_size
    S = stride
    P = padding

    H_out = (H_in + 2 * P - K) // S + 1
    W_out = (W_in + 2 * P - K) // S + 1
    TOT   = N * C * H_out * W_out
    kernel_area = K * K

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def avg_pool2d(
        X:   T.Tensor((N, C, H_in, W_in), dtype),
        Out: T.Tensor((N, C, H_out, W_out), dtype),
    ):
        with T.Kernel(T.ceildiv(TOT, block_elems), threads=block_elems) as bx:
            tx   = T.get_thread_binding(0)
            gidx = bx * block_elems + tx
            if gidx < TOT:
                w_out = gidx % W_out
                tmp1  = gidx // W_out
                h_out = tmp1 % H_out
                tmp2  = tmp1 // H_out
                c     = tmp2 % C
                n     = tmp2 // C

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                h_start = h_out * S - P
                w_start = w_out * S - P

                for ky in T.serial(K):
                    in_h = h_start + ky
                    for kx in T.serial(K):
                        in_w = w_start + kx
                        in_bound = (
                            (in_h >= 0)
                            and (in_h < H_in)
                            and (in_w >= 0)
                            and (in_w < W_in)
                        )
                        acc[0] += T.if_then_else(
                            in_bound,
                            T.Cast(accum_dtype, X[n, c, in_h, in_w]),
                            T.Cast(accum_dtype, 0),
                        )

                acc[0] = acc[0] / T.Cast(accum_dtype, kernel_area)
                Out[n, c, h_out, w_out] = T.Cast(dtype, acc[0])

    return avg_pool2d


# --------------------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    TileLang-optimised AvgPool2d (count_include_pad=True, ceil_mode=False).
    """

    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride  = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)

        self._kernel_cache = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _avg_pool2d_kernel(
                N,
                C,
                H,
                W,
                self.kernel_size,
                self.stride,
                self.padding,
                dtype=dtype,
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = x_fp16.shape
        kernel = self._get_kernel(N, C, H, W, "float16")

        y_fp16 = kernel(x_fp16)
        return y_fp16.to(orig_dtype)