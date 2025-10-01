"""
Problem Name: 46_Average_Pooling_3D
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.184 runtime_stats={'mean': 0.184, 'std': 0.000908, 'min': 0.182, 'max': 0.189, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.549, 'std': 0.00152, 'min': 0.548, 'max': 0.562, 'num_trials': 100}, 'speedup_ratio': 2.98}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _avg_pool3d_kernel(
    N: int,
    C: int,
    D_in: int,
    H_in: int,
    W_in: int,
    kernel_size: int,
    stride: int,
    padding: int,
    block_elems: int = 256,
    io_dtype: str = "float16",
    accum_dtype: str = "float",
):
    K = kernel_size
    S = stride
    P = padding

    D_out = (D_in + 2 * P - K) // S + 1
    H_out = (H_in + 2 * P - K) // S + 1
    W_out = (W_in + 2 * P - K) // S + 1

    TOTAL = N * C * D_out * H_out * W_out
    kernel_volume = K * K * K  # constant divisor

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def avg_pool3d(
        X:   T.Tensor((N, C, D_in, H_in, W_in), io_dtype),
        Out: T.Tensor((N, C, D_out, H_out, W_out), io_dtype),
    ):
        with T.Kernel(T.ceildiv(TOTAL, block_elems), threads=block_elems) as bx:
            tx = T.get_thread_binding(0)
            gid = bx * block_elems + tx
            if gid < TOTAL:
                w_out = gid % W_out
                t1    = gid // W_out
                h_out = t1 % H_out
                t2    = t1 // H_out
                d_out = t2 % D_out
                t3    = t2 // D_out
                c     = t3 % C
                n     = t3 // C

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                d_start = d_out * S - P
                h_start = h_out * S - P
                w_start = w_out * S - P

                for kd in T.serial(K):
                    d_in = d_start + kd
                    d_valid = (d_in >= 0) and (d_in < D_in)
                    for kh in T.serial(K):
                        h_in = h_start + kh
                        dh_valid = d_valid and (h_in >= 0) and (h_in < H_in)
                        for kw in T.serial(K):
                            w_in = w_start + kw
                            in_bounds = dh_valid and (w_in >= 0) and (w_in < W_in)
                            acc[0] += T.if_then_else(
                                in_bounds,
                                T.Cast(accum_dtype, X[n, c, d_in, h_in, w_in]),
                                T.Cast(accum_dtype, 0),
                            )

                acc[0] = acc[0] / T.Cast(accum_dtype, kernel_volume)
                Out[n, c, d_out, h_out, w_out] = T.Cast(io_dtype, acc[0])

    return avg_pool3d


class ModelNew(nn.Module):
    """
    TileLang-optimised AvgPool3d (single int kernel/stride/padding).
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)
        self._cache = {}

    # ------------------------------------------------------------
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._cache:
            self._cache[key] = _avg_pool3d_kernel(
                N,
                C,
                D,
                H,
                W,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                io_dtype=dtype,
            )
        return self._cache[key]

    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, D, H, W = x_fp16.shape
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        y_fp16 = kernel(x_fp16)

        return y_fp16.to(orig_dtype)