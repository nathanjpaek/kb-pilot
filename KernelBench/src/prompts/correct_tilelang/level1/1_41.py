"""
Problem Name: 41_Max_Pooling_1D
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.061 runtime_stats={'mean': 0.061, 'std': 0.00535, 'min': 0.0578, 'max': 0.0911, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.05, 'std': 0.00868, 'min': 0.0442, 'max': 0.113, 'num_trials': 100}, 'speedup_ratio': 0.82}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _maxpool1d_kernel(
    N: int,
    C: int,
    L_in: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    block_elems: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    K = kernel_size
    S = stride
    P = padding
    D = dilation
    # PyTorch formula for output length
    L_out = (L_in + 2 * P - D * (K - 1) - 1) // S + 1
    TOT = N * C * L_out
    neg_inf = -T.infinity(accum_dtype)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def maxpool1d(
        X:   T.Tensor((N, C, L_in), dtype),
        Out: T.Tensor((N, C, L_out), dtype),
    ):
        with T.Kernel(T.ceildiv(TOT, block_elems), threads=block_elems) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_elems + tx
            if idx < TOT:
                l_out = idx % L_out
                tmp1  = idx // L_out
                c     = tmp1 % C
                n     = tmp1 // C

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = neg_inf

                l_start = l_out * S - P
                for k in T.serial(K):
                    l_in = l_start + k * D
                    in_bound = (l_in >= 0) and (l_in < L_in)
                    val = T.if_then_else(
                        in_bound,
                        T.Cast(accum_dtype, X[n, c, l_in]),
                        neg_inf,
                    )
                    acc[0] = T.max(acc[0], val)

                Out[n, c, l_out] = T.Cast(dtype, acc[0])

    return maxpool1d


class ModelNew(nn.Module):
    """TileLang-optimised MaxPool1d (return_indices=False only)."""

    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
    ):
        super().__init__()
        assert not return_indices, "return_indices=True not supported in optimized path"
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self._cached_kernels = {}

    def _get_kernel(self, N: int, C: int, L_in: int, dtype: str):
        key = (N, C, L_in, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = _maxpool1d_kernel(
                N,
                C,
                L_in,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                dtype=dtype,
            )
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        N, C, L_in = x_fp16.shape
        kernel = self._get_kernel(N, C, L_in, "float16")
        y_fp16 = kernel(x_fp16)
        return y_fp16.to(x.dtype)