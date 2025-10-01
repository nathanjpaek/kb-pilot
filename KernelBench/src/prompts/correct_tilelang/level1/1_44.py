"""
Problem Name: 44_Average_Pooling_1D
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0361 runtime_stats={'mean': 0.0361, 'std': 0.00501, 'min': 0.0309, 'max': 0.0569, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0238, 'std': 0.00369, 'min': 0.0209, 'max': 0.0418, 'num_trials': 100}, 'speedup_ratio': 0.659}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _avg_pool1d_kernel(
    N: int,
    C: int,
    L_in: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    K = kernel_size
    S = stride
    P = padding
    L_out = (L_in + 2 * P - K) // S + 1
    O = N * C * L_out
    block_elems = 256
    thread_num = block_elems
    kernel_area_val = K

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def avg_pool1d(
        A: T.Tensor((N, C, L_in), dtype),
        Out: T.Tensor((N, C, L_out), dtype),
    ):
        with T.Kernel(T.ceildiv(O, block_elems), threads=thread_num) as bx:
            tx = T.get_thread_binding(0)
            gid = bx * block_elems + tx
            if gid < O:
                l_out = gid % L_out
                tmp = gid // L_out
                c = tmp % C
                n = tmp // C

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                l_in_start = l_out * S - P
                for kx in T.serial(K):
                    lin = l_in_start + kx
                    in_bound = (lin >= 0) and (lin < L_in)
                    acc[0] += T.if_then_else(
                        in_bound,
                        T.Cast(accum_dtype, A[n, c, lin]),
                        T.Cast(accum_dtype, 0),
                    )

                acc[0] = acc[0] / T.Cast(accum_dtype, kernel_area_val)
                Out[n, c, l_out] = T.Cast(dtype, acc[0])

    return avg_pool1d


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)
        self._cached_kernels = {}

    def _get_kernel(self, N: int, C: int, L_in: int, dtype: str):
        key = (N, C, L_in, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = _avg_pool1d_kernel(
                N,
                C,
                L_in,
                self.kernel_size,
                self.stride,
                self.padding,
                dtype=dtype,
            )
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, L_in = x_fp16.shape
        kernel = self._get_kernel(N, C, L_in, "float16")
        y_fp16 = kernel(x_fp16)
        return y_fp16.to(orig_dtype)