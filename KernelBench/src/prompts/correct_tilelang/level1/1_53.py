"""
Problem Name: 53_Min_reduction_over_a_dimension
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0347 runtime_stats={'mean': 0.0347, 'std': 0.00468, 'min': 0.0295, 'max': 0.0509, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0353, 'std': 0.0127, 'min': 0.0326, 'max': 0.16, 'num_trials': 100}, 'speedup_ratio': 1.02}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_min_kernel(shape, dim, block_size: int = 256, dtype: str = "float16"):
    """
    Build a specialised min-reduction kernel for a fixed tensor `shape`
    (assumed to be 3-D) and reduction axis `dim`.
    """
    B, D1, D2 = shape
    dim = dim if dim >= 0 else dim + 3  # normalise

    if dim == 0:
        OUT0, OUT1 = D1, D2

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            X: T.Tensor((B, D1, D2), dtype),
            Y: T.Tensor((OUT0, OUT1), dtype),
        ):
            total = OUT0 * OUT1
            with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
                tx = T.get_thread_binding(0)
                idx = bx * block_size + tx
                if idx < total:
                    d1 = idx // D2
                    d2 = idx % D2
                    m = T.alloc_local((1,), dtype)
                    m[0] = X[0, d1, d2]
                    for b in range(1, B):
                        m[0] = T.min(m[0], X[b, d1, d2])
                    Y[d1, d2] = m[0]

        return kernel

    if dim == 1:
        OUT0, OUT1 = B, D2

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            X: T.Tensor((B, D1, D2), dtype),
            Y: T.Tensor((OUT0, OUT1), dtype),
        ):
            total = OUT0 * OUT1
            with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
                tx = T.get_thread_binding(0)
                idx = bx * block_size + tx
                if idx < total:
                    b = idx // D2
                    d2 = idx % D2
                    m = T.alloc_local((1,), dtype)
                    m[0] = X[b, 0, d2]
                    for d1 in range(1, D1):
                        m[0] = T.min(m[0], X[b, d1, d2])
                    Y[b, d2] = m[0]

        return kernel

    # dim == 2
    OUT0, OUT1 = B, D1

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((B, D1, D2), dtype),
        Y: T.Tensor((OUT0, OUT1), dtype),
    ):
        total = OUT0 * OUT1
        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total:
                b = idx // D1
                d1 = idx % D1
                m = T.alloc_local((1,), dtype)
                m[0] = X[b, d1, 0]
                for d2 in range(1, D2):
                    m[0] = T.min(m[0], X[b, d1, d2])
                Y[b, d1] = m[0]

    return kernel


class ModelNew(nn.Module):
    """
    TileLang-accelerated replacement for the original min-reduction model.
    Supports reduction over any axis of a 3-D tensor.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self._kernel_cache = {}  # (shape, dim, dtype) -> compiled kernel

    def _get_kernel(self, shape, dtype: str):
        key = (shape, self.dim, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_min_kernel(shape, self.dim, dtype=dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move to CUDA + fp16 for the kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        kernel = self._get_kernel(tuple(x_fp16.shape), "float16")
        y_fp16 = kernel(x_fp16)
        return y_fp16.to(x.dtype)