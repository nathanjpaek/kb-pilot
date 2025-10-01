"""
Problem Name: 22_Tanh
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0485 runtime_stats={'mean': 0.0485, 'std': 0.00495, 'min': 0.042, 'max': 0.0628, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.023, 'std': 0.0033, 'min': 0.0207, 'max': 0.0431, 'num_trials': 100}, 'speedup_ratio': 0.474}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_tanh_kernel(N, block_size: int = 256, dtype: str = "float16"):
    """
    Factory that returns a TileLang kernel which applies element-wise tanh
    to a flat tensor of length N.
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def tanh_kernel(
        X: T.Tensor((N,), dtype),
        Y: T.Tensor((N,), dtype),
    ):
        # 1-D grid; each block has `block_size` threads
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)  # threadIdx.x
            idx: T.int32 = bx * block_size + tx
            if idx < N:
                Y[idx] = T.tanh(X[idx])

    return tanh_kernel


class ModelNew(nn.Module):
    """
    Optimized model using TileLang to perform element-wise Tanh.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Cache compiled kernels keyed by (numel, dtype)
        self._kernel_cache = {}

    def _get_kernel(self, numel: int, dtype: torch.dtype):
        key = (numel, dtype)
        if key not in self._kernel_cache:
            tl_dtype = "float16" if dtype == torch.float16 else "float"
            self._kernel_cache[key] = build_tanh_kernel(numel, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor using the compiled
        TileLang kernel.
        """
        orig_shape = x.shape
        # TileLang works best with fp16 on CUDA
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        flat_x = x_fp16.view(-1)

        kernel = self._get_kernel(flat_x.numel(), flat_x.dtype)
        flat_out = kernel(flat_x)
        out = flat_out.view(orig_shape).to(x.dtype)  # cast back to original dtype
        return out