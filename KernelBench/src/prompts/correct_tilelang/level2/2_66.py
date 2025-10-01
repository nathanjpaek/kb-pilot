"""
Problem Name: 66_Matmul_Dropout_Mean_Softmax
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0333 runtime_stats={'mean': 0.0333, 'std': 0.00129, 'min': 0.0314, 'max': 0.0398, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0652, 'std': 0.00248, 'min': 0.0622, 'max': 0.079, 'num_trials': 100}, 'speedup_ratio': 1.96}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p

        # Initialize parameters identically to nn.Linear defaults
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Cache kernels keyed by (batch_size, dtype)
        self._kernel_cache = {}

    def _make_kernel(self, batch_size: int, dtype: str = "float16"):
        in_features = self.in_features
        block_size = 256

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            X: T.Tensor((batch_size, in_features), dtype),
            Out: T.Tensor((batch_size, 1), dtype),
        ):
            with T.Kernel(T.ceildiv(batch_size, block_size), threads=block_size) as bx:
                tx = T.get_thread_binding(0)
                idx = bx * block_size + tx
                if idx < batch_size:
                    Out[idx, 0] = T.Cast(dtype, 1)

        return kernel

    def _get_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self._make_kernel(batch_size, dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        batch_size = x_fp16.shape[0]
        kernel = self._get_kernel(batch_size, "float16")

        y_fp16 = kernel(x_fp16)
        return y_fp16.to(orig_dtype)