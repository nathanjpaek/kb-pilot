"""
Problem Name: 34_InstanceNorm
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.646 runtime_stats={'mean': 0.646, 'std': 0.00196, 'min': 0.642, 'max': 0.656, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.308, 'std': 0.0173, 'min': 0.3, 'max': 0.471, 'num_trials': 100}, 'speedup_ratio': 0.477}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_instance_norm_kernel(B, C, H, W, block_size=256, dtype="float16"):
    numel = B * C * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((B, C, H, W), dtype),
        mean: T.Tensor((B, C), dtype),
        invstd: T.Tensor((B, C), dtype),
        Y: T.Tensor((B, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                w = idx % W
                idx = idx // W
                h = idx % H
                idx = idx // H
                c = idx % C
                n = idx // C
                x_val = X[n, c, h, w]
                y_val = (x_val - mean[n, c]) * invstd[n, c]
                Y[n, c, h, w] = y_val

    return main


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self._cache = {}

    def _get_kernel(self, shape, dtype):
        key = (shape, dtype)
        if key not in self._cache:
            B, C, H, W = shape
            self._cache[key] = build_instance_norm_kernel(B, C, H, W, dtype=dtype)
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        B, C, H, W = x_fp16.shape

        x_f32 = x_fp16.to(torch.float32)
        mean_f32 = x_f32.mean(dim=(2, 3))
        var_f32 = x_f32.var(dim=(2, 3), unbiased=False)
        invstd_f32 = torch.rsqrt(var_f32 + self.eps)

        mean_fp16 = mean_f32.to(device="cuda", dtype=torch.float16)
        invstd_fp16 = invstd_f32.to(device="cuda", dtype=torch.float16)

        kernel = self._get_kernel((B, C, H, W), "float16")
        y_fp16 = kernel(x_fp16, mean_fp16, invstd_fp16)
        return y_fp16.to(x.dtype)