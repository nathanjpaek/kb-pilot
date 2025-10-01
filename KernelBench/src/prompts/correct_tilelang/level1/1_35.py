"""
Problem Name: 35_GroupNorm_
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.684 runtime_stats={'mean': 0.684, 'std': 0.0244, 'min': 0.656, 'max': 0.735, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.65, 'std': 0.032, 'min': 0.604, 'max': 0.752, 'num_trials': 100}, 'speedup_ratio': 0.95}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_groupnorm_kernel(N, C, H, W, G, block_size: int = 256, dtype: str = "float16"):
    numel = N * C * H * W
    group_size = C // G

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),
        mean: T.Tensor((N, G), dtype),
        invstd: T.Tensor((N, G), dtype),
        weight: T.Tensor((C,), dtype),
        bias: T.Tensor((C,), dtype),
        Y: T.Tensor((N, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                w = idx % W
                idx //= W
                h = idx % H
                idx //= H
                c = idx % C
                n = idx // C
                g = c // group_size
                x_val = X[n, c, h, w]
                y_val = (x_val - mean[n, g]) * invstd[n, g] * weight[c] + bias[c]
                Y[n, c, h, w] = y_val

    return kernel


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self._cache = {}

    def _kernel(self, N, H, W, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._cache:
            self._cache[key] = build_groupnorm_kernel(
                N, self.num_features, H, W, self.num_groups, dtype=dtype
            )
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, H, W = x_fp16.shape
        assert C == self.num_features

        x_f32 = x_fp16.to(torch.float32)
        G = self.num_groups
        x_group = x_f32.view(N, G, -1)
        mean_f32 = x_group.mean(dim=2)
        var_f32 = x_group.var(dim=2, unbiased=False)
        invstd_f32 = torch.rsqrt(var_f32 + self.eps)

        mean_fp16 = mean_f32.to(device="cuda", dtype=torch.float16)
        invstd_fp16 = invstd_f32.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        y_fp16 = self._kernel(N, H, W, "float16")(
            x_fp16, mean_fp16, invstd_fp16, w_fp16, b_fp16
        )
        return y_fp16.to(x.dtype)