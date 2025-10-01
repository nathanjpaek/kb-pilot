"""
Problem Name: 40_LayerNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.671 runtime_stats={'mean': 0.671, 'std': 0.0154, 'min': 0.662, 'max': 0.819, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 8.03, 'std': 0.0177, 'min': 8.02, 'max': 8.2, 'num_trials': 100}, 'speedup_ratio': 12.0}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_layernorm_kernel(N: int, C: int, H: int, W: int,
                           block_size: int = 256, dtype: str = "float16"):
    """
    Returns a TileLang kernel performing LayerNorm over (C, H, W) for each
    batch item.
    """
    numel = N * C * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:      T.Tensor((N, C, H, W), dtype),
        mean:   T.Tensor((N,),          dtype),
        invstd: T.Tensor((N,),          dtype),
        weight: T.Tensor((C, H, W),     dtype),
        bias:   T.Tensor((C, H, W),     dtype),
        Y:      T.Tensor((N, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx

            if idx < numel:
                w_idx: T.int32 = idx % W
                idx   //= W
                h_idx: T.int32 = idx % H
                idx   //= H
                c_idx: T.int32 = idx % C
                n_idx: T.int32 = idx // C

                x_val = X[n_idx, c_idx, h_idx, w_idx]
                y_val = (x_val - mean[n_idx]) * invstd[n_idx] \
                        * weight[c_idx, h_idx, w_idx] + bias[c_idx, h_idx, w_idx]
                Y[n_idx, c_idx, h_idx, w_idx] = y_val

    return kernel


class ModelNew(nn.Module):
    """
    TileLang implementation of Layer Normalization over the last three
    dimensions (C, H, W) of the input.
    """

    def __init__(self, normalized_shape: tuple, eps: float = 1e-5):
        super().__init__()
        assert len(normalized_shape) == 3, \
            "This LayerNorm implementation expects a 3-D normalized_shape."

        self.normalized_shape = normalized_shape
        C, H, W = normalized_shape

        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(C, H, W))
        self.bias   = nn.Parameter(torch.zeros(C, H, W))
        self.eps = eps

        # Cache compiled kernels keyed by (N, C, H, W, dtype)
        self._cache = {}

    def _get_kernel(self, N: int, dtype: str):
        key = (N, *self.normalized_shape, dtype)
        if key not in self._cache:
            C, H, W = self.normalized_shape
            self._cache[key] = build_layernorm_kernel(N, C, H, W, dtype=dtype)
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to CUDA and fp16 for the custom kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N = x_fp16.shape[0]
        C, H, W = self.normalized_shape
        assert x_fp16.shape[1:] == (C, H, W), "Input shape mismatch."

        # Compute per-sample statistics in fp32 for numerical stability
        x_f32 = x_fp16.to(torch.float32)
        mean_f32 = x_f32.view(N, -1).mean(dim=1)
        var_f32  = x_f32.view(N, -1).var(dim=1, unbiased=False)
        invstd_f32 = torch.rsqrt(var_f32 + self.eps)

        # Cast everything needed by the kernel to fp16 on GPU
        mean_fp16   = mean_f32.to(device="cuda", dtype=torch.float16)
        invstd_fp16 = invstd_f32.to(device="cuda", dtype=torch.float16)
        weight_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        bias_fp16   = self.bias.to(device="cuda", dtype=torch.float16)

        # Retrieve / compile kernel and execute
        kernel = self._get_kernel(N, "float16")
        y_fp16 = kernel(x_fp16, mean_fp16, invstd_fp16, weight_fp16, bias_fp16)

        # Return result in original dtype
        return y_fp16.to(x.dtype)