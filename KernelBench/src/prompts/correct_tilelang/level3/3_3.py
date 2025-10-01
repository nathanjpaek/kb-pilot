"""
Problem Name: 3_DeepNarrowMLP
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.467 runtime_stats={'mean': 0.467, 'std': 0.0136, 'min': 0.442, 'max': 0.498, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.469, 'std': 0.0395, 'min': 0.44, 'max': 0.838, 'num_trials': 100}, 'speedup_ratio': 1.0}}
"""

import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------- #
#                        Kernel factory helpers                           #
# ----------------------------------------------------------------------- #
def _build_linear_relu_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """Y = ReLU(X @ Wᵀ + B)"""
    threads = block_M

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),  # auto-allocated
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as blk:
            tx = T.get_thread_binding(0)
            row = blk * block_M + tx

            if row < M:
                acc = T.alloc_local((N,), accum_dtype)
                # zero initialise
                for j in range(N):
                    acc[j] = T.Cast(accum_dtype, 0)

                # GEMM row
                for k in range(K):
                    x_val = T.Cast(accum_dtype, X[row, k])
                    for j in range(N):
                        acc[j] += x_val * T.Cast(accum_dtype, W[j, k])

                # bias + ReLU + store
                zero = T.Cast(accum_dtype, 0)
                for j in range(N):
                    v = acc[j] + T.Cast(accum_dtype, B[j])
                    v = T.max(v, zero)
                    Y[row, j] = T.Cast(dtype, v)

    return kernel


def _build_linear_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """Y = X @ Wᵀ + B   (no activation)"""
    threads = block_M

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),  # auto-allocated
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as blk:
            tx = T.get_thread_binding(0)
            row = blk * block_M + tx

            if row < M:
                acc = T.alloc_local((N,), accum_dtype)
                for j in range(N):
                    acc[j] = T.Cast(accum_dtype, 0)

                for k in range(K):
                    x_val = T.Cast(accum_dtype, X[row, k])
                    for j in range(N):
                        acc[j] += x_val * T.Cast(accum_dtype, W[j, k])

                for j in range(N):
                    v = acc[j] + T.Cast(accum_dtype, B[j])
                    Y[row, j] = T.Cast(dtype, v)

    return kernel


# ----------------------------------------------------------------------- #
#                              PyTorch wrapper                            #
# ----------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Feed-forward network with arbitrary hidden layers implemented via TileLang.
    Architecture:  (Linear → ReLU)*  ... → Linear
    """

    def __init__(self, input_size: int, hidden_layer_sizes: List[int], output_size: int):
        super().__init__()
        self.layer_sizes: List[Tuple[int, int]] = []

        # ----- Parameter creation -----
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        prev = int(input_size)
        for h in hidden_layer_sizes:
            h = int(h)
            self.layer_sizes.append((prev, h))
            w = nn.Parameter(torch.empty(h, prev))
            b = nn.Parameter(torch.empty(h))
            self._init_linear_params(w, b, prev)
            self.weights.append(w)
            self.biases.append(b)
            prev = h

        # final linear layer
        self.layer_sizes.append((prev, int(output_size)))
        w_out = nn.Parameter(torch.empty(output_size, prev))
        b_out = nn.Parameter(torch.empty(output_size))
        self._init_linear_params(w_out, b_out, prev)
        self.weights.append(w_out)
        self.biases.append(b_out)

        # kernel caches : list aligned with layers
        self._kernel_relu_cache: List[Dict[Tuple[int, str], callable]] = [
            {} for _ in range(len(hidden_layer_sizes))
        ]
        self._kernel_linear_cache: Dict[Tuple[int, str], callable] = {}

    # ---------------- utility ----------------
    @staticmethod
    def _init_linear_params(w: nn.Parameter, b: nn.Parameter, in_features: int):
        with torch.no_grad():
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(b, -bound, bound)

    # ---------------- kernel getters ----------------
    def _get_relu_kernel(self, idx: int, batch: int, dtype: str):
        cache = self._kernel_relu_cache[idx]
        key = (batch, dtype)
        if key not in cache:
            K, N = self.layer_sizes[idx]
            cache[key] = _build_linear_relu_kernel(
                batch, K, N, dtype=dtype
            )
        return cache[key]

    def _get_final_kernel(self, batch: int, dtype: str):
        key = (batch, dtype)
        if key not in self._kernel_linear_cache:
            K, N = self.layer_sizes[-1]
            self._kernel_linear_cache[key] = _build_linear_kernel(
                batch, K, N, dtype=dtype
            )
        return self._kernel_linear_cache[key]

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = torch.device("cuda")

        x_f16 = x.to(device=device, dtype=torch.float16).contiguous()
        batch = x_f16.shape[0]

        # iterate over hidden layers with ReLU
        for idx in range(len(self.layer_sizes) - 1):
            w = self.weights[idx].to(device=device, dtype=torch.float16).contiguous()
            b = self.biases[idx].to(device=device, dtype=torch.float16).contiguous()
            k = self._get_relu_kernel(idx, batch, "float16")
            x_f16 = k(x_f16, w, b)

        # final linear (no activation)
        w_final = self.weights[-1].to(device=device, dtype=torch.float16).contiguous()
        b_final = self.biases[-1].to(device=device, dtype=torch.float16).contiguous()
        k_final = self._get_final_kernel(batch, "float16")
        out_f16 = k_final(x_f16, w_final, b_final)

        return out_f16.to(orig_dtype)