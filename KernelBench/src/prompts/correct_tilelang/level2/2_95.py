"""
Problem Name: 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.143 runtime_stats={'mean': 0.143, 'std': 0.0115, 'min': 0.137, 'max': 0.245, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0851, 'std': 0.0223, 'min': 0.0725, 'max': 0.283, 'num_trials': 100}, 'speedup_ratio': 0.595}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """
    Linear + add_value + Swish → Tanh → GELU → HardTanh
    All fused in a single TileLang kernel.
    """

    # --------------------------- init --------------------------- #
    def __init__(self, in_features: int, out_features: int, add_value_shape):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # ----- Linear parameters (identical to nn.Linear) -----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # ----- extra add_value -----
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

        # ----- kernel cache -----
        self._kernel_cache = {}

    # --------------------- kernel factory ---------------------- #
    @staticmethod
    def _build_kernel(
        M: int,
        K: int,
        N: int,
        block_M: int = 128,
        block_N: int = 128,
        block_K: int = 32,
        num_stages: int = 2,
        threads: int = 128,
        in_dtype: str = "float16",
        accum_dtype: str = "float32",
    ):
        """
        Out = HardTanh( GELU( tanh( Swish( (X @ W.T) + Bias ) ) ) )
        Bias already includes Linear.bias + add_value when passed in.
        """

        # GELU constants (tanh approximation)
        c0 = 0.044715
        c1 = 0.7978845608028654  # sqrt(2/pi)

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def fused(
            X: T.Tensor((M, K), in_dtype),       # (B,  K)
            W: T.Tensor((N, K), in_dtype),       # (N,  K)
            Bias: T.Tensor((N,), in_dtype),      # (N,)
            Out: T.Tensor((M, N), in_dtype),     # created by TileLang
        ):
            with T.Kernel(
                T.ceildiv(N, block_N),           # grid.x
                T.ceildiv(M, block_M),           # grid.y
                threads=threads,                 # 128 thr / blk
            ) as (bx, by):
                # Shared tiles & accumulators
                A_s = T.alloc_shared((block_M, block_K), in_dtype)
                B_s = T.alloc_shared((block_N, block_K), in_dtype)
                Bias_s = T.alloc_shared((block_N,), in_dtype)
                C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

                # Pre-load bias slice
                T.copy(Bias[bx * block_N], Bias_s)

                # Clear accumulators
                T.clear(C_loc)

                k_tiles = T.ceildiv(K, block_K)
                for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                    T.copy(X[by * block_M, ko * block_K], A_s)
                    T.copy(W[bx * block_N, ko * block_K], B_s)
                    T.gemm(A_s, B_s, C_loc, transpose_B=True)

                # Elementwise post-ops
                for i, j in T.Parallel(block_M, block_N):
                    gi = by * block_M + i
                    gj = bx * block_N + j
                    if (gi < M) and (gj < N):
                        v = C_loc[i, j] + T.Cast(accum_dtype, Bias_s[j])

                        # Swish
                        v = v * T.sigmoid(v)

                        # tanh
                        v = T.tanh(v)

                        # GELU (tanh approximation)
                        v_cub = v * v * v
                        inner = v + T.Cast(accum_dtype, c0) * v_cub
                        tanh_in = T.tanh(T.Cast(accum_dtype, c1) * inner)
                        v = T.Cast(accum_dtype, 0.5) * v * (
                            T.Cast(accum_dtype, 1.0) + tanh_in
                        )

                        # HardTanh clamp to [-1, 1]
                        v = T.min(
                            T.max(v, T.Cast(accum_dtype, -1.0)),
                            T.Cast(accum_dtype, 1.0),
                        )

                        Out[gi, gj] = T.Cast(in_dtype, v)

        return fused

    # -------------------- kernel retrieval -------------------- #
    def _get_kernel(self, batch_size: int):
        key = batch_size
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self._build_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
            )
        return self._kernel_cache[key]

    # --------------------------- forward --------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # Move / cast tensors
        x_f16 = x.to(device=device, dtype=torch.float16, non_blocking=True)
        w_f16 = self.weight.to(device=device, dtype=torch.float16)
        bias_f16 = self.bias.to(device=device, dtype=torch.float16)
        addv_f16 = self.add_value.to(device=device, dtype=torch.float16)

        fused_bias = bias_f16 + addv_f16

        # Retrieve / compile kernel
        kernel = self._get_kernel(x_f16.shape[0])

        # Execute kernel
        y_f16 = kernel(x_f16, w_f16, fused_bias)

        return y_f16.to(orig_dtype)