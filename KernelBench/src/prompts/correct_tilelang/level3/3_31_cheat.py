"""
Problem Name: 31_VisionAttention
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=24.5 runtime_stats={'mean': 24.5, 'std': 0.15, 'min': 24.3, 'max': 25.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 9.74, 'std': 0.112, 'min': 9.68, 'max': 10.4, 'num_trials': 100}, 'speedup_ratio': 0.398}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_linear_kernel(M, in_features, out_features, block_M=64, block_N=64, block_K=32,
                        dtype="float16", accum_dtype="float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_kernel(
        A: T.Tensor((M, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        Bias: T.Tensor((out_features,), dtype),
        Out: T.Tensor((M, out_features), dtype),
    ):
        grid_x = T.ceildiv(out_features, block_N)
        grid_y = T.ceildiv(M, block_M)
        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            bias_shared = T.alloc_shared((block_N,), dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(W[bx * block_N, ko * block_K], W_shared)
                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            T.copy(Bias[bx * block_N], bias_shared)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += bias_shared[j]

            T.copy(C_local, Out[by * block_M, bx * block_N])

    return linear_kernel


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps

        # Projection parameters
        self.W_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_k = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_o = nn.Parameter(torch.empty(embed_dim, embed_dim))

        self.b_q = nn.Parameter(torch.zeros(embed_dim))
        self.b_k = nn.Parameter(torch.zeros(embed_dim))
        self.b_v = nn.Parameter(torch.zeros(embed_dim))
        self.b_o = nn.Parameter(torch.zeros(embed_dim))

        # LayerNorm parameters
        self.ln_weight = nn.Parameter(torch.ones(embed_dim))
        self.ln_bias = nn.Parameter(torch.zeros(embed_dim))

        # Initialize projection weights with Xavier uniform to mimic nn.MultiheadAttention
        for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(w)
        # Biases already zero-initialized

        # Kernel cache
        self._linear_kernels = {}

    def _get_linear_kernel(self, M):
        key = M
        if key not in self._linear_kernels:
            kernel = build_linear_kernel(M, self.embed_dim, self.embed_dim)
            self._linear_kernels[key] = kernel
        return self._linear_kernels[key]

    def _linear(self, X, W, b):
        M = X.shape[0]
        kernel = self._get_linear_kernel(M)
        return kernel(X, W, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = "cuda"
        dtype = torch.float16

        x = x.to(device=device, dtype=dtype)
        seq_len = H * W
        x_seq = x.permute(0, 2, 3, 1).reshape(B * seq_len, C)

        # Projections
        Q = self._linear(x_seq, self.W_q.to(device, dtype), self.b_q.to(device, dtype))
        K = self._linear(x_seq, self.W_k.to(device, dtype), self.b_k.to(device, dtype))
        V = self._linear(x_seq, self.W_v.to(device, dtype), self.b_v.to(device, dtype))

        # Reshape to (B, num_heads, seq_len, head_dim)
        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) * scale  # (B, heads, S, S)
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values
        attn_exp = torch.exp(attn_scores)
        attn_probs = attn_exp / attn_exp.sum(dim=-1, keepdim=True)

        attn_output = torch.matmul(attn_probs, V)  # (B, heads, S, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B * seq_len, C)

        # Output projection
        out_proj = self._linear(attn_output, self.W_o.to(device, dtype), self.b_o.to(device, dtype))

        # Residual connection
        out_proj = out_proj + x_seq

        # LayerNorm
        out_proj_fp32 = out_proj.to(torch.float32)
        mean = out_proj_fp32.mean(dim=-1, keepdim=True)
        var = out_proj_fp32.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (out_proj_fp32 - mean) / torch.sqrt(var + self.eps)
        normalized = normalized * self.ln_weight.to(device) + self.ln_bias.to(device)
        normalized = normalized.to(dtype)

        # Reshape back to (B, C, H, W)
        out = normalized.view(B, seq_len, C).permute(0, 2, 1).view(B, C, H, W)
        return out.to(torch.float32)