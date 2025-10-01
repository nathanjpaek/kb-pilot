"""
Problem Name: 44_MiniGPTBlock
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=10.2 runtime_stats={'mean': 10.2, 'std': 0.0182, 'min': 10.2, 'max': 10.4, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 6.78, 'std': 0.016, 'min': 6.75, 'max': 6.81, 'num_trials': 100}, 'speedup_ratio': 0.665}}
"""

import math
from typing import Dict, Tuple

import torch
import tilelang
import tilelang.language as T


# ------------------------------------------------------------
# 1.  TileLang GELU kernel factory (tanh formulation)
# ------------------------------------------------------------
def _build_gelu_kernel(numel: int,
                       block_size: int = 256,
                       in_dtype: str = "float16",
                       accum_dtype: str = "float"):
    C0 = 0.5
    C1 = 0.7978845608028654  # sqrt(2 / pi)
    C2 = 0.044715

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(X: T.Tensor((numel,), in_dtype),
               Y: T.Tensor((numel,), in_dtype)):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            for tx in T.Parallel(block_size):
                idx = bx * block_size + tx
                if idx < numel:
                    x_f32 = T.Cast(accum_dtype, X[idx])
                    x_c3  = x_f32 * x_f32 * x_f32
                    tanh_arg = C1 * (x_f32 + C2 * x_c3)
                    y_f32 = C0 * x_f32 * (1.0 + T.tanh(tanh_arg))
                    Y[idx] = T.Cast(in_dtype, y_f32)

    return kernel


# ------------------------------------------------------------
# 2.  TileLang Linear GEMM kernel factory
# ------------------------------------------------------------
def _build_linear_kernel(M: int, K: int, N: int,
                         block_M: int = 64,
                         block_N: int = 128,
                         block_K: int = 32,
                         num_stages: int = 2,
                         threads: int = 128,
                         in_dtype: str = "float16",
                         accum_dtype: str = "float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(X: T.Tensor((M, K), in_dtype),
               W: T.Tensor((N, K), in_dtype),
               B: T.Tensor((N,),   in_dtype),
               Y: T.Tensor((M, N), in_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_s = T.alloc_shared((block_M, block_K), in_dtype)
            W_s = T.alloc_shared((block_N, block_K), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            B_s = T.alloc_shared((block_N,), in_dtype)

            T.copy(B[bx * block_N:(bx + 1) * block_N], B_s)
            T.clear(C_loc)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(X[by * block_M:(by + 1) * block_M,
                         ko * block_K:(ko + 1) * block_K], X_s)
                T.copy(W[bx * block_N:(bx + 1) * block_N,
                         ko * block_K:(ko + 1) * block_K], W_s)

                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                global_m = by * block_M + i
                global_n = bx * block_N + j
                if (global_m < M) and (global_n < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, B_s[j])
                    Y[global_m, global_n] = T.Cast(in_dtype, val)

    return kernel


# ------------------------------------------------------------
# 3.  Lightweight GELU Module (TileLang powered)
# ------------------------------------------------------------
class _GELU_TL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._kern_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    def _get_kernel(self, numel: int, dtype: torch.dtype):
        key = (numel, dtype)
        if key not in self._kern_cache:
            tl_dtype = "float16" if dtype == torch.float16 else "float"
            self._kern_cache[key] = _build_gelu_kernel(numel, in_dtype=tl_dtype)
        return self._kern_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        numel = x_fp16.numel()
        y = self._get_kernel(numel, x_fp16.dtype)(x_fp16.view(-1))
        return y.view_as(x_fp16).to(orig_dtype)


# ------------------------------------------------------------
# 4.  TileLang Linear layer
# ------------------------------------------------------------
class _Linear_TL(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features

        # Parameter initialisation identical to nn.Linear
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            bound = 1 / math.sqrt(in_features)
            self.bias = torch.nn.Parameter(torch.empty(out_features))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self._kern_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # fetch / compile kernel for current batch size
    def _get_kernel(self, M: int, dtype: torch.dtype):
        key = (M, dtype)
        if key not in self._kern_cache:
            tl_dtype = "float16" if dtype == torch.float16 else "float"
            self._kern_cache[key] = _build_linear_kernel(
                M, self.in_features, self.out_features, in_dtype=tl_dtype
            )
        return self._kern_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        M = x_flat.shape[0]

        x_fp16 = x_flat.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = (
            self.bias.to(device="cuda", dtype=torch.float16).contiguous()
            if self.bias is not None
            else torch.zeros(self.out_features, device="cuda", dtype=torch.float16)
        )

        y = self._get_kernel(M, x_fp16.dtype)(x_fp16, w_fp16, b_fp16)
        return y.reshape(*orig_shape, self.out_features)


# ------------------------------------------------------------
# 5.  Simple LayerNorm (no torch.nn.compute)
# ------------------------------------------------------------
class _LayerNormSimple(torch.nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return norm * self.weight + self.bias


# ------------------------------------------------------------
# 6.  Causal Self-Attention (uses TileLang Linear)
# ------------------------------------------------------------
class _CausalSelfAttention(torch.nn.Module):
    def __init__(self, n_embd: int, n_head: int,
                 max_seqlen: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn = _Linear_TL(n_embd, 3 * n_embd)
        self.c_proj = _Linear_TL(n_embd, n_embd)

        # causal mask as buffer
        mask = torch.tril(torch.ones(max_seqlen, max_seqlen))
        self.register_buffer("bias", mask.view(1, 1, max_seqlen, max_seqlen))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)            # (B,T,3C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,nh,T,hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v                       # (B,nh,T,hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# ------------------------------------------------------------
# 7.  Complete Block
# ------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    Optimised Transformer block:
        x → LN → Attn → +x → LN → MLP(GELU) → +x
    The four Linear layers and all GELU activations are executed by TileLang.
    """

    def __init__(self,
                 n_embd: int,
                 n_head: int,
                 attn_pdrop: float,
                 resid_pdrop: float,
                 max_seqlen: int):
        super().__init__()
        self.ln_1 = _LayerNormSimple(n_embd)
        self.attn = _CausalSelfAttention(n_embd, n_head, max_seqlen)
        self.ln_2 = _LayerNormSimple(n_embd)

        # MLP
        self.c_fc   = _Linear_TL(n_embd, 4 * n_embd)
        self.act    = _GELU_TL()
        self.c_proj = _Linear_TL(4 * n_embd, n_embd)

    # ------------------------------------------------------------------
    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ensure CUDA-FP16 for custom kernels; keep a copy of original dtype
        orig_dtype = x.dtype
        x = x.to(device="cuda", dtype=torch.float16)

        x = x + self.attn(self.ln_1(x))
        x = x + self._mlp_forward(self.ln_2(x))

        return x.to(orig_dtype)