"""
Problem Name: 50_ReLUSelfAttention
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.7 runtime_stats={'mean': 3.7, 'std': 0.0126, 'min': 3.68, 'max': 3.81, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.79, 'std': 0.0123, 'min': 1.78, 'max': 1.9, 'num_trials': 100}, 'speedup_ratio': 0.484}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :   (B·T,  n_embd)  ×  (3·n_embd, n_embd)^T  + bias #
# --------------------------------------------------------------------------- #
def _build_c_attn_kernel(
    M: int,                      # B * T   (dynamic)
    K: int,                      # n_embd  (static)
    N: int,                      # 3*n_embd
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 256,
    in_dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def c_attn_kernel(
        X: T.Tensor((M, K), in_dtype),        # (B*T ,  n_embd)
        W: T.Tensor((N, K), in_dtype),        # (3*n_embd , n_embd)
        B: T.Tensor((N,), in_dtype),          # bias
        Out: T.Tensor((M, N), in_dtype),      # created by TileLang
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            # ----------------------------------------------------------------
            # Shared / fragment buffers
            X_s = T.alloc_shared((block_M, block_K), in_dtype)
            W_s = T.alloc_shared((block_N, block_K), in_dtype)   # row-major tile
            B_s = T.alloc_shared((block_N,), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Load bias slice once per block
            T.copy(B[bx * block_N : (bx + 1) * block_N], B_s)

            # Clear local accumulator
            T.clear(C_loc)

            # Main reduction loop along K
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(
                    X[by * block_M : (by + 1) * block_M,
                      ko * block_K : (ko + 1) * block_K],
                    X_s,
                )
                T.copy(
                    W[bx * block_N : (bx + 1) * block_N,
                      ko * block_K : (ko + 1) * block_K],
                    W_s,
                )
                # GEMM  (X_s) [M×K]  ×  (W_s)^T  [K×N]  →  C_loc  [M×N]
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            # Epilogue: add bias and write back
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, B_s[j])
                    Out[gm, gn] = T.Cast(in_dtype, val)

    return c_attn_kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper module                                                      #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimised version of the masked self-attention block with a TileLang-powered
    c_attn projection (QKV GEMM). All other ops remain in PyTorch.
    """

    def __init__(self, n_embd: int, n_head: int, max_seqlen: int):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_embd = int(n_embd)
        self.n_head = int(n_head)
        self.max_seqlen = int(max_seqlen)

        out_features = 3 * n_embd
        in_features = n_embd

        # --------------------  Parameters (identical init) -----------------
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # causal mask buffer
        self.register_buffer(
            "bias_mask",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
            persistent=False,
        )

        # Kernel cache  :  key = (M, dtype)
        self._kern_cache: Dict[Tuple[int, torch.dtype], tilelang.PrimFunc] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, M: int, dtype: torch.dtype):
        key = (M, dtype)
        if key not in self._kern_cache:
            tl_dtype = "float16" if dtype == torch.float16 else "float"
            kern = _build_c_attn_kernel(
                M=M,
                K=self.n_embd,
                N=3 * self.n_embd,
                in_dtype=tl_dtype,
            )
            self._kern_cache[key] = kern
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B , T , n_embd)   input embeddings
        """
        B, T, C = x.shape
        hs = C // self.n_head          # per-head dim

        # -------------------- c_attn projection (TileLang) -----------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        W_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        M = B * T
        kernel = self._get_kernel(M, x_fp16.dtype)

        proj_fp16 = kernel(x_fp16.view(M, C), W_fp16, b_fp16)
        proj_fp16 = proj_fp16.view(B, T, 3 * C)

        # Split into q,k,v and convert to fp32 for numerical stability
        q, k, v = proj_fp16.split(C, dim=2)
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)

        # -------------------- reshape to (B, nh, T, hs) --------------------
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # -------------------- scaled dot-product attention -----------------
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.relu(att)                           # ReLU instead of softmax

        y = att @ v                                 # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return y.to(x.dtype)