"""
Problem Name: 31_VisionAttention
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.57 runtime_stats={'mean': 1.57, 'std': 0.0705, 'min': 1.52, 'max': 1.94, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 8.96, 'std': 0.399, 'min': 8.8, 'max': 10.5, 'num_trials': 100}, 'speedup_ratio': 5.71}}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------- #
#   TileLang GEMM + bias kernel factory
# --------------------------------------------------------------------- #
def _build_gemm_bias_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Returns a TileLang kernel computing   C = A @ B + bias
      A : (M, K)   fp16
      B : (K, N)   fp16
      bias : (N,)  fp16
      C : (M, N)   fp16
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gemm_bias(
        A:    T.Tensor((M, K), dtype),
        B:    T.Tensor((K, N), dtype),
        bias: T.Tensor((N,),   dtype),
        C:    T.Tensor((M, N), dtype),
    ):
        grid_x = T.ceildiv(N, block_N)
        grid_y = T.ceildiv(M, block_M)

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            # Shared tiles
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)

            # Accumulator in registers (float32)
            C_acc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_acc)

            num_k_tiles = T.ceildiv(K, block_K)

            for ko in T.Pipelined(num_k_tiles, num_stages=3):
                # Global offsets
                m_base = by * block_M
                n_base = bx * block_N
                k_base = ko * block_K

                # Load tiles to shared memory
                T.copy(A[m_base, k_base], A_s)
                T.copy(B[k_base, n_base], B_s)

                # Accumulate
                T.gemm(A_s, B_s, C_acc)

            # Add bias and write back
            for mi, ni in T.Parallel(block_M, block_N):
                g_m = by * block_M + mi
                g_n = bx * block_N + ni
                if (g_m < M) and (g_n < N):
                    val = C_acc[mi, ni] + T.Cast(accum_dtype, bias[g_n])
                    C[g_m, g_n] = T.Cast(dtype, val)

    return gemm_bias


# --------------------------------------------------------------------- #
#                       PyTorch wrapper Module
# --------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Multi-Head Self-Attention + residual + LayerNorm
    rewritten to use TileLang GEMM kernels for the two
    linear projections (QKV and output).
    """

    def __init__(self, embed_dim: int, num_heads: int, eps: float = 1e-5):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.eps        = eps

        # -------- in-proj (QKV) ------------------------------------------------
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias   = nn.Parameter(torch.empty(3 * embed_dim))
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)

        # -------- out-proj -----------------------------------------------------
        self.out_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.out_proj_bias   = nn.Parameter(torch.empty(embed_dim))
        nn.init.xavier_uniform_(self.out_proj_weight)
        nn.init.constant_(self.out_proj_bias, 0.)

        # -------- layer-norm ---------------------------------------------------
        self.ln_weight = nn.Parameter(torch.ones(embed_dim))
        self.ln_bias   = nn.Parameter(torch.zeros(embed_dim))

        # Kernel cache  {(M,N,K,dtype) : compiled_kernel}
        self._kernels = {}

    # ----------------------------------------------------------------- #
    def _get_kernel(self, M: int, N: int, K: int, dtype: str):
        key = (M, N, K, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_gemm_bias_kernel(M, N, K, dtype=dtype)
        return self._kernels[key]

    # ----------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        """
        orig_dtype = x.dtype
        B, C, H, W = x.shape
        S          = H * W        # sequence length
        M          = S * B        # rows for GEMM

        # Input -> (S, B, C)   then flatten to 2-D (M, C)
        x_seq = (
            x.view(B, C, S)
             .permute(2, 0, 1)
             .contiguous()
             .to(device="cuda", dtype=torch.float16)
        )
        x_flat = x_seq.view(M, C)

        # ---------------------- QKV projection (TileLang) --------------------
        W_in  = self.in_proj_weight.t().to(device="cuda", dtype=torch.float16)
        b_in  = self.in_proj_bias.to(device="cuda", dtype=torch.float16)
        kernel_qkv = self._get_kernel(M, 3 * C, C, "float16")
        qkv_flat = kernel_qkv(x_flat, W_in, b_in)        # (M, 3C)
        qkv_seq  = qkv_flat.view(S, B, 3 * C)

        q, k, v = qkv_seq.split(C, dim=-1)               # each (S, B, C)

        # Reshape to  (B, num_heads, S, head_dim)
        def _prep(t):
            return (
                t.permute(1, 0, 2)                                           # (B, S, C)
                 .view(B, S, self.num_heads, self.head_dim)
                 .permute(0, 2, 1, 3)                                        # (B, H, S, Dh)
                 .contiguous()
            )

        q_t = _prep(q)
        k_t = _prep(k)
        v_t = _prep(v)

        # ------------------ scaled-dot-product attention ---------------------
        attn = F.scaled_dot_product_attention(
            q_t, k_t, v_t, dropout_p=0.0, is_causal=False
        )                                                # (B, H, S, Dh)

        # Back to  (S, B, C)
        attn_seq = (
            attn.permute(0, 2, 1, 3)                     # (B, S, H, Dh)
                 .reshape(B, S, C)
                 .permute(1, 0, 2)
                 .contiguous()
        )

        # --------------------- output projection (TileLang) -------------------
        W_out = self.out_proj_weight.t().to(device="cuda", dtype=torch.float16)
        b_out = self.out_proj_bias.to(device="cuda", dtype=torch.float16)
        kernel_out = self._get_kernel(M, C, C, "float16")
        attn_out_flat = kernel_out(attn_seq.view(M, C), W_out, b_out)
        attn_out_seq  = attn_out_flat.view(S, B, C)

        # --------------------- residual + layer-norm --------------------------
        residual = x_seq                                  # original input (fp16)
        y = attn_out_seq + residual                       # (S, B, C)

        y_norm = F.layer_norm(
            y,
            normalized_shape=(self.embed_dim,),
            weight=self.ln_weight.to(dtype=y.dtype, device=y.device),
            bias=self.ln_bias.to(dtype=y.dtype, device=y.device),
            eps=self.eps,
        )                                                 # (S, B, C)

        # Back to  (B, C, H, W)
        out = (
            y_norm.permute(1, 2, 0)
                  .view(B, C, H, W)
                  .to(orig_dtype)
        )
        return out