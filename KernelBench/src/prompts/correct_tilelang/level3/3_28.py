"""
Problem Name: 28_VisionTransformer
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.51 runtime_stats={'mean': 2.51, 'std': 0.0314, 'min': 2.46, 'max': 2.58, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.35, 'std': 0.0375, 'min': 2.29, 'max': 2.5, 'num_trials': 100}, 'speedup_ratio': 0.936}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                      Generic   M  ×  K   @   N × K   GEMM                   #
# --------------------------------------------------------------------------- #
def _build_linear_kernel(
    M: int,
    K: int,
    N: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    tot = M * N
    grid = (tot + threads_per_block - 1) // threads_per_block

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),          # input matrix
        W: T.Tensor((N, K), dtype),          # weight  (out, in)
        B: T.Tensor((N,), dtype),            # bias
        O: T.Tensor((M, N), dtype),          # output
    ):
        zero = T.Cast(accum_dtype, 0)

        with T.Kernel(grid, threads=threads_per_block) as bx:
            tid = T.get_thread_binding(0)
            idx = bx * threads_per_block + tid
            if idx < tot:
                m = idx // N
                n = idx - m * N        # idx % N (avoid div)

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = zero

                for k in T.serial(K):
                    acc[0] += (
                        T.Cast(accum_dtype, A[m, k])
                        * T.Cast(accum_dtype, W[n, k])
                    )

                acc[0] = acc[0] + T.Cast(accum_dtype, B[n])
                O[m, n] = T.Cast(dtype, acc[0])

    return kernel


# --------------------------------------------------------------------------- #
#                               Model  New                                    #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Vision-Transformer with TileLang-accelerated Linear layers
    (patch-embedding and two MLP-head projections).
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        # ------------------------- learnable params ----------------------- #
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # ---- patch-to-embedding (Linear) params --- identical init --------
        self.pe_weight = nn.Parameter(torch.empty(dim, patch_dim))
        self.pe_bias   = nn.Parameter(torch.empty(dim))
        nn.init.kaiming_uniform_(self.pe_weight, a=math.sqrt(5))
        bound_pe = 1 / math.sqrt(patch_dim)
        nn.init.uniform_(self.pe_bias, -bound_pe, bound_pe)

        # ------------------ transformer encoder (PyTorch) ------------------ #
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout
            ),
            num_layers=depth,
        )

        # -------------------- MLP-head (2 × Linear) ------------------------ #
        self.fc1_weight = nn.Parameter(torch.empty(mlp_dim, dim))
        self.fc1_bias   = nn.Parameter(torch.empty(mlp_dim))
        nn.init.kaiming_uniform_(self.fc1_weight, a=math.sqrt(5))
        bound1 = 1 / math.sqrt(dim)
        nn.init.uniform_(self.fc1_bias, -bound1, bound1)

        self.fc2_weight = nn.Parameter(torch.empty(num_classes, mlp_dim))
        self.fc2_bias   = nn.Parameter(torch.empty(num_classes))
        nn.init.kaiming_uniform_(self.fc2_weight, a=math.sqrt(5))
        bound2 = 1 / math.sqrt(mlp_dim)
        nn.init.uniform_(self.fc2_bias, -bound2, bound2)

        # Kernel cache :  {(M,K,N,dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, M: int, K: int, N: int, dtype: str = "float16"):
        key = (M, K, N, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_linear_kernel(
                M, K, N, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def _run_linear(
        self,
        x_2d: torch.Tensor,          # (M,K) on CUDA / fp16
        weight: nn.Parameter,
        bias:   nn.Parameter,
    ) -> torch.Tensor:              # returns (M,N) fp16
        M, K = x_2d.shape
        N = weight.shape[0]

        w_fp16 = weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(M, K, N, "float16")
        return kernel(x_2d, w_fp16, b_fp16)

    # ------------------------------------------------------------------ #
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        dtype_in = img.dtype
        B, C, H, W = img.shape
        p = self.patch_size

        # --------------------- patch extraction -------------------------- #
        patches = (
            img.unfold(2, p, p)
            .unfold(3, p, p)
            .reshape(B, -1, p * p * C)
        )
        num_patches = patches.shape[1]
        M_pe = B * num_patches

        # --------------- patch-to-embedding  (TileLang) ------------------ #
        x_pe = patches.to(device="cuda", dtype=torch.float16).contiguous()
        x_pe = x_pe.view(M_pe, -1)                    # (M_pe, patch_dim)
        x_pe = self._run_linear(x_pe, self.pe_weight, self.pe_bias)
        x_pe = x_pe.view(B, num_patches, -1)          # (B, P, dim)

        # ----------------- add CLS token & pos-embedding ----------------- #
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tok, x_pe.to(dtype_in)), dim=1)  # back to original dtype
        x = x + self.pos_embedding[:, : x.shape[1]]
        x = self.dropout(x)

        # ---------------------- transformer encoder ---------------------- #
        x = self.transformer(x)

        # ---------------------- take CLS token --------------------------- #
        x_cls = x[:, 0]  # (B, dim)

        # ------------------------ MLP head ------------------------------- #
        # first Linear (TileLang) + GELU
        y1_in = x_cls.to(device="cuda", dtype=torch.float16).contiguous()
        y1 = self._run_linear(y1_in, self.fc1_weight, self.fc1_bias)
        y1 = F.gelu(y1.to(dtype_in))
        # second Linear (TileLang)
        y1_fp16 = y1.to(device="cuda", dtype=torch.float16).contiguous()
        y2 = self._run_linear(y1_fp16, self.fc2_weight, self.fc2_bias)

        return y2.to(dtype_in)