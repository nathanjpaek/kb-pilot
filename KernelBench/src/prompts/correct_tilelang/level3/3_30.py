"""
Problem Name: 30_SwinTransformerV2
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=10.8 runtime_stats={'mean': 10.8, 'std': 0.0917, 'min': 10.6, 'max': 11.1, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 10.7, 'std': 0.0749, 'min': 10.6, 'max': 11.0, 'num_trials': 100}, 'speedup_ratio': 0.991}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------- #
# TileLang GEMM kernel factory ---------------------------------------- #
# --------------------------------------------------------------------- #
def _build_linear_kernel(
    B: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    io_dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    TileLang kernel computing Y = X @ W.T + Bias
    Shapes:
        X : (B, K)
        W : (N, K)   (row-major, will be transposed inside GEMM)
        Bias : (N,)
        Y : (B, N)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((B, K), io_dtype),
        W: T.Tensor((N, K), io_dtype),
        Bias: T.Tensor((N,), io_dtype),
        Y: T.Tensor((B, N), io_dtype),
    ):
        grid_x = T.ceildiv(N, block_N)
        grid_y = T.ceildiv(B, block_M)

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), io_dtype)
            B_s = T.alloc_shared((block_N, block_K), io_dtype)
            Bias_s = T.alloc_shared((block_N,), io_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            # preload bias slice for this block
            T.copy(Bias[bx * block_N : (bx + 1) * block_N], Bias_s)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                # copy tiles
                T.copy(
                    X[by * block_M : (by + 1) * block_M,
                      ko * block_K : (ko + 1) * block_K],
                    A_s,
                )
                T.copy(
                    W[bx * block_N : (bx + 1) * block_N,
                      ko * block_K : (ko + 1) * block_K],
                    B_s,
                )
                # GEMM (B_s transposed on-the-fly)
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # add bias and write back
            for i, j in T.Parallel(block_M, block_N):
                g_i = by * block_M + i
                g_j = bx * block_N + j
                if (g_i < B) and (g_j < N):
                    val = C_loc[i, j] + Bias_s[j].astype(accum_dtype)
                    Y[g_i, g_j] = T.Cast(io_dtype, val)

    return kernel


# --------------------------------------------------------------------- #
# PyTorch wrapper for Linear layer ------------------------------------ #
# --------------------------------------------------------------------- #
class LinearTile(nn.Module):
    """Drop-in replacement for nn.Linear accelerated via TileLang."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # --- identical parameter initialisation to nn.Linear ---------------
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)
        # -------------------------------------------------------------------

        self._cache: Dict[Tuple[int, torch.dtype], callable] = {}

    def _get_kernel(self, batch: int, dtype: torch.dtype):
        key = (batch, dtype)
        if key not in self._cache:
            self._cache[key] = _build_linear_kernel(
                B=batch,
                K=self.in_features,
                N=self.out_features,
                io_dtype="float16",
            )
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(B, x_fp16.dtype)
        y_fp16 = kernel(x_fp16, w_fp16, b_fp16)
        return y_fp16.to(dtype=x.dtype)


# --------------------------------------------------------------------- #
# Swin Transformer V2 with TileLang head ------------------------------ #
# --------------------------------------------------------------------- #
from typing import Any
# import the original reference model components
# (we assume the full original code is available in the namespace)
# Model, PatchEmbed, BasicLayer, etc. are already defined above.

class ModelNew(Model):
    """
    Swin-Transformer-V2 with TileLang-accelerated final classification head.
    All other layers remain identical to the original implementation.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: Any = [2, 2, 6, 2],
        num_heads: Any = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        pretrained_window_sizes: Any = [0, 0, 0, 0],
        **kwargs,
    ):
        # Build the backbone with an *identity* head
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,          # disable original Linear head
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=pretrained_window_sizes,
            **kwargs,
        )

        # Replace with TileLang Linear head
        self.head_tl = LinearTile(self.num_features, num_classes)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)        # (B, num_features)
        logits = self.head_tl(feats)            # TileLang GEMM
        return logits