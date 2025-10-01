"""
Problem Name: 1_SumAggregator
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0714 runtime_stats={'mean': 0.0714, 'std': 0.0492, 'min': 0.0313, 'max': 0.218, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0218, 'std': 0.0199, 'min': 0.0133, 'max': 0.123, 'num_trials': 100}, 'speedup_ratio': 0.305}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def sum_over_dim1(
    B,
    N,
    H,
    W,
    block_H=8,
    block_W=32,
    threads=256,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float32",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((B, N, H, W), in_dtype),
        O: T.Tensor((B, H, W), out_dtype),
    ):
        grid_x = T.ceildiv(W, block_W)
        grid_y = T.ceildiv(H, block_H)

        with T.Kernel(grid_x, grid_y, B, threads=threads) as (bx, by, bz):
            start_h = by * block_H
            start_w = bx * block_W

            # Local accumulator in registers
            acc_tile = T.alloc_fragment((block_H, block_W), accum_dtype)
            T.clear(acc_tile)

            # Accumulate along N dimension
            for n in range(N):
                for lh, lw in T.Parallel(block_H, block_W):
                    h_idx = start_h + lh
                    w_idx = start_w + lw
                    if (h_idx < H) and (w_idx < W):
                        acc_tile[lh, lw] += T.cast(
                            A[bz, n, h_idx, w_idx], accum_dtype
                        )

            # Write results back
            for lh, lw in T.Parallel(block_H, block_W):
                h_idx = start_h + lh
                w_idx = start_w + lw
                if (h_idx < H) and (w_idx < W):
                    O[bz, h_idx, w_idx] = T.cast(acc_tile[lh, lw], out_dtype)

    return main


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.block_H = 8
        self.block_W = 32
        self.threads = self.block_H * self.block_W
        self._kernel_cache = {}

    def forward(self, neighbor: torch.Tensor) -> torch.Tensor:
        neighbor = neighbor.to(device="cuda", dtype=torch.float16)
        B, N, H, W = neighbor.shape
        key = (B, N, H, W, neighbor.dtype)

        if key not in self._kernel_cache:
            self._kernel_cache[key] = sum_over_dim1(
                B,
                N,
                H,
                W,
                self.block_H,
                self.block_W,
                self.threads,
            )

        sum_kernel = self._kernel_cache[key]
        return sum_kernel(neighbor).to(torch.float32)