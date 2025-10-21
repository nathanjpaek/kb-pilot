"""
Problem Name: 94_MSELoss
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=9.02 runtime_stats={'mean': 9.02, 'std': 0.0102, 'min': 9.0, 'max': 9.08, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 8.35, 'std': 0.00678, 'min': 8.34, 'max': 8.39, 'num_trials': 100}, 'speedup_ratio': 0.926}}
"""

import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Kernel 1: Per-row MSE over the last dimension
# For inputs P,T of shape (B, D), computes:
#   row_mse[b] = mean_d (P[b,d] - T[b,d])^2
# ---------------------------------------------------------------------------
@cute.kernel
def _mse_row_mean_kernel(
    P: cute.Tensor,                # (B, D)
    T: cute.Tensor,                # (B, D)
    row_mse: cute.Tensor,          # (B,)
    invD: cutlass.Float32,         # 1.0 / D
):
    tidx, _, _  = cute.arch.thread_idx()
    bidx, _, _  = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    b = bidx * bdimx + tidx

    B = P.shape[0]
    D = P.shape[1]

    if b < B:
        acc = cutlass.Float32(0.0)
        for d in range(D):
            pd = P[b, d]              # scalar TensorSSA
            td = T[b, d]
            diff = (pd - td).to(cutlass.Float32)
            acc = acc + diff * diff

        row_mse[b] = (acc * invD).to(row_mse.element_type)


# ---------------------------------------------------------------------------
# Host wrapper for per-row MSE
# ---------------------------------------------------------------------------
@cute.jit
def _mse_row_mean_host(
    mP: cute.Tensor,               # (B, D)
    mT: cute.Tensor,               # (B, D)
    mRow: cute.Tensor,             # (B,)
    invD: cutlass.Float32,         # 1.0 / D
):
    B, D = mP.shape
    threads_per_block = 256
    grid_x = cute.ceil_div(B, threads_per_block)

    _mse_row_mean_kernel(mP, mT, mRow, invD).launch(
        grid  = (grid_x, 1, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
# Kernel 2: Mean of a 1-D tensor (B,) -> scalar
#   out[0] = mean(row_mse)
# Single-thread kernel for simplicity (no atomics).
# ---------------------------------------------------------------------------
@cute.kernel
def _mean1d_kernel(
    v: cute.Tensor,                # (B,)
    out: cute.Tensor,              # (1,)
    invB: cutlass.Float32,         # 1.0 / B
):
    tidx, _, _  = cute.arch.thread_idx()
    bidx, _, _  = cute.arch.block_idx()

    B = v.shape[0]

    if (bidx == 0) and (tidx == 0):
        acc = cutlass.Float32(0.0)
        for i in range(B):
            acc = acc + v[i].to(cutlass.Float32)
        out[0] = (acc * invB).to(out.element_type)


# ---------------------------------------------------------------------------
# Host wrapper for mean of 1-D tensor
# ---------------------------------------------------------------------------
@cute.jit
def _mean1d_host(
    mV: cute.Tensor,               # (B,)
    mOut: cute.Tensor,             # (1,)
    invB: cutlass.Float32,         # 1.0 / B
):
    _mean1d_kernel(mV, mOut, invB).launch(
        grid  = (1, 1, 1),
        block = (1, 1, 1),
    )


# ---------------------------------------------------------------------------
# PyTorch front-end
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    CuTe-accelerated Mean Squared Error (global mean over all elements).
    Works with arbitrary shapes by reshaping to (B, D), computing per-row MSE,
    then averaging across rows.
    """
    def __init__(self):
        super().__init__()
        self._cache_row = {}
        self._cache_mean = {}

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert predictions.shape == targets.shape, "predictions and targets must have the same shape"
        x = predictions.contiguous().cuda() if not predictions.is_cuda else predictions.contiguous()
        y = targets.contiguous().cuda() if not targets.is_cuda else targets.contiguous()
        assert x.is_contiguous() and y.is_contiguous()

        # Reshape to (B, D): keep the last dimension as D
        if x.dim() == 0:
            # Scalar input
            diff = (x - y)
            return (diff * diff).mean()
        elif x.dim() == 1:
            B, D = 1, x.numel()
            x2d = x.view(B, D)
            y2d = y.view(B, D)
        else:
            D = x.shape[-1]
            B = x.numel() // D
            x2d = x.view(B, D)
            y2d = y.view(B, D)

        # Allocate intermediates
        row = torch.empty((B,), dtype=x.dtype, device=x.device)   # per-row MSE
        out = torch.empty((1,), dtype=x.dtype, device=x.device)   # final scalar

        # Wrap tensors for CuTe (row-major)
        mP = from_dlpack(x2d, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mT = from_dlpack(y2d, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mRow = from_dlpack(row, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0,)
        )
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0,)
        )

        invD = cutlass.Float32(1.0 / float(D))
        invB = cutlass.Float32(1.0 / float(B))

        key = (x.dtype,)
        if key not in self._cache_row:
            # Compile row-wise MSE
            self._cache_row[key] = cute.compile(
                _mse_row_mean_host, mP, mT, mRow, cutlass.Float32(1.0)
            )
        if key not in self._cache_mean:
            # Compile 1D mean
            self._cache_mean[key] = cute.compile(
                _mean1d_host, mRow, mOut, cutlass.Float32(1.0)
            )

        # Launch kernels
        self._cache_row[key](mP, mT, mRow, invD)
        self._cache_mean[key](mRow, mOut, invB)

        # Return scalar
        return out[0]


# Suggested harness matching the original snippet
batch_size = 32768
input_shape = (32768,)

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape) * scale, torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []