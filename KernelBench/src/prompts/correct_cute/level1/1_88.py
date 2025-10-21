"""
Problem Name: 88_MinGPTNewGelu
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.203 runtime_stats={'mean': 0.203, 'std': 0.00183, 'min': 0.199, 'max': 0.213, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.61, 'std': 0.00613, 'min': 1.61, 'max': 1.64, 'num_trials': 100}, 'speedup_ratio': 7.93}}
"""

import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel – vectorized GELU (tanh approximation)
# y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# ---------------------------------------------------------------------------
@cute.kernel
def _gelu_vec_kernel(
    gXv: cute.Tensor,    # ((1,V), (B, D/V))
    gYv: cute.Tensor,    # ((1,V), (B, D/V))
):
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi = bidy
    dg = bidx * bdimx + tidx

    B  = gXv.shape[1][0]
    Dg = gXv.shape[1][1]  # D / V

    if bi < B and dg < Dg:
        x_view = gXv[(None, (bi, dg))]  # (1,V)
        y_view = gYv[(None, (bi, dg))]

        # Fragments for vectorized IO
        x_frag = cute.make_fragment_like(x_view, gXv.element_type)
        y_frag = cute.make_fragment_like(y_view, gYv.element_type)

        # Load → FP32 TensorSSA
        cute.autovec_copy(x_view, x_frag)
        x_f32 = x_frag.load().to(cutlass.Float32)

        # GELU in FP32: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x^3)))
        half  = cutlass.Float32(0.5)
        one   = cutlass.Float32(1.0)
        c0    = cutlass.Float32(0.044715)
        c1    = cutlass.Float32(0.7978845608028654)  # sqrt(2/pi)

        t     = x_f32 * x_f32 * x_f32 * c0 + x_f32
        t     = t * c1
        y_f32 = half * x_f32 * (one + cute.tanh(t))

        # Store back to gmem (cast to original dtype)
        y_frag.store(y_f32.to(gXv.element_type))
        cute.autovec_copy(y_frag, y_view)


# ---------------------------------------------------------------------------
# Host configuration
# ---------------------------------------------------------------------------
@cute.jit
def _gelu_vec_host(
    mX: cute.Tensor,               # (B, D)
    mY: cute.Tensor,               # (B, D)
    V : cutlass.Constexpr,         # vector width (compile-time)
):
    B, D = mX.shape

    # Tile along D with width V
    gXv = cute.zipped_divide(mX, (1, V))   # ((1,V), (B, D/V))
    gYv = cute.zipped_divide(mY, (1, V))   # ((1,V), (B, D/V))

    threads_per_block = 256
    D_groups = D // V

    grid_x = cute.ceil_div(D_groups, threads_per_block)
    grid_y = B

    _gelu_vec_kernel(gXv, gYv).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
# PyTorch front-end module
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    CuTe-accelerated GELU (tanh approximation) on (B, D) row-major tensors.
    Uses 128-bit vectorized loads/stores when possible.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, D: int) -> int:
        # Prefer 128-bit vectors
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8*16-bit = 128-bit
        elif dtype == torch.float32:
            V = 4   # 4*32-bit = 128-bit
        else:
            V = 2   # conservative fallback
        while V > 1 and (D % V):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Expected a 2-D tensor (B, D)"
        # Ensure CUDA + contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()

        B, D = x.shape
        V = self._pick_vec_width(x.dtype, D)

        y = torch.empty_like(x)

        # Wrap for CuTe (row-major (B, D))
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (x.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(_gelu_vec_host, mX, mY, V)

        # Launch compiled callable
        self._cache[key](mX, mY)
        return y


# Suggested harness (matching original)
batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim)]

def get_init_inputs():
    return []