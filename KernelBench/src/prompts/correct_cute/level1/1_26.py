"""
Problem Name: 26_GELU_
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.27 runtime_stats={'mean': 4.27, 'std': 0.00255, 'min': 4.27, 'max': 4.28, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.27, 'std': 0.00219, 'min': 4.26, 'max': 4.27, 'num_trials': 100}, 'speedup_ratio': 1.0}}
"""

import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel – vectorised GELU (tanh approximation)
# ---------------------------------------------------------------------------
@cute.kernel
def _gelu_vec_kernel(
    gXv: cute.Tensor,    # ((1,V), (B, D/V)) – tiled input
    gYv: cute.Tensor,    # ((1,V), (B, D/V)) – tiled output
):
    # CUDA indices -----------------------------------------------------------
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi  = bidy                      # batch row
    dg  = bidx * bdimx + tidx       # dim-vector group index

    B  = gXv.shape[1][0]
    Dg = gXv.shape[1][1]            # D / V

    if bi < B and dg < Dg:
        # -------------------------------------------------------------------
        # 1. Load X vector from global memory
        # -------------------------------------------------------------------
        x_gmem   = gXv[(None, (bi, dg))]               # (1,V) view
        x_frag   = cute.make_fragment_like(x_gmem, gXv.element_type)
        cute.autovec_copy(x_gmem, x_frag)
        x_vec_f32 = x_frag.load().to(cutlass.Float32)  # TensorSSA(Float32)

        # -------------------------------------------------------------------
        # 2. GELU computation in Float32
        #     y = 0.5 * x * (1 + tanh(√(2/π)*(x + 0.044715*x^3)))
        # -------------------------------------------------------------------
        half          = cutlass.Float32(0.5)
        one           = cutlass.Float32(1.0)
        sqrt_2_over_pi= cutlass.Float32(0.7978845608028654)   # √(2/π)
        coeff         = cutlass.Float32(0.044715)

        t     = x_vec_f32 * x_vec_f32 * x_vec_f32 * coeff + x_vec_f32
        t     = t * sqrt_2_over_pi
        y_vec = half * x_vec_f32 * (one + cute.tanh(t))

        # -------------------------------------------------------------------
        # 3. Store back to global memory
        # -------------------------------------------------------------------
        y_frag = cute.make_fragment_like(x_gmem, gXv.element_type)
        y_frag.store(y_vec.to(gXv.element_type))
        cute.autovec_copy(y_frag, gYv[(None, (bi, dg))])


# ---------------------------------------------------------------------------
# Host configuration
# ---------------------------------------------------------------------------
@cute.jit
def _gelu_vec_host(
    mX: cute.Tensor,                # (B, D)
    mY: cute.Tensor,                # (B, D)
    V : cutlass.Constexpr,          # vector width
):
    B = mX.shape[0]
    D = mX.shape[1]

    # Tile innermost dimension with width V
    gXv = cute.zipped_divide(mX, (1, V))   # ((1,V), (B, D/V))
    gYv = cute.zipped_divide(mY, (1, V))   # ((1,V), (B, D/V))

    threads_per_block = 256
    D_groups = D // V
    grid_x = cute.ceil_div(D_groups, threads_per_block)
    grid_y = B

    _gelu_vec_kernel(gXv, gYv).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


# ---------------------------------------------------------------------------
# Torch-front module
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    CuTe implementation of element-wise GELU with vectorised loads/stores.
    Input / output tensors are (B, D) row-major.
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    # pick 128-bit vector width
    def _pick_vec_width(self, dtype: torch.dtype, D: int) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            V = 8     # 8×16  = 128 bit
        else:
            V = 4     # 4×32  = 128 bit
        while V > 1 and (D % V):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Expected 2-D (batch, dim) input"
        B, D = x.shape

        # Ensure contiguous CUDA tensor
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()

        V = self._pick_vec_width(x.dtype, D)

        y = torch.empty_like(x)

        # Wrap with CuTe -----------------------------------------------------
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)      # row-major (B,D)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (x.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(_gelu_vec_host, mX, mY, V)

        # launch
        self._cache[key](mX, mY)
        return y