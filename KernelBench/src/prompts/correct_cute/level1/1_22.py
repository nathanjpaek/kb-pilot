"""
Problem Name: 22_Tanh
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.24 runtime_stats={'mean': 4.24, 'std': 0.00291, 'min': 4.24, 'max': 4.26, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.24, 'std': 0.00175, 'min': 4.23, 'max': 4.24, 'num_trials': 100}, 'speedup_ratio': 1.0}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel – vectorised Tanh
# ---------------------------------------------------------------------------
@cute.kernel
def _tanh_vec_kernel(
    gXv: cute.Tensor,    # ((1,V), (B, D/V))
    gYv: cute.Tensor,    # ((1,V), (B, D/V))
):
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    b  = bidy                          # batch row index
    dg = bidx * bdimx + tidx           # vector-group along D/V

    B   = gXv.shape[1][0]              # number of rows
    Dg  = gXv.shape[1][1]              # D / V

    if (b < B) and (dg < Dg):
        # Slice V-wide vector views
        x_vec_gmem = gXv[(None, (b, dg))]   # shape (1, V)
        y_vec_gmem = gYv[(None, (b, dg))]

        # Load V elements from global memory to registers
        x_frag = cute.make_fragment_like(x_vec_gmem, gXv.element_type)
        cute.autovec_copy(x_vec_gmem, x_frag)
        x_vec_f32 = x_frag.load().to(cutlass.Float32)

        # Tanh in Float32
        y_vec_f32 = cute.tanh(x_vec_f32)

        # Store back to global memory with original dtype
        y_frag = cute.make_fragment_like(y_vec_gmem, gXv.element_type)
        y_frag.store(y_vec_f32.to(gXv.element_type))
        cute.autovec_copy(y_frag, y_vec_gmem)


# ---------------------------------------------------------------------------
# Host configuration
# ---------------------------------------------------------------------------
@cute.jit
def _tanh_vec_host(
    mX: cute.Tensor,           # (B, D)
    mY: cute.Tensor,           # (B, D)
    V : cutlass.Constexpr,     # compile-time vector width
):
    B, D = mX.shape

    # Tile along contiguous D dimension with width V
    gXv = cute.zipped_divide(mX, (1, V))   # ((1,V), (B, D/V))
    gYv = cute.zipped_divide(mY, (1, V))   # ((1,V), (B, D/V))

    threads_per_block = 256
    D_groups = D // V

    grid_x = cute.ceil_div(D_groups, threads_per_block)
    grid_y = B

    _tanh_vec_kernel(gXv, gYv).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
# PyTorch front-end module
# ---------------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe implementation of element-wise Tanh with vectorised loads/stores.
    Works on (B, D) row-major tensors and picks a 128-bit vector width.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, D: int) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8 x 16-bit = 128-bit
        elif dtype == torch.float32:
            V = 4   # 4 x 32-bit = 128-bit
        else:
            V = 2   # conservative fallback
        while V > 1 and (D % V):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Expected a 2-D tensor (B, D)"
        # Ensure CUDA & contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        B, D = x.shape

        V = self._pick_vec_width(x.dtype, D)
        y = torch.empty_like(x)

        # Wrap tensors for CuTe (row-major, dynamic leading dim)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (x.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(_tanh_vec_host, mX, mY, V)

        # Launch compiled kernel
        self._cache[key](mX, mY)
        return y