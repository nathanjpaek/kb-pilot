"""
Problem Name: 30_Softsign
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.27 runtime_stats={'mean': 4.27, 'std': 0.00447, 'min': 4.26, 'max': 4.29, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 14.8, 'std': 0.00366, 'min': 14.8, 'max': 14.8, 'num_trials': 100}, 'speedup_ratio': 3.47}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel: element-wise Softsign with V-wide vectorization
# y = x / (1 + |x|)
# ---------------------------------------------------------------------------
@cute.kernel
def _softsign_vec_kernel(
    gXv: cute.Tensor,   # ((1,V), (B, D/V))
    gYv: cute.Tensor,   # ((1,V), (B, D/V))
):
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi = bidy                         # batch row
    dg = bidx * bdimx + tidx          # vector group along D/V

    B  = gXv.shape[1][0]
    Dg = gXv.shape[1][1]              # D / V

    if bi < B and dg < Dg:
        x_gmem = gXv[(None, (bi, dg))]           # shape (1,V)
        y_gmem = gYv[(None, (bi, dg))]

        # Load vector to registers (autovec, 128-bit when possible)
        x_frag = cute.make_fragment_like(x_gmem, gXv.element_type)
        cute.autovec_copy(x_gmem, x_frag)
        x_vec = x_frag.load().to(cutlass.Float32)

        # Softsign in Float32: x / (1 + |x|)
        zero = cutlass.Float32(0.0)
        one  = cutlass.Float32(1.0)
        abs_x = cute.where(x_vec >= zero, x_vec, -x_vec)
        y_vec = x_vec / (one + abs_x)

        # Store back in original dtype
        y_frag = cute.make_fragment_like(y_gmem, gYv.element_type)
        y_frag.store(y_vec.to(gYv.element_type))
        cute.autovec_copy(y_frag, y_gmem)


# ---------------------------------------------------------------------------
# Host configuration
# ---------------------------------------------------------------------------
@cute.jit
def _softsign_vec_host(
    mX: cute.Tensor,              # (B, D)
    mY: cute.Tensor,              # (B, D)
    V : cutlass.Constexpr,        # vector width (compile-time)
):
    B, D = mX.shape

    # Tile along contiguous D with width V
    gXv = cute.zipped_divide(mX, (1, V))   # ((1,V), (B, D/V))
    gYv = cute.zipped_divide(mY, (1, V))   # ((1,V), (B, D/V))

    threads_per_block = 256
    D_groups = D // V

    grid_x = cute.ceil_div(D_groups, threads_per_block)
    grid_y = B

    _softsign_vec_kernel(gXv, gYv).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
# PyTorch front-end
# ---------------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated element-wise Softsign on a (B, D) tensor.
    Vectorises the contiguous (D) dimension with 128-bit loads/stores.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    # Choose 128-bit vector width
    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, D: int) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8×16-bit = 128-bit
        else:
            V = 4   # 4×32-bit = 128-bit
        while V > 1 and (D % V):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Expected a 2-D tensor (B, D)"
        B, D = x.shape

        # Ensure CUDA & contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()

        V = self._pick_vec_width(x.dtype, D)

        # Allocate output
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
            self._cache[key] = cute.compile(_softsign_vec_host, mX, mY, V)

        # Launch
        self._cache[key](mX, mY)
        return y