"""
Problem Name: 27_SELU_
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.28 runtime_stats={'mean': 4.28, 'std': 0.00747, 'min': 4.27, 'max': 4.29, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.27, 'std': 0.00374, 'min': 4.26, 'max': 4.28, 'num_trials': 100}, 'speedup_ratio': 0.998}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel – vectorized SELU
# y = scale * (x if x > 0 else alpha * (exp(x) - 1))
# ---------------------------------------------------------------------------
@cute.kernel
def _selu_vec_kernel(
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
    Dg = gXv.shape[1][1]    # D / V

    if bi < B and dg < Dg:
        # Slice V-wide view for this thread
        x_view = gXv[(None, (bi, dg))]
        y_view = gYv[(None, (bi, dg))]

        # Fragments for vectorized copy
        x_frag = cute.make_fragment_like(x_view, gXv.element_type)
        y_frag = cute.make_fragment_like(y_view, gXv.element_type)

        # Load → FP32 TensorSSA
        cute.autovec_copy(x_view, x_frag)
        x_vec_f32 = x_frag.load().to(cutlass.Float32)

        # SELU in FP32
        zero  = cutlass.Float32(0.0)
        one   = cutlass.Float32(1.0)
        alpha = cutlass.Float32(1.6732632423543772)
        scale = cutlass.Float32(1.0507009873554805)

        pos_mask   = x_vec_f32 > zero
        neg_branch = alpha * (cute.exp(x_vec_f32) - one)
        selu_f32   = scale * cute.where(pos_mask, x_vec_f32, neg_branch)

        # Store back (cast to original dtype)
        y_frag.store(selu_f32.to(gXv.element_type))
        cute.autovec_copy(y_frag, y_view)


# ---------------------------------------------------------------------------
# Host configuration
# ---------------------------------------------------------------------------
@cute.jit
def _selu_vec_host(
    mX: cute.Tensor,               # (B, D)
    mY: cute.Tensor,               # (B, D)
    V : cutlass.Constexpr,         # vector width (compile-time)
):
    B, D = mX.shape

    # Tile along the contiguous D dimension
    gXv = cute.zipped_divide(mX, (1, V))   # ((1,V), (B, D/V))
    gYv = cute.zipped_divide(mY, (1, V))   # ((1,V), (B, D/V))

    threads_per_block = 256
    D_groups = D // V

    grid_x = cute.ceil_div(D_groups, threads_per_block)
    grid_y = B

    _selu_vec_kernel(gXv, gYv).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
# PyTorch front-end module
# ---------------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated SELU for (B, D) row-major tensors with 128-bit vector I/O.
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    # Choose vector width targeting 128-bit ops
    def _pick_vec_width(self, dtype: torch.dtype, D: int) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8 * 16-bit = 128-bit
        elif dtype == torch.float32:
            V = 4   # 4 * 32-bit = 128-bit
        else:
            V = 2   # fallback (e.g., float64)
        while V > 1 and (D % V):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Expected a 2-D tensor (B, D)"
        # Ensure contiguous CUDA tensor
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()

        B, D = x.shape
        V = self._pick_vec_width(x.dtype, D)

        y = torch.empty_like(x)

        # Wrap tensors for CuTe (row-major (B, D))
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (x.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(_selu_vec_host, mX, mY, V)

        # Launch compiled callable
        self._cache[key](mX, mY)
        return y