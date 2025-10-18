"""
Problem Name: 25_Swish
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.25 runtime_stats={'mean': 4.25, 'std': 0.00355, 'min': 4.25, 'max': 4.27, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 10.5, 'std': 0.00472, 'min': 10.5, 'max': 10.5, 'num_trials': 100}, 'speedup_ratio': 2.47}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel : element-wise Swish with V-wide vectorization
# ---------------------------------------------------------------------------
@cute.kernel
def _swish_vec_kernel(
    gXv : cute.Tensor,     # ((1,V), (B, D/V))
    gYv : cute.Tensor,     # ((1,V), (B, D/V))
):
    # CUDA indices
    tidx, _, _  = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    # Logical coordinates
    bi = bidy                       # batch row
    ng = bidx * bdimx + tidx        # vector-group along D/V

    B         = gYv.shape[1][0]     # number of rows
    D_groups  = gYv.shape[1][1]     # D / V

    if bi < B and ng < D_groups:
        x_view = gXv[(None, (bi, ng))]
        y_view = gYv[(None, (bi, ng))]

        # fragments for aligned vector copy
        frag_in  = cute.make_fragment_like(x_view, gXv.element_type)
        frag_out = cute.make_fragment_like(y_view, gXv.element_type)

        cute.autovec_copy(x_view, frag_in)           # load -> frag_in
        x_vec = frag_in.load().to(cutlass.Float32)   # TensorSSA (1,V) FP32

        # Swish = x * sigmoid(x) with element-wise ops
        sig = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-x_vec))
        y_vec_f32 = x_vec * sig

        # store back, converting to original dtype
        frag_out.store(y_vec_f32.to(gXv.element_type))
        cute.autovec_copy(frag_out, y_view)


# ---------------------------------------------------------------------------
# Host configuration
# ---------------------------------------------------------------------------
@cute.jit
def _swish_vec_host(
    mX : cute.Tensor,            # (B, D)
    mY : cute.Tensor,            # (B, D)
    V  : cutlass.Constexpr,      # vector width
):
    B, D = mX.shape
    gXv = cute.zipped_divide(mX, (1, V))   # ((1,V), (B, D/V))
    gYv = cute.zipped_divide(mY, (1, V))   # ((1,V), (B, D/V))

    threads_per_block = 256
    D_groups = D // V
    grid_x = cute.ceil_div(D_groups, threads_per_block)
    grid_y = B

    _swish_vec_kernel(gXv, gYv).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


# ---------------------------------------------------------------------------
# Torch-front model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    CuTe implementation of Swish:  y = x * sigmoid(x)
    Works on any (B, D) input, vectorises over the contiguous D dimension.
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    # Choose 128-bit vector width
    def _pick_vec_width(self, dtype: torch.dtype, D: int) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8×16-bit = 128-bit
        elif dtype == torch.float32:
            V = 4   # 4×32-bit = 128-bit
        else:
            V = 2   # fallback (e.g., FP64: 2×64-bit)

        while V > 1 and (D % V):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Make contiguous CUDA tensor
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        assert x.dim() == 2, "expected (B, D) tensor"

        B, D = x.shape
        V = self._pick_vec_width(x.dtype, D)

        y = torch.empty_like(x)

        # Wrap for CuTe
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)   # row-major (B, D)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (x.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _swish_vec_host, mX, mY, V
            )

        # launch
        self._cache[key](mX, mY)
        return y