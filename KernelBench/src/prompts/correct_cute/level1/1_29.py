"""
Problem Name: 29_Softplus
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.27 runtime_stats={'mean': 4.27, 'std': 0.0113, 'min': 4.25, 'max': 4.36, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.62, 'std': 0.712, 'min': 4.23, 'max': 6.68, 'num_trials': 100}, 'speedup_ratio': 1.08}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
#                          Device kernel
# ---------------------------------------------------------------------------
@cute.kernel
def _softplus_vec_kernel(
    gXv: cute.Tensor,     # ((1,V), (B, D/V))
    gYv: cute.Tensor,     # ((1,V), (B, D/V))
):
    # CUDA indices -----------------------------------------------------------
    tidx, _, _      = cute.arch.thread_idx()
    bidx, bidy, _   = cute.arch.block_idx()   # x , y , z
    bdimx, _, _     = cute.arch.block_dim()

    b  = bidy                       # batch row
    dg = bidx * bdimx + tidx        # vector-group along D

    B   = gYv.shape[1][0]           # batch size
    Dg  = gYv.shape[1][1]           # D / V

    if (b < B) and (dg < Dg):
        # ---------------- Load V values ------------------------------------
        x_vec_gmem = gXv[(None, (b, dg))]             # shape (1,V)

        x_frag = cute.make_fragment_like(x_vec_gmem, gXv.element_type)
        cute.autovec_copy(x_vec_gmem, x_frag)         # gmem → regs
        x_vec_f32 = x_frag.load().to(cutlass.Float32) # TensorSSA FP32

        # ---------------- Softplus -----------------------------------------
        one      = cutlass.Float32(1.0)
        soft_vec = cute.log(one + cute.exp(x_vec_f32))  # ln(1+exp(x))

        # ---------------- Store --------------------------------------------
        y_frag = cute.make_fragment_like(x_vec_gmem, gXv.element_type)
        y_frag.store(soft_vec.to(gXv.element_type))    # back-convert
        y_vec_gmem = gYv[(None, (b, dg))]
        cute.autovec_copy(y_frag, y_vec_gmem)          # regs → gmem


# ---------------------------------------------------------------------------
#                          Host configuration
# ---------------------------------------------------------------------------
@cute.jit
def _softplus_vec_host(
    mX: cute.Tensor,          # (B, D)
    mY: cute.Tensor,          # (B, D)
    V : cutlass.Constexpr,    # vector width (compile-time)
):
    B, D = mX.shape

    # 1) Tile tensors along D with width V ----------------------------------
    gXv = cute.zipped_divide(mX, (1, V))   # ((1,V), (B, D/V))
    gYv = cute.zipped_divide(mY, (1, V))   # ((1,V), (B, D/V))

    # 2) Launch configuration ----------------------------------------------
    threads_per_block = 256
    D_groups          = D // V

    grid_x = cute.ceil_div(D_groups, threads_per_block)
    grid_y = B

    _softplus_vec_kernel(gXv, gYv).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
#                          PyTorch front-end
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    CuTe-accelerated element-wise Softplus on a (B, D) tensor.
    Vectorises the contiguous (D) dimension with 128-bit loads/stores.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    # Choose 128-bit vector width -------------------------------------------
    def _pick_vec_width(self, dtype: torch.dtype, D: int) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8×16-bit = 128-bit
        else:
            V = 4   # 4×32-bit = 128-bit
        while V > 1 and (D % V):
            V //= 2
        return V

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Expected a 2-D tensor (B, D)"
        B, D = x.shape

        # Ensure CUDA & contiguous ------------------------------------------
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()

        V = self._pick_vec_width(x.dtype, D)

        # Allocate output ----------------------------------------------------
        y = torch.empty_like(x)

        # Wrap for CuTe ------------------------------------------------------
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)   # row-major (B, D)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        # Compile / cache per (dtype, V) ------------------------------------
        key = (x.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _softplus_vec_host, mX, mY, V
            )

        # Launch -------------------------------------------------------------
        self._cache[key](mX, mY)

        return y