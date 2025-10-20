"""
Problem Name: 20_LeakyReLU
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.27 runtime_stats={'mean': 4.27, 'std': 0.0267, 'min': 4.26, 'max': 4.52, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.25, 'std': 0.00777, 'min': 4.24, 'max': 4.32, 'num_trials': 100}, 'speedup_ratio': 0.995}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# --------------------------------------------
# Device kernel
# --------------------------------------------
@cute.kernel
def _leaky_relu_vec_kernel(
    gXv: cute.Tensor,          # ((1,V), (M, N_groups))
    gYv: cute.Tensor,          # same layout as gXv
    negative_slope: cutlass.Float32,
):
    # Thread / block coordinates
    tidx, _, _   = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _  = cute.arch.block_dim()

    # Logical coordinates
    mi = bidy                              # row (batch element)
    ng = bidx * bdimx + tidx               # vector group in dim

    M         = gXv.shape[1][0]            # number of rows
    N_groups  = gXv.shape[1][1]            # dim / V

    # Guard against partial blocks (when dim not multiple of 256*V)
    if mi < M and ng < N_groups:
        x_view = gXv[(None, (mi, ng))]     # shape (1,V) after slice

        # ---- load --------
        x_frag_gmem = cute.make_fragment_like(x_view, gXv.element_type)
        cute.autovec_copy(x_view, x_frag_gmem)
        x_vec_f32 = x_frag_gmem.load().to(cutlass.Float32)

        # ---- LeakyReLU math in Float32 --------
        zero     = cutlass.Float32(0.0)
        pos_mask = x_vec_f32 > zero
        y_vec_f32 = cute.where(
            pos_mask, 
            x_vec_f32, 
            x_vec_f32 * negative_slope
        )

        # ---- store --------
        y_frag_gmem = cute.make_fragment_like(x_view, gXv.element_type)
        y_frag_gmem.store(y_vec_f32.to(gXv.element_type))
        cute.autovec_copy(y_frag_gmem, gYv[(None, (mi, ng))])


# --------------------------------------------
# Host wrapper
# --------------------------------------------
@cute.jit
def _leaky_relu_host(
    mX: cute.Tensor,
    mY: cute.Tensor,
    V: cutlass.Constexpr,                  # vector width (compile-time)
    negative_slope: cutlass.Float32,       # runtime parameter
):
    M = mX.shape[0]
    N = mX.shape[1]

    # Vectorise along the contiguous dim
    gXv = cute.zipped_divide(mX, (1, V))   # ((1,V), (M, N/V))
    gYv = cute.zipped_divide(mY, (1, V))

    threads_per_block = 256
    N_groups  = N // V
    grid_x    = cute.ceil_div(N_groups, threads_per_block)
    grid_y    = M

    _leaky_relu_vec_kernel(
        gXv, gYv, negative_slope
    ).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


# --------------------------------------------
# PyTorch module façade
# --------------------------------------------
class ModelNew(nn.Module):
    """
    CuTe implementation of LeakyReLU.
    Vectorises along the innermost (contiguous) dimension for 128-bit gmem ops.
    """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)
        self._cache = {}

    # Choose vector width for 128-bit accesses
    def _pick_vector_width(self, dtype: torch.dtype, dim: int) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            V = 8          # 8 * 16  = 128 bits
        else:
            V = 4          # 4 * 32  = 128 bits

        while V > 1 and (dim % V != 0):    # make V divide dim
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous CUDA tensor
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        assert x.dim() == 2, "Input must be 2-D (batch, dim)"

        M, N = x.shape
        V = self._pick_vector_width(x.dtype, N)
        assert N % V == 0, "Vector width does not divide dim"

        y = torch.empty_like(x)

        # Wrap for CuTe (row-major with dynamic leading dim)
        mX = from_dlpack(x , assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mY = from_dlpack(y , assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (x.dtype, V)                 # slope is runtime, so not in key
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _leaky_relu_host, mX, mY, V, cutlass.Float32(self.negative_slope)
            )

        # Launch
        self._cache[key](
            mX, mY, cutlass.Float32(self.negative_slope)
        )
        return y