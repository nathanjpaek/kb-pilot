"""
Problem Name: 47_Sum_reduction_over_a_dimension
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.67 runtime_stats={'mean': 3.67, 'std': 0.00824, 'min': 3.65, 'max': 3.69, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 3.03, 'std': 0.0123, 'min': 3.0, 'max': 3.06, 'num_trials': 100}, 'speedup_ratio': 0.826}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _sum_reduce_dim1_vecN_kernel(
    gXv: cute.Tensor,    # ((1,V), (B, M, N/V))
    gYv: cute.Tensor,    # ((1,V), (B, 1, N/V))
):
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    b  = bidy                         # batch index
    ng = bidx * bdimx + tidx          # vector-group along N/V

    # Shapes from tiled tensors
    B   = gXv.shape[1][0]
    M   = gXv.shape[1][1]
    Ng  = gXv.shape[1][2]             # N / V

    if (b < B) and (ng < Ng):
        # Output view (1,V) at (b, 0, ng)
        y_view = gYv[(None, (b, 0, ng))]

        # FP32 accumulator TensorSSA initialized to zeros
        acc_frag = cute.make_fragment_like(y_view, cutlass.Float32)
        acc_frag.fill(0.0)
        acc = acc_frag.load()  # TensorSSA(Float32, (1,V))

        # Accumulate over the reduction dimension M
        x_frag = cute.make_fragment_like(y_view, gXv.element_type)
        for m in range(M):
            x_view = gXv[(None, (b, m, ng))]
            cute.autovec_copy(x_view, x_frag)
            x_vec_f32 = x_frag.load().to(cutlass.Float32)
            acc = acc + x_vec_f32

        # Store back to gmem with original dtype
        out_frag = cute.make_fragment_like(y_view, gYv.element_type)
        out_frag.store(acc.to(gYv.element_type))
        cute.autovec_copy(out_frag, y_view)


@cute.jit
def _sum_reduce_dim1_vecN_host(
    mX: cute.Tensor,            # (B, M, N)
    mY: cute.Tensor,            # (B, 1, N)
    V : cutlass.Constexpr,      # vector width (compile-time)
):
    B, M, N = mX.shape

    # Tile along the contiguous N dimension with vector width V
    gXv = cute.zipped_divide(mX, (1, V))   # ((1,V), (B, M, N/V))
    gYv = cute.zipped_divide(mY, (1, V))   # ((1,V), (B, 1, N/V))

    threads_per_block = 256
    N_groups = N // V

    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = B

    _sum_reduce_dim1_vecN_kernel(gXv, gYv).launch(
        grid=(grid_x, grid_y, 1),
        block=(threads_per_block, 1, 1),
    )


class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated sum reduction over dimension=1 for 3D inputs (B, M, N) with keepdim=True.
    Vectorizes along the last (contiguous) dimension N.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self._cache = {}

    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, N: int) -> int:
        # Aim for 128-bit vector width when possible
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8 * 16-bit = 128-bit
        elif dtype == torch.float32:
            V = 4   # 4 * 32-bit = 128-bit
        else:
            V = 2   # conservative fallback
        while V > 1 and (N % V):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect 3D input and reduction over dim=1 as in the example
        assert x.dim() == 3, "Expected a 3-D tensor of shape (B, M, N)"
        assert self.dim == 1, "This implementation reduces over dim=1"

        # Ensure CUDA & contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        B, M, N = x.shape

        V = self._pick_vec_width(x.dtype, N)

        # Allocate output (B, 1, N) with keepdim=True
        y = torch.empty((B, 1, N), dtype=x.dtype, device=x.device)

        # Wrap tensors for CuTe (row-major, dynamic leading dims)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        )

        key = (x.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(_sum_reduce_dim1_vecN_host, mX, mY, V)

        # Launch compiled callable
        self._cache[key](mX, mY)
        return y