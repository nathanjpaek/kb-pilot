"""
Problem Name: 21_Sigmoid
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.27 runtime_stats={'mean': 4.27, 'std': 0.00274, 'min': 4.26, 'max': 4.28, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.26, 'std': 0.00209, 'min': 4.26, 'max': 4.27, 'num_trials': 100}, 'speedup_ratio': 0.998}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _sigmoid_vecD_kernel(
    gXv: cute.Tensor,   # tiled, V-wide view of input
    gYv: cute.Tensor,   # tiled, V-wide view of output
):
    # CUDA indices
    tidx, _, _  = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    # Logical coordinates
    mi = bidy                                       # batch     (B)
    ng = bidx * bdimx + tidx                        # col group (D/V)

    # Shapes after tiling: ((1,V), (B, D/V))
    B       = gXv.shape[1][0]
    N_groups = gXv.shape[1][1]

    if mi < B and ng < N_groups:
        # Slice V-element vector views
        x_vec_gmem = gXv[(None, (mi, ng))]
        y_vec_gmem = gYv[(None, (mi, ng))]

        # Temporary register fragments for vector copy
        x_frag = cute.make_fragment_like(x_vec_gmem, gXv.element_type)
        y_frag = cute.make_fragment_like(y_vec_gmem, gYv.element_type)

        # Load vector from global → register   (128-bit autovec)
        cute.autovec_copy(x_vec_gmem, x_frag)
        x_vec = x_frag.load().to(cutlass.Float32)   # promote to FP32

        # Sigmoid:  1 / (1 + exp(-x))
        neg_x   = -x_vec
        exp_val = cute.exp(neg_x)
        sig     = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + exp_val)

        # Cast back to original type & store
        y_frag.store(sig.to(gXv.element_type))
        cute.autovec_copy(y_frag, y_vec_gmem)


@cute.jit
def _sigmoid_vecD_host(
    mX: cute.Tensor,
    mY: cute.Tensor,
    V: cutlass.Constexpr,
):
    """
    Host configuration for vectorised sigmoid.
    V – compile-time constant vector width
    """
    B, D = mX.shape
    gXv = cute.zipped_divide(mX, (1, V))   # ((1,V), (B,D/V))
    gYv = cute.zipped_divide(mY, (1, V))   # same layout

    threads_per_block = 256
    N_groups = D // V
    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = B

    _sigmoid_vecD_kernel(gXv, gYv).launch(
        grid=(grid_x, grid_y, 1),
        block=(threads_per_block, 1, 1),
    )


class ModelNew(nn.Module):
    """
    CuTe implementation of point-wise sigmoid with 128-bit vector accesses.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, D: int) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            V = 8      # 8×16 = 128 bits
        else:
            V = 4      # default (FP32, others)
        while V > 1 and (D % V != 0):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move to CUDA, make contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        B, D = x.shape

        V = self._pick_vector_width(x.dtype, D)

        # Allocate output
        y = torch.empty_like(x)

        # Wrap in CuTe tensors (row-major, stride_order=(0,1))
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (x.dtype, V)
        if key not in self._cache:
            # Compile kernel once per (dtype, V)
            self._cache[key] = cute.compile(_sigmoid_vecD_host, mX, mY, V)

        # Launch compiled callable
        self._cache[key](mX, mY)
        return y