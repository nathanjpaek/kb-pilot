"""
Problem Name: 12_Matmul_with_diagonal_matrices_
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0773 runtime_stats={'mean': 0.0773, 'std': 0.00165, 'min': 0.0751, 'max': 0.0872, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.66, 'std': 0.0241, 'min': 2.66, 'max': 2.9, 'num_trials': 100}, 'speedup_ratio': 34.4}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _scale_rows_vecM_kernel(
    gA: cute.Tensor,   # (N,)
    gBv: cute.Tensor,  # ((1,V), (N, M/V))
    gCv: cute.Tensor,  # ((1,V), (N, M/V))
):
    # Thread ↔ coordinates ---------------------------------------------
    tidx, _, _  = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    row = bidy                               #   i  ∈ [0, N)
    group = bidx * bdimx + tidx              #   mg ∈ [0, M/V)

    N_rows    = gCv.shape[1][0]              # from tiled shape ((1,V),(N,Mg))
    M_groups  = gCv.shape[1][1]

    if row < N_rows and group < M_groups:
        # ----------------------------------------------------------------
        # Logical slice for this thread – shape (1,V)
        c_out = gCv[(None, (row, group))]

        # Load B vector (global → register fragment → TensorSSA)
        b_gmem = gBv[(None, (row, group))]
        b_frag = cute.make_fragment_like(c_out, gCv.element_type)
        cute.autovec_copy(b_gmem, b_frag)
        b_vec  = b_frag.load()                              # TensorSSA (V,)

        # Load scaling scalar A[row] and broadcast in arithmetic
        a_val  = gCv.element_type(gA[row])                  # scalar
        res_vec = a_val * b_vec                             # broadcast mul

        # Store back
        out_frag = cute.make_fragment_like(c_out, gCv.element_type)
        out_frag.store(res_vec)
        cute.autovec_copy(out_frag, c_out)


@cute.jit
def _scale_rows_vecM_host(
    mA: cute.Tensor,          # (N,)
    mB: cute.Tensor,          # (N, M)
    mC: cute.Tensor,          # (N, M)
    V:  cutlass.Constexpr,    # vector width
):
    N = mA.shape[0]
    M = mB.shape[1]

    # Tile along the contiguous M dimension
    gBv = cute.zipped_divide(mB, (1, V))    # ((1,V),(N, M/V))
    gCv = cute.zipped_divide(mC, (1, V))

    threads_per_block = 256
    M_groups  = M // V
    grid_x = cute.ceil_div(M_groups, threads_per_block)
    grid_y = N                                 # one block-row per matrix row

    _scale_rows_vecM_kernel(mA, gBv, gCv).launch(
        grid=(grid_x, grid_y, 1),
        block=(threads_per_block, 1, 1)
    )


class ModelNew(torch.nn.Module):
    """
    CuTe implementation of C = diag(A) @ B (row-wise scaling)
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, M: int) -> int:
        # 128-bit vector preference
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (M % V):
            V //= 2
        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous CUDA tensors
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.dim() == 1 and B.dim() == 2
        N = A.shape[0]
        assert B.shape[0] == N, "diag(A) length must match B rows"

        M = B.shape[1]
        V = self._pick_vector_width(B.dtype, M)

        # Output tensor
        C = torch.empty_like(B)

        # Wrap as CuTe tensors (row-major, dynamic leading dim)
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0,)
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        # Compile & cache per (dtype, V)
        key = (A.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _scale_rows_vecM_host, mA, mB, mC, V
            )

        # Launch
        self._cache[key](mA, mB, mC)
        return C