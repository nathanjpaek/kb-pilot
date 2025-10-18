"""
Problem Name: 16_Matmul_with_transposed_A
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=17.9 runtime_stats={'mean': 17.9, 'std': 0.204, 'min': 17.6, 'max': 18.6, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.63, 'std': 0.00874, 'min': 2.61, 'max': 2.65, 'num_trials': 100}, 'speedup_ratio': 0.147}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel – identical to Example 1 except A is accessed as A[k, m]
# ---------------------------------------------------------------------------
@cute.kernel
def _matmul_Atrans_vecN_kernel(
    gA : cute.Tensor,   # 2-D tensor, physical shape (K, M)
    gBv: cute.Tensor,   # ((1,V), (K, N/V))
    gCv: cute.Tensor,   # ((1,V), (M, N/V))
    K  : cutlass.Int32,
):
    # CUDA indices
    tidx, _, _  = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    # Logical position handled by this thread
    m  = bidy                      # row of C
    ng = bidx * bdimx + tidx       # N-vector group index

    M        = gCv.shape[1][0]     # rows
    N_groups = gCv.shape[1][1]     # N / V

    if m < M and ng < N_groups:
        c_out = gCv[(None, (m, ng))]            # output view (1,V)

        b_frag       = cute.make_fragment_like(c_out, gA.element_type)
        acc_frag_f32 = cute.make_fragment_like(c_out, cutlass.Float32)
        acc          = acc_frag_f32.load()      # zeros

        # K-loop
        for k in range(K):
            # NOTE: transposed access – physical A[k, m]
            a_val = cutlass.Float32(gA[k, m])

            b_vec_gmem = gBv[(None, (k, ng))]
            cute.autovec_copy(b_vec_gmem, b_frag)
            b_vec = b_frag.load().to(cutlass.Float32)

            acc = acc + a_val * b_vec

        # Store
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


# ---------------------------------------------------------------------------
# Host configuration
# ---------------------------------------------------------------------------
@cute.jit
def _matmul_Atrans_vecN_host(
    mA: cute.Tensor,  # (K, M)
    mB: cute.Tensor,  # (K, N)
    mC: cute.Tensor,  # (M, N)
    V : cutlass.Constexpr,          # vector width
    K : cutlass.Constexpr,          # reduction dim
):
    M = mC.shape[0]
    N = mC.shape[1]

    # Tile B and C along N with width V
    gBv = cute.zipped_divide(mB, (1, V))   # ((1,V), (K, N/V))
    gCv = cute.zipped_divide(mC, (1, V))   # ((1,V), (M, N/V))

    threads_per_block = 256
    N_groups = N // V
    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = M

    _matmul_Atrans_vecN_kernel(mA, gBv, gCv, cutlass.Int32(K)).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


# ---------------------------------------------------------------------------
# Torch-front model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    CuTe implementation of  C = A.T @ B

    Inputs:
      A : (K, M)  row-major
      B : (K, N)  row-major
    Output:
      C : (M, N)
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    # pick 128-bit vector width
    def _pick_vec_width(self, dtype: torch.dtype, N: int) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            V = 8     # 8×16 bit = 128 bit
        else:
            V = 4     # 4×32 bit = 128 bit
        while V > 1 and (N % V):
            V //= 2
        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        assert A.dim() == 2 and B.dim() == 2
        K, M = A.shape
        Kb, N = B.shape
        assert K == Kb, "dimension mismatch"

        # make contiguous CUDA tensors
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        V = self._pick_vec_width(A.dtype, N)
        C = torch.zeros((M, N), dtype=A.dtype, device=A.device)

        # Wrap for CuTe
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)     # (K, M) row-major
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)     # want row-major (M,N)
        )

        key = (A.dtype, V, K)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _matmul_Atrans_vecN_host, mA, mB, mC, V, K
            )

        # launch
        self._cache[key](mA, mB, mC)
        return C