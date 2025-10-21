"""
Problem Name: 17_Matmul_with_transposed_B
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=17.2 runtime_stats={'mean': 17.2, 'std': 0.0608, 'min': 17.2, 'max': 17.4, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.74, 'std': 0.0076, 'min': 2.72, 'max': 2.76, 'num_trials': 100}, 'speedup_ratio': 0.159}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _matmul_Btrans_vecN_kernel(
    gA: cute.Tensor,   # (M, K)
    gBv: cute.Tensor,  # ((1,V), (K, N/V))  – tiled BT
    gCv: cute.Tensor,  # ((1,V), (M, N/V))
    K:  cutlass.Int32,
):
    # Thread/block indices
    tidx, _, _   = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _  = cute.arch.block_dim()

    mi = bidy                       # row index in C
    ng = bidx * bdimx + tidx        # N-vector group index

    M        = gCv.shape[1][0]
    N_groups = gCv.shape[1][1]

    if mi < M and ng < N_groups:
        # Output slice for this thread: shape (1,V)
        c_out = gCv[(None, (mi, ng))]

        # Register fragments
        b_frag = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32 = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32.fill(0.0)
        acc = acc_f32.load()

        # Reduction over K
        for k in range(K):
            a_val = cutlass.Float32(gA[mi, k])

            b_vec_gmem = gBv[(None, (k, ng))]
            cute.autovec_copy(b_vec_gmem, b_frag)         # gmem → regs
            b_vec = b_frag.load().to(cutlass.Float32)     # to FP32

            acc = acc + a_val * b_vec

        # Store back in original dtype
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


@cute.jit
def _matmul_Btrans_vecN_host(
    mA: cute.Tensor,   # (M, K)
    mBT: cute.Tensor,  # (K, N)  — contiguous transpose of B
    mC: cute.Tensor,   # (M, N)
    V:  cutlass.Constexpr,
    K:  cutlass.Constexpr,
):
    M = mA.shape[0]
    N = mBT.shape[1]

    # Tile along N with vector width V
    gBv = cute.zipped_divide(mBT, (1, V))  # ((1,V),(K,N/V))
    gCv = cute.zipped_divide(mC,  (1, V))  # ((1,V),(M,N/V))

    threads_per_block = 256
    N_groups = N // V

    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = M

    _matmul_Btrans_vecN_kernel(mA, gBv, gCv, cutlass.Int32(K)).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


class ModelNew(torch.nn.Module):
    """
    CuTe implementation of C = A @ B.T
    Inputs:
      A : (M, K)
      B : (N, K)
    Output:
      C : (M, N)
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, N: int) -> int:
        # Prefer 128-bit vector transactions
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8 * 16-bit
        else:
            V = 4   # 4 * 32-bit
        while V > 1 and (N % V):
            V //= 2
        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA & contiguous
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.dim() == 2 and B.dim() == 2
        M, K = A.shape
        N_in, Kb = B.shape
        assert K == Kb, "Incompatible dimensions for A @ B.T"
        N = N_in

        # Make contiguous BT for coalesced vectorized access along N
        BT = B.t().contiguous()

        V = self._pick_vector_width(A.dtype, N)
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        # Wrap tensors for CuTe (row-major)
        mA  = from_dlpack(A,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBT = from_dlpack(BT, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC  = from_dlpack(C,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (A.dtype, V, K)
        if key not in self._cache:
            self._cache[key] = cute.compile(_matmul_Btrans_vecN_host, mA, mBT, mC, V, K)

        # Launch compiled kernel
        self._cache[key](mA, mBT, mC)
        return C


# Convenience sizes for harness parity with the original snippet
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(N, K)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed