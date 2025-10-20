"""
Problem Name: 18_Matmul_with_transposed_both
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=17.3 runtime_stats={'mean': 17.3, 'std': 0.0582, 'min': 17.3, 'max': 17.7, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.79, 'std': 0.0103, 'min': 2.78, 'max': 2.83, 'num_trials': 100}, 'speedup_ratio': 0.161}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------
#  Device kernel: each thread computes V adjacent C elements (row-major)
# ---------------------------------------------------------------------
@cute.kernel
def _matmul_vecN_kernel(
    gA: cute.Tensor,      # (M, K)   – row-major scalars
    gBv: cute.Tensor,     # ((1,V), (K, N/V)) – vector-tiled B
    gCv: cute.Tensor,     # ((1,V), (M, N/V)) – vector-tiled C
    K: cutlass.Int32,     # reduction length
):
    # CUDA indices
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    # Logical output coordinates
    mi  = bidy                           # row in C
    ng  = bidx * bdimx + tidx            # vector group along N

    M         = gCv.shape[1][0]
    N_groups  = gCv.shape[1][1]

    if mi < M and ng < N_groups:
        # Tensor view for the V-wide output slice handled by this thread
        c_out = gCv[(None, (mi, ng))]          # shape (1,V)

        # Convenience fragments
        b_frag  = cute.make_fragment_like(c_out, gA.element_type)
        acc_f32 = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32.fill(0.0)
        acc = acc_f32.load()                   # TensorSSA(Float32, (1,V))

        # Main reduction
        for k in range(K):
            a_val = cutlass.Float32(gA[mi, k])     # scalar A[mi,k] → f32

            b_vec_gmem = gBv[(None, (k, ng))]      # vector slice of B
            cute.autovec_copy(b_vec_gmem, b_frag)  # gmem → registers
            b_vec = b_frag.load().to(cutlass.Float32)

            acc = acc + a_val * b_vec              # FMA in f32

        # Write back in original dtype
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


# ---------------------------------------------------------------------
#  Host wrapper: builds tiled layouts and launches the kernel
# ---------------------------------------------------------------------
@cute.jit
def _matmul_vecN_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    V : cutlass.Constexpr,    # compile-time vector width
    K : cutlass.Constexpr,    # compile-time K
):
    M = mA.shape[0]
    N = mB.shape[1]

    # Tile B and C along the contiguous N dimension
    gBv = cute.zipped_divide(mB, (1, V))   # ((1,V), (K, N/V))
    gCv = cute.zipped_divide(mC, (1, V))   # ((1,V), (M, N/V))

    threads_per_block = 256
    N_groups          = N // V

    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = M

    _matmul_vecN_kernel(mA, gBv, gCv, cutlass.Int32(K)).launch(
        grid  =(grid_x, grid_y, 1),
        block =(threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------
#  PyTorch front-end module
# ---------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe implementation of C = (A.T) @ (B.T) with
    - vectorised access along N for coalesced memory
    - 128-bit global-memory transactions
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    # Pick vector width (128-bit loads/stores)
    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, N: int) -> int:
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (N % V):
            V //= 2
        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Logical matrices after the transposes required by the original model
        AT = A.t().contiguous().cuda() if not A.is_cuda else A.t().contiguous()
        BT = B.t().contiguous().cuda() if not B.is_cuda else B.t().contiguous()

        M, K = AT.shape
        _, N = BT.shape

        V = self._pick_vector_width(AT.dtype, N)

        C = torch.empty((M, N), dtype=AT.dtype, device=AT.device)

        # Wrap into CuTe tensors (row-major, leading stride in dim-1)
        mA = from_dlpack(AT, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mB = from_dlpack(BT, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C,  assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (AT.dtype, V, K)
        if key not in self._cache:
            # One-time JIT compilation for this (dtype, V, K)
            self._cache[key] = cute.compile(_matmul_vecN_host, mA, mB, mC, V, K)

        # Launch compiled kernel
        self._cache[key](mA, mB, mC)
        return C


# ---------------------------------------------------------------------
#  Convenience helpers for the benchmark harness
# ---------------------------------------------------------------------
M = 1024 * 2        # 2048
K = 4096 * 2        # 8192
N = 2048 * 2        # 4096


def get_inputs():
    # Original shapes (K, M) and (N, K) as in the reference code
    A = torch.rand(K, M)
    B = torch.rand(N, K)
    return [A, B]


def get_init_inputs():
    return []  # No special initialisation required