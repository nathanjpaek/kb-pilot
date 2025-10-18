"""
Problem Name: 13_Matmul_for_symmetric_matrices
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=16.3 runtime_stats={'mean': 16.3, 'std': 0.0123, 'min': 16.2, 'max': 16.3, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.67, 'std': 0.00167, 'min': 2.67, 'max': 2.68, 'num_trials': 100}, 'speedup_ratio': 0.164}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# ──────────────────────────────────────────────────────────────────────────────
# Device kernel: one thread computes a V-wide vector of C
# ──────────────────────────────────────────────────────────────────────────────
@cute.kernel
def _sym_matmul_vecN_kernel(
    gA: cute.Tensor,     # (N, K)   – plain row-major
    gBv: cute.Tensor,    # ((1,V),(K,N/V))
    gCv: cute.Tensor,    # ((1,V),(M,N/V))
    K:  cutlass.Int32,   # reduction dimension
):
    # CUDA indices
    tidx, _, _   = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _  = cute.arch.block_dim()

    mi = bidy                            # row of C handled by this block
    ng = bidx * bdimx + tidx             # vector group along N handled by this thread

    M        = gCv.shape[1][0]
    N_groups = gCv.shape[1][1]

    if mi < M and ng < N_groups:
        # Logical (1,V) slice of the output tile
        c_view = gCv[(None, (mi, ng))]

        # Temporary register fragments
        b_frag   = cute.make_fragment_like(c_view, gA.element_type)
        acc_frag = cute.make_fragment_like(c_view, cutlass.Float32)
        acc_frag.fill(0.0)
        acc = acc_frag.load()

        # Reduction over K
        for k in range(K):
            a_val = cutlass.Float32(gA[mi, k])

            b_vec_mem = gBv[(None, (k, ng))]
            cute.autovec_copy(b_vec_mem, b_frag)          # gmem → reg (same bit width)
            b_vec = b_frag.load().to(cutlass.Float32)

            acc = acc + a_val * b_vec

        # Write back – convert to original dtype
        out_frag = cute.make_fragment_like(c_view, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_view)


# ──────────────────────────────────────────────────────────────────────────────
# Host JIT wrapper
# ──────────────────────────────────────────────────────────────────────────────
@cute.jit
def _sym_matmul_vecN_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    V:  cutlass.Constexpr,   # vector width
    K:  cutlass.Constexpr,   # reduction dim  (= N)
):
    M = mA.shape[0]
    N = mB.shape[1]

    # Tile along N with width V
    gBv = cute.zipped_divide(mB, (1, V))
    gCv = cute.zipped_divide(mC, (1, V))

    threads_per_block = 256
    N_groups = N // V
    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = M

    _sym_matmul_vecN_kernel(mA, gBv, gCv, cutlass.Int32(K)).launch(
        grid=(grid_x, grid_y, 1),
        block=(threads_per_block, 1, 1)
    )


# ──────────────────────────────────────────────────────────────────────────────
# nn.Module wrapper – drop-in replacement for the original PyTorch model
# ──────────────────────────────────────────────────────────────────────────────
class ModelNew(torch.nn.Module):
    """
    Vectorised CuTe implementation of C = A @ B with symmetric inputs.
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    # Choose a 128-bit-aligned vector width that divides N
    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, N: int) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            V = 8          # 8 × 16-bit  = 128 bit
        else:              # e.g. float32
            V = 4          # 4 × 32-bit  = 128 bit
        while V > 1 and (N % V != 0):
            V //= 2
        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA, contiguous
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.shape == B.shape and A.dim() == 2, "Inputs must be square 2-D tensors"
        N = A.shape[0]

        V = self._pick_vector_width(A.dtype, N)
        C = torch.zeros_like(A)

        # Wrap PyTorch tensors into CuTe tensors
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (A.dtype, V, N)
        if key not in self._cache:
            # Compile once per configuration
            self._cache[key] = cute.compile(_sym_matmul_vecN_host, mA, mB, mC, V, N)

        # Run the kernel
        self._cache[key](mA, mB, mC)

        return C