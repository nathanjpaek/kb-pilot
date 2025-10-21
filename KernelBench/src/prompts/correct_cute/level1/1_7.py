"""
Problem Name: 7_Matmul_with_small_K_dimension_
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=34.2 runtime_stats={'mean': 34.2, 'std': 0.0161, 'min': 34.2, 'max': 34.2, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.08, 'std': 0.00589, 'min': 4.07, 'max': 4.1, 'num_trials': 100}, 'speedup_ratio': 0.119}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _matmul_smallK_vecN_kernel(
    gA: cute.Tensor,    # (M, K)
    gBv: cute.Tensor,   # ((1,V), (K, N/V))
    gCv: cute.Tensor,   # ((1,V), (M, N/V))
    K:  cutlass.Int32,  # reduction length
):
    # Thread/block indices
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    # Map thread to output position
    mi = bidy                          # row in C
    ng = bidx * bdimx + tidx           # vector group along N

    # Dimensions from tiled shapes
    M = gCv.shape[1][0]
    N_groups = gCv.shape[1][1]

    if mi < M and ng < N_groups:
        # Output view for this thread: static (1,V)
        c_out = gCv[(None, (mi, ng))]

        # Fragments and accumulator
        b_frag = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32 = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32.fill(0.0)
        acc = acc_f32.load()  # TensorSSA(Float32, (1,V))

        # Reduction over K
        for k in range(K):
            a_val = cutlass.Float32(gA[mi, k])

            b_vec_gmem = gBv[(None, (k, ng))]
            cute.autovec_copy(b_vec_gmem, b_frag)
            b_vec = b_frag.load().to(cutlass.Float32)

            acc = acc + a_val * b_vec

        # Store back to gmem with original dtype
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


@cute.jit
def _matmul_smallK_vecN_host(
    mA: cute.Tensor,    # (M, K)
    mB: cute.Tensor,    # (K, N)
    mC: cute.Tensor,    # (M, N)
    V:  cutlass.Constexpr,  # vector width
    K:  cutlass.Constexpr,  # reduction length
):
    M = mA.shape[0]
    N = mB.shape[1]

    # Tile B and C along N with width V
    gBv = cute.zipped_divide(mB, (1, V))   # ((1,V), (K, N/V))
    gCv = cute.zipped_divide(mC, (1, V))   # ((1,V), (M, N/V))

    threads_per_block = 256
    N_groups = N // V

    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = M

    _matmul_smallK_vecN_kernel(mA, gBv, gCv, cutlass.Int32(K)).launch(
        grid=(grid_x, grid_y, 1),
        block=(threads_per_block, 1, 1),
    )


class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated matmul for small K: C = A @ B
      A: (M, K), B: (K, N), C: (M, N)
    Vectorizes along N to use 128-bit transactions.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, N: int) -> int:
        # Prefer 128-bit vectors
        if dtype in (torch.float16, torch.bfloat16):
            V = 8    # 8 * 16-bit
        else:
            V = 4    # 4 * 32-bit
        while V > 1 and (N % V != 0):
            V //= 2
        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA & contiguous
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, "Incompatible shapes"

        V = self._pick_vector_width(A.dtype, N)

        # Output
        C = torch.zeros((M, N), dtype=A.dtype, device=A.device)

        # Wrap tensors for CuTe as row-major with dynamic leading dims
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (A.dtype, V, K)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _matmul_smallK_vecN_host, mA, mB, mC, V, K
            )

        # Launch compiled kernel
        self._cache[key](mA, mB, mC)
        return C


# Optional: benchmark harness alignment with the original definitions
M = 16384 * 2
N = 16384 * 2
K = 32 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed