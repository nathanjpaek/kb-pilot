"""
Problem Name: 8_Matmul_with_irregular_shapes_
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=70.4 runtime_stats={'mean': 70.4, 'std': 1.97, 'min': 68.9, 'max': 82.1, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 6.42, 'std': 0.00624, 'min': 6.41, 'max': 6.46, 'num_trials': 100}, 'speedup_ratio': 0.0912}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _matmul_vecN_kernel(
    gA: cute.Tensor,      # (M, K) row-major
    gBv: cute.Tensor,     # ((1,V), (K, N/V))
    gCv: cute.Tensor,     # ((1,V), (M, N/V))
    K:  cutlass.Int32,    # reduction dimension
):
    # Thread/block indices
    tidx, _, _   = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _  = cute.arch.block_dim()

    mi = bidy                      # row index in C
    ng = bidx * bdimx + tidx       # group index along N (V-wide)

    # Tiled shapes: ((1,V), (M, N_groups))
    M = gCv.shape[1][0]
    N_groups = gCv.shape[1][1]

    if mi < M and ng < N_groups:
        # View of (1,V) output handled by this thread
        c_out = gCv[(None, (mi, ng))]

        # Make fragments
        b_frag = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()  # TensorSSA

        # K-loop
        for k in range(K):
            a_val = cutlass.Float32(gA[mi, k])

            b_vec_gmem = gBv[(None, (k, ng))]
            cute.autovec_copy(b_vec_gmem, b_frag)
            b_vec = b_frag.load().to(cutlass.Float32)

            acc = acc + a_val * b_vec

        # Store back in original dtype
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


@cute.jit
def _matmul_vecN_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    V:  cutlass.Constexpr,   # vector width
    K:  cutlass.Constexpr,   # reduction dimension
):
    M = mA.shape[0]
    N = mB.shape[1]

    # Tile along N with width V
    gBv = cute.zipped_divide(mB, (1, V))   # ((1,V), (K, N/V))
    gCv = cute.zipped_divide(mC, (1, V))   # ((1,V), (M, N/V))

    threads_per_block = 256
    N_groups = N // V
    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = M

    _matmul_vecN_kernel(mA, gBv, gCv, cutlass.Int32(K)).launch(
        grid=(grid_x, grid_y, 1),
        block=(threads_per_block, 1, 1),
    )


class ModelNew(torch.nn.Module):
    """
    CuTe vectorized matmul: C = A @ B
    Handles irregular shapes (e.g., M=8205, K=2949, N=5921). Vectorizes along N.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, N: int) -> int:
        # Prefer 128-bit vectors; fall back until divides N
        if dtype in (torch.float16, torch.bfloat16):
            V = 8
        elif dtype == torch.float32:
            V = 4
        else:
            V = 4
        while V > 1 and (N % V):
            V //= 2
        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA + contiguous
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.dim() == 2 and B.dim() == 2
        assert A.shape[1] == B.shape[0], "Incompatible shapes"

        M, K = A.shape
        _, N = B.shape

        V = self._pick_vector_width(A.dtype, N)

        # Output
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        # Wrap as CuTe tensors (row-major)
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        # Compile/cache per (dtype, V, K)
        key = (A.dtype, V, K)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _matmul_vecN_host, mA, mB, mC, V, K
            )

        # Launch
        self._cache[key](mA, mB, mC)
        return C


# Problem sizes (irregular)
M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed