"""
Problem Name: 2_Standard_matrix_multiplication_
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=17.0 runtime_stats={'mean': 17.0, 'std': 0.0338, 'min': 16.9, 'max': 17.1, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.65, 'std': 0.0136, 'min': 2.64, 'max': 2.77, 'num_trials': 100}, 'speedup_ratio': 0.156}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def _matmul_vecN_kernel(
    gA: cute.Tensor,
    gBv: cute.Tensor,
    gCv: cute.Tensor,
    K: cutlass.Int32,
):
    # Thread and block indices
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    # Map this thread to output position
    mi = bidy
    ng = bidx * bdimx + tidx

    # Get dimensions from the tiled tensor shape ((1,V), (M, N_groups))
    M = gCv.shape[1][0]
    N_groups = gCv.shape[1][1]

    if mi < M and ng < N_groups:
        # Get the output location - has static shape (1,V) after slicing
        c_out = gCv[(None, (mi, ng))]
        
        # Create register fragment with SAME type as source for autovec_copy
        b_frag = cute.make_fragment_like(c_out, gA.element_type)
        
        # Initialize accumulator in Float32
        acc_frag_f32 = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_frag_f32.fill(0.0)
        acc = acc_frag_f32.load()
        
        # Accumulate over K dimension
        for k in range(K):
            # Load scalar from A[mi, k] and convert to Float32
            a_val = cutlass.Float32(gA[mi, k])
            
            # Load B vector with matching type, then convert to Float32
            b_vec_gmem = gBv[(None, (k, ng))]
            cute.autovec_copy(b_vec_gmem, b_frag)  # Same bit width
            b_vec = b_frag.load().to(cutlass.Float32)  # Convert TensorSSA
            
            # Accumulate: acc += a_val * b_vec
            acc = acc + a_val * b_vec

        # Store result - convert back to original type
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


@cute.jit
def _matmul_vecN_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    V: cutlass.Constexpr,
    K: cutlass.Constexpr,
):
    """
    Host-side configuration for vectorized matmul.
    V = vector width (compile-time constant)
    K = reduction dimension (compile-time constant)
    """
    M = mA.shape[0]
    N = mB.shape[1]

    # Tile B and C along N dimension with vector width V
    gBv = cute.zipped_divide(mB, (1, V))   # ((1,V), (K, N/V))
    gCv = cute.zipped_divide(mC, (1, V))   # ((1,V), (M, N/V))

    # Launch config: 256 threads per block
    threads_per_block = 256
    N_groups = N // V
    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = M

    _matmul_vecN_kernel(mA, gBv, gCv, cutlass.Int32(K)).launch(
        grid=(grid_x, grid_y, 1),
        block=(threads_per_block, 1, 1)
    )


class ModelNew(torch.nn.Module):
    """
    CuTe vectorized matmul: C = A @ B
    - Vectorizes along N (columns) for coalesced memory access
    - Each thread computes V adjacent output elements
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    def _pick_vector_width(self, dtype: torch.dtype, N: int) -> int:
        """
        Choose optimal vector width for 128-bit loads/stores:
        - FP32: 4 elements (128 bits)
        - FP16/BF16: 8 elements (128 bits)
        """
        if dtype in (torch.float16, torch.bfloat16):
            V = 8
        elif dtype == torch.float32:
            V = 4
        else:
            V = 4

        # Ensure N is divisible by V
        while V > 1 and (N % V != 0):
            V = V // 2

        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous CUDA tensors
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
        assert A.shape[1] == B.shape[0], f"Incompatible shapes: A{A.shape} @ B{B.shape}"

        M, K = A.shape
        _, N = B.shape

        # Choose vector width
        V = self._pick_vector_width(A.dtype, N)

        # Allocate output
        C = torch.zeros((M, N), dtype=A.dtype, device=A.device)

        # Wrap for CuTe - mark M dimension as dynamic
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        # Compile once per (dtype, V, K) configuration
        key = (A.dtype, V, K)
        if key not in self._cache:
            self._cache[key] = cute.compile(_matmul_vecN_host, mA, mB, mC, V, K)

        # Launch - only pass tensors
        self._cache[key](mA, mB, mC)
        
        return C