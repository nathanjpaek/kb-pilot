"""
Problem Name: 98_KLDivLoss
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.43 runtime_stats={'mean': 1.43, 'std': 0.00386, 'min': 1.43, 'max': 1.45, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 3.85, 'std': 0.00401, 'min': 3.85, 'max': 3.87, 'num_trials': 100}, 'speedup_ratio': 2.69}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel: compute elementwise KL term t * (log(t) - log(p))
# Inputs/Outputs are vector-tiled along D for 128-bit transactions
# ---------------------------------------------------------------------------
@cute.kernel
def _kl_elem_vec_kernel(
    gPv: cute.Tensor,  # ((1,V), (B, D/V))  predictions (probabilities)
    gTv: cute.Tensor,  # ((1,V), (B, D/V))  targets (probabilities)
    gLv: cute.Tensor,  # ((1,V), (B, D/V))  output per-element loss
):
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    b  = bidy                     # batch index
    dg = bidx * bdimx + tidx      # vector group along D

    B   = gLv.shape[1][0]         # batch size
    Dg  = gLv.shape[1][1]         # D / V

    if (b < B) and (dg < Dg):
        # Vector views
        p_vec_gmem = gPv[(None, (b, dg))]   # shape (1,V)
        t_vec_gmem = gTv[(None, (b, dg))]   # shape (1,V)
        l_vec_gmem = gLv[(None, (b, dg))]   # shape (1,V)

        # Fragments for vectorized copies
        p_frag = cute.make_fragment_like(p_vec_gmem, gPv.element_type)
        t_frag = cute.make_fragment_like(t_vec_gmem, gTv.element_type)

        cute.autovec_copy(p_vec_gmem, p_frag)  # gmem → regs
        cute.autovec_copy(t_vec_gmem, t_frag)  # gmem → regs

        # Convert to FP32 for math
        p_vec = p_frag.load().to(cutlass.Float32)  # TensorSSA (1,V)
        t_vec = t_frag.load().to(cutlass.Float32)  # TensorSSA (1,V)

        eps   = cutlass.Float32(1e-12)
        logp  = cute.log(p_vec + eps)
        logt  = cute.log(t_vec + eps)
        loss_vec = t_vec * (logt - logp)

        # Store back in original dtype
        out_frag = cute.make_fragment_like(l_vec_gmem, gPv.element_type)
        out_frag.store(loss_vec.to(gPv.element_type))
        cute.autovec_copy(out_frag, l_vec_gmem)


# ---------------------------------------------------------------------------
# Host wrapper: tile along D and launch
# ---------------------------------------------------------------------------
@cute.jit
def _kl_elem_vec_host(
    mP: cute.Tensor,         # (B, D)
    mT: cute.Tensor,         # (B, D)
    mL: cute.Tensor,         # (B, D)
    V : cutlass.Constexpr,   # vector width (compile-time)
):
    B, D = mP.shape

    # Tile along contiguous D with width V
    gPv = cute.zipped_divide(mP, (1, V))   # ((1,V), (B, D/V))
    gTv = cute.zipped_divide(mT, (1, V))   # ((1,V), (B, D/V))
    gLv = cute.zipped_divide(mL, (1, V))   # ((1,V), (B, D/V))

    threads_per_block = 256
    D_groups = D // V

    grid_x = cute.ceil_div(D_groups, threads_per_block)
    grid_y = B

    _kl_elem_vec_kernel(gPv, gTv, gLv).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
# PyTorch front-end module
# ---------------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated KLDivLoss with reduction='batchmean' for inputs (B, D).
    Computes per-element term t * (log(t) - log(p)) vectorized along D,
    then reduces on host as sum()/B to match PyTorch semantics.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, D: int) -> int:
        # Prefer 128-bit vectors
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8 x 16-bit
        else:
            V = 4   # 4 x 32-bit
        while V > 1 and (D % V):
            V //= 2
        return V

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA & contiguous
        P = predictions.contiguous().cuda() if not predictions.is_cuda else predictions.contiguous()
        T = targets.contiguous().cuda() if not targets.is_cuda else targets.contiguous()

        assert P.shape == T.shape and P.dim() == 2, "Expected (B, D) tensors with matching shapes"
        B, D = P.shape

        V = self._pick_vector_width(P.dtype, D)

        # Output elementwise loss buffer
        L = torch.empty_like(P)

        # Wrap tensors for CuTe (row-major)
        mP = from_dlpack(P, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mT = from_dlpack(T, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mL = from_dlpack(L, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (P.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(_kl_elem_vec_host, mP, mT, mL, V)

        # Launch kernel: computes per-element terms
        self._cache[key](mP, mT, mL)

        # Batchmean reduction: sum over all elements divided by batch size
        loss = L.sum() / B
        return loss


# Optional helpers mirroring original harness
batch_size = 8192 * 2
input_shape = (8192 * 2,)

def get_inputs():
    scale = torch.rand(())
    preds = (torch.rand(batch_size, *input_shape) * scale).softmax(dim=-1)
    targs = torch.rand(batch_size, *input_shape).softmax(dim=-1)
    return [preds, targs]

def get_init_inputs():
    return []