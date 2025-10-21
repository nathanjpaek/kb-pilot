"""
Problem Name: 96_HuberLoss
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=5.57 runtime_stats={'mean': 5.57, 'std': 0.0414, 'min': 5.55, 'max': 5.87, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 5.54, 'std': 0.00848, 'min': 5.53, 'max': 5.6, 'num_trials': 100}, 'speedup_ratio': 0.995}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel: vectorized Smooth L1 (Huber) per-element loss
# loss(d) = 0.5 * d^2 / beta    if |d| < beta
#         = |d| - 0.5 * beta    otherwise
# where d = prediction - target
# ---------------------------------------------------------------------------
@cute.kernel
def _smooth_l1_vec_kernel(
    gPv: cute.Tensor,               # ((1,V), (B, D/V)) - predictions
    gTv: cute.Tensor,               # ((1,V), (B, D/V)) - targets
    gLv: cute.Tensor,               # ((1,V), (B, D/V)) - loss output
    beta: cutlass.Float32,          # runtime beta (typically 1.0)
):
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi = bidy                          # batch index
    dg = bidx * bdimx + tidx           # vector group index along D/V

    B  = gPv.shape[1][0]
    Dg = gPv.shape[1][1]               # D / V

    if bi < B and dg < Dg:
        p_view = gPv[(None, (bi, dg))]     # (1,V)
        t_view = gTv[(None, (bi, dg))]     # (1,V)
        l_view = gLv[(None, (bi, dg))]     # (1,V)

        # Fragments for vector copy
        p_frag = cute.make_fragment_like(p_view, gPv.element_type)
        t_frag = cute.make_fragment_like(t_view, gTv.element_type)
        l_frag = cute.make_fragment_like(l_view, gLv.element_type)

        # Load predictions and targets -> FP32 TensorSSA
        cute.autovec_copy(p_view, p_frag)
        cute.autovec_copy(t_view, t_frag)
        p_vec = p_frag.load().to(cutlass.Float32)
        t_vec = t_frag.load().to(cutlass.Float32)

        # Smooth L1 computation in FP32
        d      = p_vec - t_vec
        zero   = cutlass.Float32(0.0)
        half   = cutlass.Float32(0.5)
        abs_d  = cute.where(d >= zero, d, -d)
        small  = half * d * d / beta                  # 0.5 * d^2 / beta
        large  = abs_d - half * beta                 # |d| - 0.5 * beta
        loss_v = cute.where(abs_d < beta, small, large)

        # Store back to original dtype
        l_frag.store(loss_v.to(gLv.element_type))
        cute.autovec_copy(l_frag, l_view)


# ---------------------------------------------------------------------------
# Host configuration: tiling + launch
# ---------------------------------------------------------------------------
@cute.jit
def _smooth_l1_vec_host(
    mP: cute.Tensor,                 # (B, D) predictions
    mT: cute.Tensor,                 # (B, D) targets
    mL: cute.Tensor,                 # (B, D) loss
    V : cutlass.Constexpr,           # vector width (compile-time)
    beta: cutlass.Float32,           # runtime beta
):
    B, D = mP.shape

    # Vectorize along contiguous D with width V
    gPv = cute.zipped_divide(mP, (1, V))   # ((1,V), (B, D/V))
    gTv = cute.zipped_divide(mT, (1, V))
    gLv = cute.zipped_divide(mL, (1, V))

    threads_per_block = 256
    D_groups = D // V

    grid_x = cute.ceil_div(D_groups, threads_per_block)
    grid_y = B

    _smooth_l1_vec_kernel(gPv, gTv, gLv, beta).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
# PyTorch front-end
# ---------------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated Smooth L1 (Huber) loss.
    Computes per-element loss with vectorized I/O, then returns mean reduction
    to match torch.nn.functional.smooth_l1_loss default behavior.
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = float(beta)
        self._cache = {}

    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, D: int) -> int:
        # Prefer 128-bit transactions
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8×16-bit = 128-bit
        elif dtype == torch.float32:
            V = 4   # 4×32-bit = 128-bit
        else:
            V = 2   # fallback (e.g., float64)
        while V > 1 and (D % V):
            V //= 2
        return V

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure 2-D tensors (B, D); make contiguous and on CUDA
        preds = predictions.contiguous().cuda() if not predictions.is_cuda else predictions.contiguous()
        targs = targets.contiguous().cuda() if not targets.is_cuda else targets.contiguous()
        assert preds.dim() == 2 and targs.dim() == 2, "Expected 2-D (B, D) tensors"
        assert preds.shape == targs.shape, "predictions and targets must have the same shape"
        B, D = preds.shape

        V = self._pick_vec_width(preds.dtype, D)

        # Output tensor for per-element loss
        loss_elem = torch.empty_like(preds)

        # Wrap for CuTe (row-major (B, D))
        mP = from_dlpack(preds, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mT = from_dlpack(targs, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mL = from_dlpack(loss_elem, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (preds.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _smooth_l1_vec_host, mP, mT, mL, V, cutlass.Float32(self.beta)
            )

        # Launch kernel (beta is runtime)
        self._cache[key](mP, mT, mL, cutlass.Float32(self.beta))

        # PyTorch reduction to match default smooth_l1_loss reduction='mean'
        return loss_elem.mean()


# Example harness consistent with the original snippet
batch_size = 32768
input_shape = (32768,)

def get_inputs():
    scale = torch.rand(())
    predictions = torch.rand(batch_size, *input_shape) * scale
    targets     = torch.rand(batch_size, *input_shape)
    return [predictions, targets]

def get_init_inputs():
    return []