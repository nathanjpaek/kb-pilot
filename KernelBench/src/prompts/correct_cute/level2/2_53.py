"""
Problem Name: 53_Gemm_Scaling_Hardtanh_GELU
Generated using DSPy RAG with openai/o3
RAG Examples: 5
"""

import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ----------------------------------------------------------------------------
# 1) Device kernel
# ----------------------------------------------------------------------------
@cute.kernel
def _linear_scale_clamp_gelu_kernel(
    gA        : cute.Tensor,           # (B, K)
    gBTv      : cute.Tensor,           # ((1,V), (K, O/V))
    gBiasv    : cute.Tensor,           # ((1,V), (O/V))
    gCv       : cute.Tensor,           # ((1,V), (B, O/V))
    K         : cutlass.Int32,         # reduction length
    scale_val : cutlass.Float32,       # scaling factor
    clamp_lo  : cutlass.Float32,       # HardTanh lower bound
    clamp_hi  : cutlass.Float32,       # HardTanh upper bound
    s2p       : cutlass.Float32,       # √(2/π)
    k1        : cutlass.Float32,       # 0.044715
    half_val  : cutlass.Float32,       # 0.5
):
    # ------------------------------------------------------------------------
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi  = bidy                       # batch index
    og  = bidx * bdimx + tidx        # output-vector group (0 … O/V-1)

    B  = gCv.shape[1][0]
    Og = gCv.shape[1][1]             # O / V

    if (bi < B) and (og < Og):
        # --------------------------------------------------------------------
        # Logical views / fragments
        c_out = gCv[(None, (bi, og))]             # (1,V)

        b_frag     = cute.make_fragment_like(c_out, gA.element_type)
        bias_frag  = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()                 # TensorSSA(Float32,(1,V))

        # ---------------------------- GEMM ----------------------------------
        for k in range(K):
            a_val_f32  = cutlass.Float32(gA[bi, k])           # scalar
            b_vec_gmem = gBTv[(None, (k, og))]                # (1,V)
            cute.autovec_copy(b_vec_gmem, b_frag)             # gmem → regs
            b_vec_f32  = b_frag.load().to(cutlass.Float32)
            acc = acc + a_val_f32 * b_vec_f32                 # FMA accumulate

        # ---------------------- add bias ------------------------------------
        bias_gmem = gBiasv[(None, og)]
        cute.autovec_copy(bias_gmem, bias_frag)
        bias_f32  = bias_frag.load().to(cutlass.Float32)
        acc = acc + bias_f32

        # ---------------------- scale ---------------------------------------
        acc = acc * scale_val

        # ---------------------- HardTanh clamp ------------------------------
        zero = cutlass.Float32(0.0)            # for broadcasting
        lo_t = acc * zero + clamp_lo
        hi_t = acc * zero + clamp_hi
        acc  = cute.where(acc < lo_t, lo_t, acc)
        acc  = cute.where(acc > hi_t, hi_t, acc)

        # ---------------------- GELU (tanh approximation) -------------------
        # inner = s2p * (acc + k1 * acc^3)
        acc_sq   = acc * acc
        acc_cube = acc * acc_sq
        inner    = acc + k1 * acc_cube
        inner    = inner * s2p

        # tanh(inner) via exp: tanh(y) = (e^{2y} - 1)/(e^{2y} + 1)
        two   = cutlass.Float32(2.0)
        e2y   = cute.exp(inner * two)
        one_t = acc * zero + cutlass.Float32(1.0)
        tanh  = (e2y - one_t) / (e2y + one_t)

        gelu  = half_val * acc * (one_t + tanh)
        acc   = gelu

        # ---------------------- Store back ----------------------------------
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


# ----------------------------------------------------------------------------
# 2) Host wrapper
# ----------------------------------------------------------------------------
@cute.jit
def _linear_scale_clamp_gelu_host(
    mA        : cute.Tensor,            # (B, K)
    mBT       : cute.Tensor,            # (K, O)      – Wᵀ contiguous
    mBias     : cute.Tensor,            # (O)
    mC        : cute.Tensor,            # (B, O)
    V         : cutlass.Constexpr,      # vector width (compile-time)
    K         : cutlass.Constexpr,      # reduction dim (compile-time)
    scale_val : cutlass.Float32,        # scaling factor
    clamp_lo  : cutlass.Float32,        # HardTanh lo
    clamp_hi  : cutlass.Float32,        # HardTanh hi
    s2p       : cutlass.Float32,        # √(2/π)
    k1        : cutlass.Float32,        # 0.044715
    half_val  : cutlass.Float32,        # 0.5
    threads_per_block: cutlass.Int32,   # TPB (runtime)
):
    B, O = mA.shape[0], mBT.shape[1]

    # --- tile contiguous output dim ----------------------------------------
    gBTv   = cute.zipped_divide(mBT,  (1, V))   # ((1,V),(K,O/V))
    gBiasv = cute.zipped_divide(mBias, (V,))    # ((1,V),(O/V))
    gCv    = cute.zipped_divide(mC,    (1, V))  # ((1,V),(B,O/V))

    # --- launch geometry ----------------------------------------------------
    tpb = int(threads_per_block)
    O_groups          = O // V
    grid_x = cute.ceil_div(O_groups, tpb)
    grid_y = B

    _linear_scale_clamp_gelu_kernel(
        mA, gBTv, gBiasv, gCv,
        cutlass.Int32(K),
        scale_val,
        clamp_lo,
        clamp_hi,
        s2p,
        k1,
        half_val,
    ).launch(
        grid  = (grid_x, grid_y, 1),
        block = (tpb, 1, 1)
    )


# ----------------------------------------------------------------------------
# 3) PyTorch façade
# ----------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Fused CuTe implementation of:
        Y = GELU( clamp( (X @ Wᵀ + b) * scale , lo, hi) )
    Vectorised along the output dimension for 128-bit memory accesses.
    """

    def __init__(self,
                 in_features   : int,
                 out_features  : int,
                 scaling_factor: float,
                 hardtanh_min  : float,
                 hardtanh_max  : float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self.hardtanh_min   = float(hardtanh_min)
        self.hardtanh_max   = float(hardtanh_max)
        self._cache = {}
        self._best_cfg = {}

    # ---------------- helper: choose 128-bit vector width -------------------
    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, O: int) -> int:
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (O % V):
            V //= 2
        return V

    # Flush cache on dtype / device moves
    def _apply(self, fn):
        self._cache.clear()
        return super()._apply(fn)

    # ---------------- forward ------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Input must be 2-D (batch, in_features)"
        B, K = x.shape
        O, K_w = self.linear.weight.shape
        assert K_w == K, "in_features mismatch"

        # Ensure CUDA & contiguous, cast to parameter dtype
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)

        # Parameters ----------------------------------------------------------
        W   = self.linear.weight.detach().contiguous()
        b   = self.linear.bias.detach().contiguous()
        W_T = W.t().contiguous()

        # Vector width --------------------------------------------------------
        V = self._pick_vec_width(x.dtype, O)
        assert O % V == 0, "Vector width must divide out_features"

        # Output tensor -------------------------------------------------------
        y = torch.empty((B, O), dtype=x.dtype, device=x.device)

        # Wrap tensors for CuTe ----------------------------------------------
        mA    = from_dlpack(x,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBT   = from_dlpack(W_T, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(b,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC    = from_dlpack(y,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        # Constants -----------------------------------------------------------
        scale_f32 = cutlass.Float32(self.scaling_factor)
        lo_f32    = cutlass.Float32(self.hardtanh_min)
        hi_f32    = cutlass.Float32(self.hardtanh_max)
        s2p_f32   = cutlass.Float32(0.7978845608)   # √(2/π)
        k1_f32    = cutlass.Float32(0.044715)
        half_f32  = cutlass.Float32(0.5)

        # ---------------- Autotune (V, threads_per_block) -------------------
        tune_key = (x.dtype, K, O)
        if tune_key not in self._best_cfg:
            candidate_V = []
            base_V = self._pick_vec_width(x.dtype, O)
            v = base_V
            while v >= 1:
                if O % v == 0:
                    candidate_V.append(v)
                v //= 2
            candidate_V = list(dict.fromkeys(candidate_V))  # dedup
            tpb_list = [128, 256, 512]

            best_time = float("inf")
            best = (base_V, 256)

            # allocate one timing helper
            torch.cuda.synchronize()
            for v in candidate_V:
                key = (x.dtype, v, K, O)
                if key not in self._cache:
                    self._cache[key] = cute.compile(
                        _linear_scale_clamp_gelu_host,
                        mA, mBT, mBias, mC,
                        v,              # constexpr V
                        K,              # constexpr K
                        scale_f32,
                        lo_f32,
                        hi_f32,
                        s2p_f32,
                        k1_f32,
                        half_f32,
                        cutlass.Int32(256),  # placeholder, overridden at call
                    )
                fn = self._cache[key]
                for tpb in tpb_list:
                    # warmup
                    for _ in range(3):
                        fn(mA, mBT, mBias, mC,
                           scale_f32, lo_f32, hi_f32, s2p_f32, k1_f32, half_f32,
                           cutlass.Int32(tpb))
                    torch.cuda.synchronize()
                    # time few iters
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        fn(mA, mBT, mBias, mC,
                           scale_f32, lo_f32, hi_f32, s2p_f32, k1_f32, half_f32,
                           cutlass.Int32(tpb))
                    end.record()
                    torch.cuda.synchronize()
                    ms = start.elapsed_time(end) / 10.0
                    if ms < best_time:
                        best_time = ms
                        best = (v, tpb)
            self._best_cfg[tune_key] = best

        V_best, tpb_best = self._best_cfg[tune_key]
        key = (x.dtype, V_best, K, O)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _linear_scale_clamp_gelu_host,
                mA, mBT, mBias, mC,
                V_best,          # constexpr
                K,               # constexpr
                scale_f32,
                lo_f32,
                hi_f32,
                s2p_f32,
                k1_f32,
                half_f32,
                cutlass.Int32(tpb_best),
            )

        # Launch with tuned config -------------------------------------------
        self._cache[key](
            mA, mBT, mBias, mC,
            scale_f32,
            lo_f32,
            hi_f32,
            s2p_f32,
            k1_f32,
            half_f32,
            cutlass.Int32(tpb_best),
        )
        return y


# ----------------------------------------------------------------------------
# 4) Convenience variables for benchmark harness
# ----------------------------------------------------------------------------
batch_size    = 128
in_features   = 1024
out_features  = 512
scaling_factor= 0.5
hardtanh_min  = -2.0
hardtanh_max  =  2.0


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features,
            scaling_factor, hardtanh_min, hardtanh_max]


