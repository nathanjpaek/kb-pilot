import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def maxpool3d_kernel(
    N: int,
    C: int,
    Di: int,
    Hi: int,
    Wi: int,
    K: int,
    S: int,
    P: int,
    DIL: int,
    dtype: str = "float16",
):
    Do = (Di + 2 * P - DIL * (K - 1) - 1) // S + 1
    Ho = (Hi + 2 * P - DIL * (K - 1) - 1) // S + 1
    Wo = (Wi + 2 * P - DIL * (K - 1) - 1) // S + 1

    block_O = 128
    grid_x = T.ceildiv(Do * Ho * Wo, block_O)
    grid_y = N * C

    @T.prim_func
    def main(
        inp: T.Tensor((N, C, Di, Hi, Wi), dtype),
        out: T.Tensor((N, C, Do, Ho, Wo), dtype),
    ):
        with T.Kernel(grid_x, grid_y, threads=block_O) as (bx, by):
            tx = T.get_thread_binding(0)

            lin_idx = bx * block_O + tx
            if lin_idx >= Do * Ho * Wo:
                return

            n = by // C
            c = by % C

            od = lin_idx // (Ho * Wo)
            oh = (lin_idx % (Ho * Wo)) // Wo
            ow = lin_idx % Wo

            max_val_buf = T.alloc_fragment((1,), dtype)
            max_val_buf[0] = -T.infinity(dtype)

            for kd in range(K):
                id_ = od * S - P + kd * DIL
                if 0 <= id_ < Di:
                    for kh in range(K):
                        ih = oh * S - P + kh * DIL
                        if 0 <= ih < Hi:
                            for kw in range(K):
                                iw = ow * S - P + kw * DIL
                                if 0 <= iw < Wi:
                                    val = inp[n, c, id_, ih, iw]
                                    max_val_buf[0] = T.max(max_val_buf[0], val)

            out[n, c, od, oh, ow] = max_val_buf[0]

    return main


class ModelNew(nn.Module):
    """
    Optimized MaxPool3d model using TileLang.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super(ModelNew, self).__init__()
        if return_indices or ceil_mode:
            raise NotImplementedError("return_indices and ceil_mode are not supported.")
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self._kernel_cache = {}

    def _get_kernel(self, shape, dtype="float16"):
        N, C, Di, Hi, Wi = shape
        key = (N, C, Di, Hi, Wi, dtype)
        if key not in self._kernel_cache:
            prog = maxpool3d_kernel(
                N,
                C,
                Di,
                Hi,
                Wi,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                dtype=dtype,
            )
            self._kernel_cache[key] = tilelang.jit(out_idx=-1)(prog)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16).contiguous()
        kernel = self._get_kernel(x.shape, dtype="float16")
        y = kernel(x)
        return y.to(torch.float32)