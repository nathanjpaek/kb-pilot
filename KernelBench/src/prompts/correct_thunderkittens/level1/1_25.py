# model_new.py ---------------------------------------------------------------
# High-level PyTorch wrapper that drives the ThunderKittens CUDA micro-kernel.
# All math happens in tk_kernels.cu – this file
#   1. copies the user tensor to GPU / FP16,
#   2. allocates an identically-shaped output buffer,
#   3. launches the CUDA kernel.

import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()                    # Swish contains no trainable state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Materialise the tensor on the GPU in FP16 row-major form.
        x = x.contiguous().cuda().to(torch.float16)

        # (2) Extract matrix dimensions (M rows × N cols).
        M, N = x.shape

        # (3) Allocate the output tensor on the same device / dtype / shape.
        y = torch.empty_like(x)

        # (4) Invoke the micro-kernel.  All scalar arguments must be passed
        #     as vanilla Python ints so that pybind11 can convert them to C++.
        tk_kernels.dispatch_micro(x, y, int(M), int(N))

        return y