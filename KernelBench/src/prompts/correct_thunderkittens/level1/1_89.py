# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim  # only dim == 1 is supported

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.dim == 1, "ModelNew only supports dim == 1"
        x = x.contiguous().to(torch.float16).cuda()

        M, N = x.shape
        y = torch.empty_like(x)

        tk_kernels.dispatch_micro(x, y, int(M), int(N))
        return y