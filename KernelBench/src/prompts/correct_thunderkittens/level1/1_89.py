# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)  # expected to be 1 (columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.dim == 1 and x.dim() == 2
        x = x.contiguous().cuda().to(torch.float16)
        M, N = x.shape
        y = torch.empty_like(x)
        tk_kernels.dispatch_micro(x, y, int(M), int(N))
        return y