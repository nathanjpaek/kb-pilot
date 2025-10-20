# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda().to(torch.float16)
        M, N = x.shape
        y = torch.empty_like(x)
        tk_kernels.dispatch_micro(x, y, int(M), int(N), float(self.alpha))
        return y