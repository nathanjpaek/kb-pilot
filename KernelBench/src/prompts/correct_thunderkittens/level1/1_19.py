# model_new.py
import torch
import torch.nn as nn
import tk_kernels


class ModelNew(nn.Module):
    """
    ThunderKittens-accelerated ReLU
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda().to(torch.float16)
        M, N = x.shape
        y = torch.zeros_like(x)
        tk_kernels.dispatch_micro(x, y, int(M), int(N))
        return y