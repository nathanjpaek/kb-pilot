# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2
        rows, cols = x.shape
        x = x.contiguous().cuda().to(torch.float16)
        y = torch.empty_like(x)
        tk_kernels.dispatch_micro(x, y, int(rows), int(cols))
        return y