# model_new.py
import torch
import torch.nn.functional as F
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else self.kernel_size
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda()
        y = F.avg_pool3d(
            x,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding
        ).contiguous()
        y_flat = y.view(1, 1, 1, y.numel()).contiguous()
        tk_kernels.dispatch_micro(y_flat, int(y.numel()))
        return y