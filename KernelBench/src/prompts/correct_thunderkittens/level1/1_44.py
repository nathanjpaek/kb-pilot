# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda().to(torch.float16)
        B, C, I = x.shape
        O = (I + 2 * self.padding - self.kernel_size) // self.stride + 1
        y = torch.empty((B, C, O), dtype=x.dtype, device=x.device)

        tk_kernels.dispatch_micro(
            x, y,
            int(self.kernel_size),
            int(self.stride),
            int(self.padding),
            int(B), int(C), int(I)
        )
        return y