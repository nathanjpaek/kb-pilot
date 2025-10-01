import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as F


def hard_swish(x: 'torch.Tensor', inplace: 'bool'=False):
    """Hard swish."""
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):
    """Custom hardswish to work with onnx."""

    def __init__(self, inplace: 'bool'=False):
        """Initialize."""
        super().__init__()
        self.inplace = inplace

    def forward(self, x: 'torch.Tensor'):
        """Forward."""
        return hard_swish(x, self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
