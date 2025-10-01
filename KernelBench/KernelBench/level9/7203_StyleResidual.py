import torch
from torch import nn
import torch.utils.data
import torch.optim


class StyleResidual(nn.Module):
    """Styling."""

    def __init__(self, d_channel: 'int', d_style: 'int', kernel_size: 'int'=1):
        super().__init__()
        self.rs = nn.Conv1d(in_channels=d_style, out_channels=d_channel,
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: 'torch.Tensor', s: 'torch.Tensor') ->torch.Tensor:
        """`x`: [B,C,T], `s`: [B,S,T] => [B,C,T]."""
        return x + self.rs(s)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_channel': 4, 'd_style': 4}]
