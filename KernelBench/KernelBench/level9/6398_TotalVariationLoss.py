import torch
from typing import Optional


class TotalVariationLoss(torch.nn.Module):
    """
    Calculates the total variation loss of a tensor.
    """
    loss: 'Optional[torch.Tensor]'

    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, x):
        b, _c, h, w = x.shape
        a = torch.square(x[:, :, :h - 1, :w - 1] - x[:, :, 1:, :w - 1])
        b = torch.square(x[:, :, :h - 1, :w - 1] - x[:, :, :h - 1, 1:])
        self.loss = torch.mean(torch.pow(a + b, 1.25))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
