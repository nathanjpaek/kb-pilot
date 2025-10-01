import torch
import torch.nn as nn


class L2Norm(nn.Module):
    """l2-normalization as layer. """

    def __init__(self, *, eps: float=1e-10) ->None:
        super().__init__()
        self.eps = eps

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        norm = torch.sqrt(torch.sum(x * x, dim=-1) + self.eps)
        x = x / norm.unsqueeze(-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
