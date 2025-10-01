import torch
from torch import Tensor
from torch import nn


class RMSELoss(nn.Module):
    """ Root mean square error. """

    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(**kwargs)

    def forward(self, preds: 'Tensor', target: 'Tensor') ->Tensor:
        return torch.sqrt(self.mse(preds, target))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
