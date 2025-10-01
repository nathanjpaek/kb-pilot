import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class LocalSnrLoss(nn.Module):

    def __init__(self, factor: 'float'=1):
        super().__init__()
        self.factor = factor

    def forward(self, input: 'Tensor', target_lsnr: 'Tensor'):
        input = input.squeeze(-1)
        return F.mse_loss(input, target_lsnr) * self.factor


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
