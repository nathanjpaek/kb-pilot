import torch
from torch import nn


class WDLoss(nn.Module):

    def __init__(self, _lambda):
        super(WDLoss, self).__init__()
        self._lambda = _lambda

    def forward(self, t_x, t_y, t_z):
        return -(torch.mean(t_x) - torch.mean(t_y) - self._lambda * torch.
            mean((torch.norm(t_z, dim=1) - 1).pow(2)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'_lambda': 4}]
