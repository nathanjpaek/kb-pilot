import torch
import torch.nn as nn


class MAPELoss(nn.Module):

    def forward(self, estimation: 'torch.Tensor', target: 'torch.Tensor'):
        AER = torch.abs((target - estimation) / (target + 1e-10))
        MAPE = AER.mean() * 100
        return MAPE


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
