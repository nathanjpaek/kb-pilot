import torch
from torch import Tensor
from torch import nn


class MeanNormLoss(nn.Module):

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        size = [input.size(0), input.size(1), -1]
        input = input.view(*size)
        target = target.view(*size)
        diff = target - input
        loss = torch.norm(diff, dim=2)
        return torch.mean(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
