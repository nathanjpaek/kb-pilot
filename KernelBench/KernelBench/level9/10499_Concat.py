import torch
from torch import nn
import torch.nn
import torch.optim


class Concat(nn.Module):

    def forward(self, state: 'torch.Tensor', action: 'torch.Tensor'):
        return torch.cat((state, action), dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
