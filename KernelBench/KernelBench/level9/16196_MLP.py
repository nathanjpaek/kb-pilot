import torch
from torch import Tensor
from torch import nn


class MLP(nn.Module):

    def __init__(self, dim, hidden_dim, out_dim=None) ->None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = nn.ReLU6(True)
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.fc2(self.act(self.fc1(x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'hidden_dim': 4}]
