import torch
import torch.nn as nn


class MaxOut(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int') ->None:
        super(MaxOut, self).__init__()
        self._ops_1 = nn.Linear(input_size, hidden_size)
        self._ops_2 = nn.Linear(input_size, hidden_size)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        feature_1 = self._ops_1(x)
        feature_2 = self._ops_2(x)
        return feature_1.max(feature_2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
