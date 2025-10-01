import torch
from torch import nn


class Fusion3_MinusFCLayer(nn.Module):

    def __init__(self, input_dim):
        super(Fusion3_MinusFCLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 6, input_dim)

    def forward(self, input1, input2, input3):
        norm_input = self._norm_layer1(torch.cat([input1, input2, input3, 
            input1 - input2, input1 - input3, input2 - input3], dim=-1))
        return norm_input


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
