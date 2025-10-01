import torch
from torch import Tensor
from torch import nn


class Attention(nn.Module):

    def forward(self, selected_input: 'Tensor', attention: 'Tensor'):
        attended_input = selected_input * attention.unsqueeze(-1)
        return attended_input


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
