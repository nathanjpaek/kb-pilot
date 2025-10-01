import torch
import torch.nn as nn


class SENet(nn.Module):
    """support estimation network"""

    def __init__(self, input_size: 'int', hidden_size: 'int', output_dims:
        'int') ->None:
        super(SENet, self).__init__()
        self.l_1 = nn.Linear(input_size, hidden_size)
        self.l_2 = nn.Linear(hidden_size, output_dims)
        self.act = nn.Tanh()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.l_1(x)
        out = self.act(out)
        out = self.l_2(out)
        out = self.act(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_dims': 4}]
