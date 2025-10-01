import torch
from torch import nn


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, init_scale=1.0,
        no_weight_init=False):
        super(GRUCell, self).__init__()
        self.recurrent = nn.GRUCell(input_size, hidden_size)
        if not no_weight_init:
            for name, param in self.recurrent.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, init_scale)
                    weight_inited = True
                elif 'bias' in name:
                    with torch.no_grad():
                        param.zero_()
            assert weight_inited

    def forward(self, x, h=None):
        return self.recurrent(x, h)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
