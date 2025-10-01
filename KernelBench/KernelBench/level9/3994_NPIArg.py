import torch
import torch.nn as nn
import torch.nn.functional as F


class NPIArg(nn.Module):

    def __init__(self, input_dim: 'int', arg_dim: 'int'):
        super(NPIArg, self).__init__()
        self.f_arg = nn.Linear(input_dim, arg_dim)

    def forward(self, x):
        x = self.f_arg(x)
        x = F.log_softmax(x.view(1, -1), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'arg_dim': 4}]
