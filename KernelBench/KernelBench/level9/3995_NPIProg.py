import torch
import torch.nn as nn
import torch.nn.functional as F


class NPIProg(nn.Module):

    def __init__(self, input_dim: 'int', prog_key_dim: 'int', prog_num: 'int'):
        super(NPIProg, self).__init__()
        self._fcn1 = nn.Linear(in_features=input_dim, out_features=prog_key_dim
            )
        self._fcn2 = nn.Linear(in_features=prog_key_dim, out_features=prog_num)

    def forward(self, x):
        x = self._fcn1(x)
        x = self._fcn2(F.relu_(x))
        x = F.log_softmax(x.view(1, -1), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'prog_key_dim': 4, 'prog_num': 4}]
