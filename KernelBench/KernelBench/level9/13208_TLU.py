import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class TLU(nn.Module):

    def __init__(self, num_features):
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(1, num_features, 1, 1),
            requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def forward(self, x):
        return torch.max(x, self.tau)

    def extra_repr(self):
        return '{num_features}'.format(**self.__dict__)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
