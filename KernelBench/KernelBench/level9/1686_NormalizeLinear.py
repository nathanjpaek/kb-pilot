import torch
from torch.nn import functional as F
from torch import nn
import torch.utils.data
import torch.utils.data.distributed


class NormalizeLinear(nn.Module):

    def __init__(self, act_dim, k_value):
        super().__init__()
        self.lin = nn.Linear(act_dim, k_value)

    def normalize(self):
        self.lin.weight.data = F.normalize(self.lin.weight.data, p=2, dim=1)

    def forward(self, x):
        self.normalize()
        return self.lin(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'act_dim': 4, 'k_value': 4}]
